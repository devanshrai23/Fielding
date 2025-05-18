# -*- coding: utf-8 -*-
import collections
import gc
import pickle
import random
import time
from argparse import Namespace

import numpy as np
import torch
import wandb

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.logger.executor_logging as logger
from fedscale.cloud.channels.channel_context import ClientConnections
from fedscale.cloud.execution.tensorflow_client import TensorflowClient
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.data_processor import collate, voice_collate_fn
from fedscale.cloud.execution.rl_client import RLClient
from fedscale.cloud.fllibs import *
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset
from collections import defaultdict

import copy
import time


class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # initiate the executor log path, and executor ips
        logger.initiate_client_setting()

        self.model_adapter = [self.get_client_trainer(args).get_model_adapter(init_model())]
        if args.use_gradient_cluster:
            self.representation_model = \
                self.get_client_trainer(args).get_model_adapter(init_model(for_embedding=True))
            logging.info(f"create representation model adapter")

        self.args = args
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.training_sets = self.test_dataset = None

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(
            args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.round = [0]
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        self.wandb = None

        logging.info(f"Executor {self.this_rank} initial learning rate: {self.args.learning_rate}")

        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment
        """
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=1)

    def setup_communication(self):
        """Set up grpc connection
        """
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=1):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)
        """
        pass

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.args.task == "rl":
            return train_dataset, test_dataset
        if self.args.task == 'nlp':
            self.collate_fn = collate
        elif self.args.task == 'voice':
            self.collate_fn = voice_collate_fn
        logging.info(f"Dataset size: {len(train_dataset)}, {len(test_dataset)}")
        # load data partitionxr (entire_train_data)

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=self.args.data_map_file, 
            data_frame_file=self.args.data_frame_file, client_start_end_rounds_file=self.args.client_start_end_rounds_file)

        # NOTE: don't set isTest if we want to get test samples by client_id
        testing_sets = DataPartitioner(
            data=test_dataset, args=self.args, numOfClass=self.args.num_class)
        # testing_sets = DataPartitioner(
        #     data=test_dataset, args=self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=self.args.test_data_map_file, 
            data_frame_file=self.args.test_data_frame_file, client_start_end_rounds_file=self.args.client_start_end_rounds_file)

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.training_sets, self.testing_sets = self.init_data()
        time.sleep(2) # allow data to be initialzed
        self.setup_communication()
        self.event_monitor()

    def dispatch_worker_events(self, request):
        """Add new events to worker queues

        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.event_queue.append(request)

    def deserialize_response(self, responses):
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.

        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (string, bool, or bytes): TorchClient responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def UpdateModel(self, model_weights, config, cluster_id=0):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config

        """
        self.round[cluster_id] = config['round']
        self.model_adapter[cluster_id].set_weights(model_weights, is_aggregator=False)

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config, cluster_id, train_round = \
            config['client_id'], config['task_config'], config['cluster_id'], config['round']

        train_config['use_shared_model'] = config.get('global_gradient', False)
        train_config['get_representation'] = train_config['use_shared_model'] and self.args.get_projection
        train_config['round'] = train_round
        if train_round != self.round[cluster_id]:
            logging.info(f"Executor {self.this_rank} Cluster {cluster_id} client {client_id} updated round {self.round[cluster_id]} to {train_round}")
            self.round[cluster_id] = train_round
            
        if 'model' not in config or not config['model']:
            raise "The 'model' object must be a non-null value in the training config."
        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            client_id=client_id, conf=client_conf, model=config['model'], cluster_id=cluster_id)

        report_success = False
        retried = False
        while not report_success:
            try:
                if not retried:
                    # Report execution completion meta information
                    response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                        job_api_pb2.CompleteRequest(
                            client_id=str(client_id), executor_id=self.executor_id,
                            event=commons.encode_clusterid(
                            commons.GLOBAL_GRADIENT if config['global_gradient'] else commons.CLIENT_TRAIN, cluster_id), 
                            status=True, msg=None, meta_result=None, data_result=None
                        )
                    )
                else:
                    response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                        job_api_pb2.CompleteRequest(
                            client_id=str(client_id), executor_id=self.executor_id,
                            event=commons.encode_clusterid(commons.EXECUTOR_RECONNECT, cluster_id), status=True, msg=None,
                            meta_result=None, data_result=None
                        )
                    )
                self.dispatch_worker_events(response)
                report_success = True
            except Exception as e:
                logging.info(f"Caught exception {e} in client training handler, retry dispatching training result")
                # reconnect to the server
                self.init_control_communication()
                retried = True
        # logging.info(f"Executor {self.this_rank} Cluster {cluster_id} client {client_id} dispatched training result")

        return client_id, train_res

    def Test(self, config, cluster_id=0):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group

        Args:
            config (dictionary): The client testing config.

        """
        test_res, per_client_record = self.testing_handler(config['client_id'], cluster_id)
        test_res = {'executorId': self.this_rank, 'results': test_res, \
                    'per_client_record': per_client_record}

        report_success = False
        while not report_success:
            try:
                # Report execution completion information
                response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                    job_api_pb2.CompleteRequest(
                        client_id=self.executor_id, executor_id=self.executor_id,
                        event=commons.encode_clusterid(commons.MODEL_TEST, cluster_id), status=True, msg=None,
                        meta_result=None, data_result=self.serialize_response(test_res)
                    )
                )
                self.dispatch_worker_events(response)
                report_success = True
            except Exception as e:
                logging.info(f"Caught exception {e} in client test handler, retry dispatching test result")
                # reconnect to the server
                self.init_control_communication()
        logging.info(f"Executor {self.this_rank} Cluster {cluster_id} dispatched testing result")

    def Stop(self):
        """Stop the current executor
        """
        logging.info(f"Terminating the executor ...")
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True
        if self.wandb != None:
            self.wandb.finish()

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def override_conf(self, config):
        """ Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """
        Returns a framework-specific client that handles training and evaluation.
        :param conf: job config
        :return: framework-specific client instance
        """
        if conf.engine == commons.TENSORFLOW:
            return TensorflowClient(conf)
        elif conf.engine == commons.PYTORCH:
            if conf.task == 'rl':
                return RLClient(conf)
            else:
                # logging.info(f"New PyTorch client with config {conf}")
                return TorchClient(conf)
        raise "Currently, FedScale supports tensorflow and pytorch."

    def training_handler(self, client_id, conf, model, cluster_id=0, pick_best_test=False):
        """Train model given client id

        Args:
            client_id (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result

        """
        if conf.use_shared_model:
            self.representation_model.set_weights(model, is_aggregator=False)
        else:
            self.model_adapter[cluster_id].set_weights(model, is_aggregator=False)
        conf.client_id = client_id
        conf.tokenizer = tokenizer
        training_select_results = (self.training_sets, None) if self.args.task == "rl" else \
            select_dataset(client_id, self.training_sets,
                            batch_size=conf.batch_size, args=self.args,
                            collate_fn=self.collate_fn,
                            round=conf.round
                            )
        client_data, training_label_counts = training_select_results[0], training_select_results[1]
        # logging.info(f"Cluster {cluster_id} client {client_id} training sampler: {list(copy.deepcopy(client_data.batch_sampler))}")
        client = self.get_client_trainer(self.args)
        if len(client_data) == 0:
            state_dicts = self.model_adapter[cluster_id].get_model().state_dict()
            logging.info(f"Cluster {cluster_id} client {client_id} no enough data, skip training and return empty result")
            if sum(training_label_counts) != 0:
                logging.info(f"WARNING: Cluster {cluster_id} client {client_id} skipped training but with non-empty label counts {training_label_counts}")
            return {'client_id': client_id, 'moving_loss': 0,
                    'trained_size': 0, 'utility': 0, 'wall_duration': 0,
                    'update_weight': {p: state_dicts[p].data.cpu().numpy()
                        for p in state_dicts},
                    'success': 0, 'top_1': 0, 'top_5': 0,
                    'training_label_counts': training_label_counts}
       
        train_res = client.train(
            client_data=client_data, 
            model=self.representation_model.get_model() if conf.use_shared_model else self.model_adapter[cluster_id].get_model(),
            conf=conf)
        if training_label_counts is not None:
            train_res['training_label_counts'] = training_label_counts
        logging.info(f"Executor {self.this_rank} Cluster {cluster_id} client {client_id} training done, lr {conf.learning_rate}, loss {train_res['moving_loss']}, trained size {train_res['trained_size']}, utility {train_res['utility']}, wall duration {train_res['wall_duration']}")

        return train_res

    def testing_handler(self, client_list, cluster_id=0, pick_best_test=False):
        """Test model

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        per_client_record = []
        if pick_best_test:
            test_results_accumulator = []
        else:
            test_results_accumulator = {'top_1': 0, 'top_5': 0, 'test_loss': 0, 'test_len': 0, 'wrong_predictions': []}
        # test_results_accumulator = {'top_1': 0, 'top_5': 0, 'test_loss': 0, 'test_len': 0}
        executor_test_num = len(client_list) // self.num_executors
        if executor_test_num == 0 and self.this_rank != 1:
            return test_results_accumulator, per_client_record
        # NOTE: rank starts from 1
        # make sure that we are not always discarding some clients for testing by letting rank 1 handle the remaining clients
        executor_test_client_id = client_list[(self.this_rank - 1) * executor_test_num\
                                               : min(self.this_rank * executor_test_num, len(client_list))]
        if self.this_rank == 1:
            executor_test_client_id += client_list[self.num_executors * executor_test_num:]
        
        test_config = self.override_conf({
            'rank': self.this_rank,
            'memory_capacity': self.args.memory_capacity,
            'tokenizer': tokenizer
        })

        client = self.get_client_trainer(test_config)
        
        for test_client in executor_test_client_id:
            # using client_id requires setting isTest to False
            try:
                data_loader, _ = select_dataset(test_client, self.testing_sets if (not pick_best_test) else self.training_sets,
                                        batch_size=self.args.test_bsz, args=self.args,
                                        isTest=(False if pick_best_test else True), collate_fn=self.collate_fn,
                                        client_test=(False if pick_best_test else True),
                                        round=self.round[cluster_id]-1) # test data is from the previous round
                if len(data_loader) == 0:
                    logging.info(f"Cluster {cluster_id} client {test_client} no enough data, skip testing")
                else:
                    logging.info(f"Executor {self.this_rank} Cluster {cluster_id} client {test_client} testing sampler: {list(copy.deepcopy(data_loader.batch_sampler))}")
                    # logging.info(f"Executor {self.this_rank} Cluster {cluster_id} client {test_client} testing")
                    test_results = client.test(data_loader, self.model_adapter[cluster_id].get_model(), test_config)
                    per_client_record.append((test_client, \
                        float(test_results['top_1']) / max(1, float(test_results['test_len']))))
                    if pick_best_test:
                        test_results_accumulator.append((test_client, test_results))
                    else:
                        test_results_accumulator['top_1'] += test_results['top_1']
                        test_results_accumulator['top_5'] += test_results['top_5']
                        test_results_accumulator['test_loss'] += test_results['test_loss']
                        test_results_accumulator['test_len'] += test_results['test_len']
                        test_results_accumulator['wrong_predictions'] += test_results['wrong_predictions']
                        # logging.info(f"client {test_client} testing all predictions: {test_results['all_predictions']}")
            except Exception as ex:
                logging.info(f"Caught exception {ex} in client testing handler, skip testing")

        if (not pick_best_test) and test_results_accumulator['test_len'] > 0:
            self.log_test_result(test_results_accumulator, cluster_id)
        
        gc.collect()

        logging.info(f"[Cluster {cluster_id}] Executor {self.this_rank} Test {len(executor_test_client_id)} clients, ID: {executor_test_client_id}")

        # return test_results
        return test_results_accumulator, per_client_record

    def client_register(self):
        """Register the executor information to the aggregator
        """
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler())
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task
        """
        response = self.aggregator_communicator.stub.CLIENT_PING(job_api_pb2.PingRequest(
            client_id=self.executor_id,
            executor_id=self.executor_id
        ))
        self.dispatch_worker_events(response)

    def _init_split(self, new_cluster_meta):
        new_cluster_id = new_cluster_meta['new_cluster_id']
        prev_cluster_id = new_cluster_meta['prev_cluster_id']
        # expand self.model_adapter if necessary
        if len(self.model_adapter) <= new_cluster_id:
            while len(self.model_adapter) <= new_cluster_id:
                self.model_adapter.append(copy.deepcopy(self.model_adapter[prev_cluster_id]))
                self.round.append(new_cluster_meta['round'])
        else:
            self.model_adapter[new_cluster_id] = copy.deepcopy(self.model_adapter[prev_cluster_id])
            self.round[new_cluster_id] = new_cluster_meta['round']

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()
        logging.info(f"Executor {self.this_rank} registered to aggregator")

        while not self.received_stop_request:
            if len(self.event_queue) > 0:
                # logging.info(f"Executor {self.this_rank} received new event ...")
                request = self.event_queue.popleft()
                current_event = request.event
                event_type, cluster_id = commons.decode_clusterid(current_event)

                if event_type == commons.CLIENT_TRAIN or event_type == commons.GLOBAL_GRADIENT:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    if event_type == commons.GLOBAL_GRADIENT:
                        train_config['global_gradient'] = True
                    else:
                        train_config['global_gradient'] = False
                    client_id, train_res = self.Train(train_config)

                    report_success = False
                    while not report_success:
                        try:
                            # Upload model updates
                            response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
                                job_api_pb2.CompleteRequest(
                                    client_id=str(client_id), executor_id=self.executor_id,
                                    event=commons.encode_clusterid(
                                        commons.GLOBAL_GRADIENT_COMPLETE if event_type == commons.GLOBAL_GRADIENT else commons.UPLOAD_MODEL, cluster_id), 
                                    status=True, msg=None, meta_result=None, data_result=self.serialize_response(train_res)
                                    ))
                            self.dispatch_worker_events(response)
                            report_success = True
                        except Exception as e:
                            logging.info(f"Caught exception {e} when uploading model updates, retry dispatching model updates")
                            # reconnect to the server
                            self.init_control_communication()

                elif event_type == commons.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta), cluster_id)

                elif event_type == commons.UPDATE_MODEL:
                    model_weights = self.deserialize_response(request.data)
                    self.UpdateModel(model_weights, self.deserialize_response(request.meta), cluster_id)
                
                elif event_type == commons.CLUSTER_SPLIT:
                    new_cluster_meta = self.deserialize_response(request.meta)
                    self._init_split(new_cluster_meta)

                elif event_type == commons.SHUT_DOWN:
                    self.Stop()

                elif event_type == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                # logging.info(f"Executor {self.this_rank} is pinging aggregator ...")
                ping_success = False
                while not ping_success:
                    try:
                        self.client_ping()
                        ping_success = True
                    except Exception as e:
                        logging.info(f"Caught exception {e} from aggregator, executor {self.this_rank} resend ping ...")
                        # reconnect to the server
                        self.init_control_communication()
                        # logging.info(f"Caught exception {e} from aggregator, terminating executor {self.this_rank} ...")
                        # self.Stop()

    
    def log_test_result(self, test_res, cluster_id=0):
        """Log test results to wandb server if enabled
        """
        acc = round(test_res["top_1"] / test_res["test_len"], 4)
        acc_5 = round(test_res["top_5"] / test_res["test_len"], 4)
        test_loss = test_res["test_loss"] / test_res["test_len"]
        if self.wandb != None:
            self.wandb.log({
                f'Test/round_to_top1_accuracy_cluster{cluster_id}': acc,
                f'Test/round_to_top5_accuracy_cluster{cluster_id}': acc_5,
                f'Test/round_to_loss_cluster{cluster_id}': test_loss,
            }, step=self.round[cluster_id])

if __name__ == "__main__":
    executor = Executor(parser.args)
    try:
        executor.run()
    except Exception as e:
        logging.error(f"Executor {executor.this_rank} caught exception {e}")
        executor.Stop()
        raise e
