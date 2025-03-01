# -*- coding: utf-8 -*-
import collections
import copy
import math
import os
import pickle
import random
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc
import fedscale.cloud.logger.aggregator_logging as logger
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.channels import job_api_pb2
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.internal.tensorflow_model_adapter import TensorflowModelAdapter
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.cloud.resource_manager import ResourceManager
from fedscale.cloud.fllibs import *
from torch.utils.tensorboard import SummaryWriter

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB


class Aggregator(job_api_pb2_grpc.JobServiceServicer):
    """This centralized aggregator collects training/testing feedbacks from executors

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """

    def __init__(self, args):
        # init aggregator loger
        logger.initiate_aggregator_setting()

        logging.info(f"Job args {args}")
        self.args = args
        self.experiment_mode = args.experiment_mode
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')

        # ======== env information ========
        self.this_rank = 0
        self.last_update_clock_round = 0
        self.global_virtual_clock = [0.]
        self.max_global_virtual_clock = 0
        self.round_duration = [0.]
        self.resource_manager = ResourceManager(self.experiment_mode)
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model_wrapper = None
        self.model_in_update = [0]
        self.update_lock = threading.Lock()
        # all weights including bias/#_batch_tracked (e.g., state_dict)
        self.model_weights = None
        self.temp_model_path = os.path.join(
            logger.logDir, 'model_'+str(args.this_rank)+".npy")
        self.last_saved_round = [0]
        self.checkpoint_path = os.path.join(args.checkpoint_dir, args.job_name, args.time_stamp, f'model_{args.model}')
        # create checkpoint directory if not exists
        if args.save_checkpoint and (not os.path.exists(os.path.join(args.checkpoint_dir, args.job_name, args.time_stamp))):
            os.makedirs(os.path.join(args.checkpoint_dir, args.job_name, args.time_stamp))
        logging.info(f"Checkpoint path: {self.checkpoint_path}")
        self.need_loading_cluster_checkpoint = False
        self.saved_states = None

        # ======== channels ========
        self.connection_timeout = self.args.connection_timeout
        self.executors = None
        self.grpc_server = None

        # ======== Event Queue =======
        self.individual_client_events = {}  # Unicast
        self.sever_events_queue = collections.deque()
        self.broadcast_events_queue = collections.deque()  # Broadcast

        # ======== runtime information ========
        self.tasks_round = [0]
        self.num_of_clients = 0

        # ======== train accuracy information ========
        self.total_trained_samples = [0]
        self.total_top1 = [0]
        self.per_client_top1 = [[]]
        self.total_top5 = [0]

        # NOTE: sampled_participants = sampled_executors in deployment,
        # because every participant is an executor. However, in simulation mode,
        # executors is the physical machines (VMs), thus:
        # |sampled_executors| << |sampled_participants| as an VM may run multiple participants
        self.sampled_participants = [[]]
        self.sampled_executors = []

        self.round_stragglers = [[]]
        self.model_update_size = 0.

        self.collate_fn = None
        self.round = [0]
        self.global_round = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = [[]]
        self.loss_accumulator = [[]]
        self.client_training_results = [[]]
        self.virtual_client_clock = [[]]

        # ======== clustering information ========
        self.split_round = args.split_round
        self.default_global_recluster = args.default_global_recluster
        self.force_incremental = args.force_incremental
        self.use_global_model = args.use_global_model
        self.adjust_cluster_selected_num = args.adjust_cluster_selected_num
        self.num_cluster = 0
        self.new_cluster_mapping = {}
        self.need_optimizer_reset = False
        self.round_completion_cluster_count = 0
        self.test_complete_cluster_count = 0

        self.cluster_clients_to_test = {}

        self.global_testing = False

        self.client_to_existing_cluster = {}

        # ======== timing information ========
        self.model_update_aggregate_time = [0.]
        self.curr_round_start_time = None

        # ======== executor current job information for simulation mode reconnect ========
        self.executor_current_clientid = {}

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = [[]]
        self.test_reported = [[]]
        self.testing_history = [{'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                                'gradient_policy': args.gradient_policy, 'task': args.task,
                                'perf': collections.OrderedDict()}]
        self.total_test_top1 = [0]
        self.total_test_top5 = [0]
        self.total_test_samples = [0]
        self.per_client_test_top1 = [[]]
        self.log_writer = SummaryWriter(log_dir=logger.logDir)
        if args.wandb_token != "":
            os.environ['WANDB_API_KEY'] = args.wandb_token
            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(project=f'fedscale-{args.job_name}',
                                name=f'aggregator-{args.time_stamp}',
                                group=f'{args.time_stamp}')
                self.wandb.config.update({
                    "num_participants": args.num_participants,
                    "data_set": args.data_set,
                    "test_data_map_file": args.test_data_map_file,
                    "model": args.model,
                    "model_zoo": args.model_zoo,
                    "pretrained": args.pretrained,
                    "gradient_policy": args.gradient_policy,
                    "eval_interval": args.eval_interval,
                    "rounds": args.rounds,
                    "local_steps": args.local_steps,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "test_bsz": args.test_bsz,
                    "test_client_ratio": args.test_client_ratio,
                    "malicious_ratio": args.malicious_ratio,
                    "use_cuda": args.use_cuda,
                    "sample_mode": args.sample_mode,
                    "split_round": args.split_round,
                    "save_checkpoint": args.save_checkpoint,
                    "load_checkpoint": args.load_checkpoint,
                    "run_epochs": args.run_epochs,
                    "proxy_mu": args.proxy_mu,
                    "default_global_recluster": args.default_global_recluster,
                    "global_recluster_thres_ratio": args.global_recluster_thres_ratio,
                    "recluster_at_drift": args.recluster_at_drift,
                    "force_incremental": args.force_incremental,
                    "adjust_cluster_selected_num": args.adjust_cluster_selected_num,
                    "min_num_cluster": args.min_num_cluster,
                })
                self.training_logs_for_wandb = []
                self.testing_logs_for_wandb = []
                # avoid unnecessary logging to wandb
                logging.getLogger('wandb').setLevel(logging.WARNING)
            else:
                logging.error("Warning: wandb has already been initialized")
        else:
            self.wandb = None

        # ======== Task specific ============
        self.init_task_context()

    def setup_env(self):
        """Set up experiments environment and server optimizer
        """
        self.setup_seed(seed=1)

    def setup_seed(self, seed=1):
        """Set global random seed for better reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.rng = random.Random()
        self.rng.seed(seed)
        self.numpy_rng = np.random.default_rng(seed=seed)
        torch.backends.cudnn.deterministic = True

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        logging.info(f"Initiating control plane communication ...")
        if self.experiment_mode == commons.SIMULATION_MODE:
            num_of_executors = 0
            for ip_numgpu in self.args.executor_configs.split("="):
                ip, numgpu = ip_numgpu.split(':')
                for numexe in numgpu.strip()[1:-1].split(','):
                    for _ in range(int(numexe.strip())):
                        num_of_executors += 1
            self.executors = list(range(num_of_executors))
        else:
            self.executors = list(range(self.args.num_participants))

        # initiate a server process
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(
            self, self.grpc_server)
        port = '[::]:{}'.format(self.args.ps_port)

        logging.info(f'%%%%%%%%%% Opening aggregator sever using port {port} %%%%%%%%%%')

        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()

    def init_data_communication(self):
        """For jumbo traffics (e.g., training results).
        """
        pass

    def init_model(self):
        """Initialize the model"""
        if self.args.engine == commons.TENSORFLOW:
            self.model_wrapper = [TensorflowModelAdapter(init_model())]
        elif self.args.engine == commons.PYTORCH:
            self.model_wrapper = [TorchModelAdapter(
                init_model(),
                optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device))]
        else:
            raise ValueError(f"{self.args.engine} is not a supported engine.")
        if self.args.load_checkpoint:
            checkpoint_load_path = os.path.join(self.args.checkpoint_dir, self.args.job_name, f'model_{self.args.model}_cluster0.pth')
            if os.path.exists(checkpoint_load_path):
                self.model_wrapper[0].load_checkpoint(checkpoint_load_path)
                logging.info(f"Loaded model from {checkpoint_load_path}")
            else:
                logging.info(f"Checkpoint file {checkpoint_load_path} not found, start from scratch")
            state_load_path = os.path.join(self.args.checkpoint_dir, self.args.job_name, f'model_{self.args.model}_feasible_clients.pkl')
            with open(state_load_path, 'rb') as fin:
                self.saved_states = pickle.load(fin)
                self.num_cluster = len(self.saved_states['current_clusters'])
                if self.num_cluster > 0:
                    self.need_loading_cluster_checkpoint = True
                self.round[0] = self.saved_states['round']
                self.global_round = self.saved_states['round']
                # update the learning rate (note that testing din't update learning rate, so lr will be updated again during next training round)
                num_decays = (self.round[0] - 1) // self.args.decay_round
                self.args.learning_rate = max(
                    self.args.learning_rate * self.args.decay_factor ** num_decays, self.args.min_learning_rate)
        self.model_weights = [self.model_wrapper[0].get_weights()]

    def init_task_context(self):
        """Initiate execution context for specific tasks
        """
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb(
                "voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

    def init_client_manager(self, args):
        """ Initialize client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            ClientManager: The client manager class

        Currently we implement two client managers:

        1. Random client sampler - it selects participants randomly in each round
        [Ref]: https://arxiv.org/abs/1902.01046

        2. Oort sampler
        Oort prioritizes the use of those clients who have both data that offers the greatest utility
        in improving model accuracy and the capability to run training quickly.
        [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai

        """

        # sample_mode: random or oort
        client_manager = ClientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        """For Simulation Mode: load client profiles/traces

        Args:
            file_path (string): File path for the client profiles/traces

        Returns:
            dictionary: Return the client profiles/traces

        """
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {client_id: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)
            logging.info(f"Loaded device traces from {file_path}")

        return global_client_profile

    def client_register_handler(self, executorId, info):
        """Triggered once receive new executor registration.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        logging.info(f"Model updata size: {self.model_update_size} (kbits)")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (self.num_of_clients + 1) % len(
                self.client_profiles) if len(self.client_profiles) > 0 else 1
            systemProfile = self.client_profiles.get(
                mapped_id, {'computation': 1.0, 'communication': 1.0})

            client_id = (
                    self.num_of_clients + 1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
            self.client_manager.register_client(
                executorId, client_id, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(
                client_id,
                batch_size=self.args.batch_size,
                local_steps=self.args.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size
            )
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(
            self.client_manager.getDataInfo()))

    def executor_info_handler(self, executorId, info):
        """Handler for register executor info and it will start the round after number of
        executor reaches requirement.

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        self.registered_executor_info.add(executorId)
        logging.info(
            f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")

        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout
        if self.experiment_mode == commons.SIMULATION_MODE:
            if len(self.registered_executor_info) == len(self.executors):
                self.client_register_handler(executorId, info)
                # start to sample clients
                if not self.args.load_checkpoint:
                    self.global_round += 1
                self.round_completion_handler()
        else:
            # In real deployments, we need to register for each client
            self.client_register_handler(executorId, info)
            if len(self.registered_executor_info) == len(self.executors):
                if not self.args.load_checkpoint:
                    self.global_round += 1
                self.round_completion_handler()

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect, cluster_id=0):
        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            Tuple: (the List of clients to run, the List of stragglers in the round, a Dict of the virtual clock of each
            client, the duration of the aggregation round, and the durations of each client's task).

        """
        random_comm_delay = False
        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            completionTimes = []
            completed_client_clock = {}
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                try:
                    client_cfg = self.client_conf.get(client_to_run, self.args)

                    exe_cost = self.client_manager.get_completion_time(client_to_run,
                                                                    batch_size=client_cfg.batch_size,
                                                                    local_steps=client_cfg.local_steps,
                                                                    upload_size=self.model_update_size,
                                                                    download_size=self.model_update_size)
                    if random_comm_delay:
                        # randomly change communication cost
                        if self.rng.random() < 0.5:
                            exe_cost['communication'] = self.rng.uniform(1.0, 10.0) * exe_cost['communication']
                    roundDuration = exe_cost['computation'] + \
                                    exe_cost['communication']
                except Exception as ex:
                    logging.info(f"failed to get completion time of client {client_to_run} due to {ex}")
                    continue
                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock[cluster_id]):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                    completed_client_clock[client_to_run] = exe_cost

            num_clients_to_collect = min(
                num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k])
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            stragglers = [sampledClientsReal[k]
                          for k in workers_sorted_by_completion_time[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            final_client = sampled_clients[top_k_index[-1]]
            logging.info(f"Cluster {cluster_id} last client clock: {completed_client_clock[final_client]}")

            return (clients_to_run, stragglers,
                    completed_client_clock, round_duration,
                    completionTimes[:num_clients_to_collect])
        else:
            completed_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            return (sampled_clients, sampled_clients, completed_client_clock,
                    1, completionTimes)

    def run(self):
        """Start running the aggregator server by setting up execution
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file)
            
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.model_update_size = sys.getsizeof(
            pickle.dumps(self.model_wrapper[0])) / 1024.0 * 8.  # kbits

        self.event_monitor()
        self.stop()

    def _is_first_result_in_round(self, cluster_id=0):
        return self.model_in_update[cluster_id] == 1

    def _is_last_result_in_round(self, cluster_id=0):
        return self.model_in_update[cluster_id] == self.tasks_round[cluster_id]

    def select_participants(self, select_num_participants, overcommitment=1.3, cluster_id=0, test=False,
                            check_rank_avail=False):
        """Select clients for next round.

        Args:
            select_num_participants (int): Number of clients to select.
            overcommitment (float): Overcommit ration for next round.

        Returns:
            list of int: The list of sampled clients id.

        """
        if check_rank_avail:
            return sorted(self.client_manager.select_participants(
                int(select_num_participants * overcommitment),
                cur_time=self.global_virtual_clock[cluster_id], cluster_id=cluster_id,
                test=test, curr_round=self.round[cluster_id], check_client_avail=True)
            )
        return sorted(self.client_manager.select_participants(
            int(select_num_participants * overcommitment),
            cur_time=self.global_virtual_clock[cluster_id], cluster_id=cluster_id,
            test=test)
        )

    def client_completion_handler(self, results, cluster_id=0):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        try:
            if results['success'] == 0:
                logging.info(f"Cluster {cluster_id} client {results['client_id']} failed in round {self.round[cluster_id]}, skip client_completion_handler")
                self.stats_util_accumulator[cluster_id].append(0)
                return
            if self.args.gradient_policy in ['q-fedavg']:
                self.client_training_results[cluster_id].append(results)
            # Feed metrics to client sampler
            logging.info("In round: {} (cluster {}), client_id: {}, moving_loss: {}, avg moving_loss: {}, utility: {}, train_info (top 1, top 5, trained_size): ({},{},{})".format(\
                self.round[cluster_id], cluster_id, results['client_id'], results['moving_loss'], float(results['moving_loss'])/float(results['trained_size']) , results['utility'], \
                results['top_1'], results['top_5'], results['trained_size']))
            self.stats_util_accumulator[cluster_id].append(results['utility'])
            self.loss_accumulator[cluster_id].append(results['moving_loss'])
            self.total_trained_samples[cluster_id] += int(results['trained_size'])
            self.total_top1[cluster_id] += int(results['top_1'])
            self.per_client_top1[cluster_id].append(int(results['top_1']) / max(1, int(results['trained_size'])))
            self.total_top5[cluster_id] += int(results['top_5'])
        
            if (self.args.data_mode != "all") and (not self.args.recluster_at_drift) and 'training_label_counts' in results:
                self.client_manager.client_update_label_counts(results['client_id'], results['training_label_counts'])
            # ================== Aggregate weights ======================
            self.update_lock.acquire()

            if results['client_id'] not in self.virtual_client_clock[cluster_id]:
                logging.info(f"Cluster {cluster_id} client {results['client_id']} not in virtual_client_clock, add zeros")
                self.virtual_client_clock[cluster_id][results['client_id']] = {'computation': 0, 'communication': 0}
            self.client_manager.register_feedback(results['client_id'], 
                                                results['utility'],
                                                auxi=math.sqrt(
                                                    results['moving_loss']),
                                                time_stamp=self.round[cluster_id],
                                                duration=self.virtual_client_clock[cluster_id][results['client_id']]['computation'] +
                                                        self.virtual_client_clock[cluster_id][results['client_id']]['communication'],
                                                cluster_id=cluster_id,
                                                top1_accu=int(results['top_1'])/max(1, int(results['trained_size'])),
                                                )
        
        except Exception as ex:
            logging.info(f"client completion handler failed on Cluster {cluster_id} Client {results['client_id']} due to {ex}")

        finally:
            self.model_in_update[cluster_id] += 1
            self.update_weight_aggregation(results, cluster_id)
            self.update_lock.release()

    def update_weight_aggregation(self, results, cluster_id = 0):
        """Updates the aggregation with the new results.

        :param results: the results collected from the client.
        """
        try:
            weight_update_start = time.time()
            update_weights = results['update_weight']
            if type(update_weights) is dict:
                update_weights = [x for x in update_weights.values()]
            if self._is_first_result_in_round(cluster_id):
                self.model_weights[cluster_id] = update_weights
            else:
                self.model_weights[cluster_id] = [weight + update_weights[i] for i, weight in enumerate(self.model_weights[cluster_id])]
            if self._is_last_result_in_round(cluster_id):
                self.model_weights[cluster_id] = [np.divide(weight, self.tasks_round[cluster_id]) for weight in self.model_weights[cluster_id]]
                self.model_wrapper[cluster_id].set_weights(copy.deepcopy(self.model_weights[cluster_id]), 
                                                           is_aggregator=True, 
                                                           client_training_results=self.client_training_results[cluster_id],
                                                           current_learning_rate=self.args.learning_rate)
            self.model_update_aggregate_time[cluster_id] += (time.time() - weight_update_start)
            
            if self._is_last_result_in_round(cluster_id):
                logging.info(f"Cluster {cluster_id} update weight time: {self.model_update_aggregate_time[cluster_id]}s")
        except Exception as ex:
            logging.info(f"Cluster {cluster_id} update weight aggregation failed due to {ex}")

    def aggregate_test_result(self, cluster_id=0):
        logging.info(f"aggregate Cluster {cluster_id} test results")
        accumulator = self.test_result_accumulator[cluster_id][0]
        for i in range(1, len(self.test_result_accumulator[cluster_id])):
            if self.args.task == "detection":
                for key in accumulator:
                    if key == "boxes":
                        for j in range(596):
                            accumulator[key][j] = accumulator[key][j] + \
                                                  self.test_result_accumulator[cluster_id][i][key][j]
                    else:
                        accumulator[key] += self.test_result_accumulator[cluster_id][i][key]
            else:
                for key in accumulator:
                    accumulator[key] += self.test_result_accumulator[cluster_id][i][key]
        self.testing_history[cluster_id]['perf'][self.round[cluster_id]] = {'round': self.round[cluster_id], 'clock': self.global_virtual_clock[cluster_id]}
        for metric_name in accumulator.keys():
            if metric_name == 'test_loss':
                self.testing_history[cluster_id]['perf'][self.round[cluster_id]]['loss'] = accumulator['test_loss'] \
                    if self.args.task == "detection" else accumulator['test_loss'] / max(accumulator['test_len'],1)
            elif metric_name not in ['test_len', 'wrong_predictions']:
                self.testing_history[cluster_id]['perf'][self.round[cluster_id]][metric_name] \
                    = accumulator[metric_name] / max(accumulator['test_len'],1)

        self.testing_history[cluster_id]['perf'][self.round[cluster_id]]['total_test_len'] = accumulator['test_len']
        self.testing_history[cluster_id]['perf'][self.round[cluster_id]]['wrong_predictions'] = accumulator['wrong_predictions']
        round_perf = self.testing_history[cluster_id]['perf'][self.round[cluster_id]]
        logging.info(
            "Cluster {} FL Testing in round: {}, virtual_clock: {}, results: {}"
            .format(cluster_id, self.round[cluster_id], self.global_virtual_clock[cluster_id], round_perf))

    def update_default_task_config(self, cluster_id=0):
        """Update the default task configuration after each round
        """
        # only update learning rate when the global model calls to avoid over decay 
        if cluster_id == 0 and self.round[cluster_id] % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate * self.args.decay_factor, self.args.min_learning_rate)
            logging.info(f"Learning rate decayed to {self.args.learning_rate}")

    def init_cluster_tasks(self, cluster_id=0):
        if cluster_id == 0:
            num_to_select = self.args.num_participants
        else:
            if self.adjust_cluster_selected_num:
                num_to_select = max(5, self.args.num_participants // self.num_cluster)
            else:
                num_to_select = self.args.num_participants
        # update select participants
        self.sampled_participants[cluster_id] = self.select_participants(
            select_num_participants=num_to_select, overcommitment=self.args.overcommitment,
            cluster_id=cluster_id, check_rank_avail=(self.args.data_mode != "all"))

        if len(self.sampled_participants[cluster_id]) == 0:
            logging.info(f"No available clients in cluster {cluster_id}, skip the round")
            self.round_completion_cluster_count += 1
            self.round_duration[cluster_id] = 0
            self.round_stragglers[cluster_id] = []
            self.loss_accumulator[cluster_id] = []
            self.stats_util_accumulator[cluster_id] = []
            self.test_result_accumulator[cluster_id] = []
            self.test_reported[cluster_id] = []
            return
        try:
            (clients_to_run, round_stragglers, virtual_client_clock, round_duration,
            flatten_client_duration) = self.tictak_client_tasks(
                self.sampled_participants[cluster_id], num_to_select, cluster_id)
        except Exception as ex:
            logging.info(f"Cohort {cluster_id} Selected participants failed due to {ex}, defualt to use all some sampled clients")
            clients_to_run = self.sampled_participants[cluster_id][:min(len(self.sampled_participants[cluster_id]), num_to_select)]
            round_stragglers = self.sampled_participants[cluster_id]
            virtual_client_clock = {
            client: {'computation': 1, 'communication': 1} for client in self.sampled_participants[cluster_id]}
            flatten_client_duration = [1 for c in self.sampled_participants[cluster_id]]
            round_duration = 1
        logging.info(f"Cluster {cluster_id} Selected {len(clients_to_run)} participants to run: {clients_to_run}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clients_to_run, cluster_id)
        self.tasks_round[cluster_id] = len(clients_to_run)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants[cluster_id]]
        self.round_stragglers[cluster_id] = round_stragglers
        self.virtual_client_clock[cluster_id] = virtual_client_clock
        self.flatten_client_duration = np.array(flatten_client_duration)
        self.round_duration[cluster_id] = round_duration
        self.model_in_update[cluster_id] = 0
        self.test_result_accumulator[cluster_id] = []
        self.test_reported[cluster_id] = []
        self.stats_util_accumulator[cluster_id] = []
        self.client_training_results[cluster_id] = []
        self.loss_accumulator[cluster_id] = []
        self.model_update_aggregate_time[cluster_id] = 0.0

    def get_cluster_avg_model(self, cluster_id):
        cluster_clients = self.client_manager.getClusterFeasibleClients(cluster_id)
        aggregate_client_weights = None
        num_clients_with_model_record = 0
        for client_id in cluster_clients:
            if client_id not in self.client_to_existing_cluster:
                logging.info(f"Client {client_id} not in self.client_to_existing_cluster, skip")
                continue
            if self.client_to_existing_cluster[client_id] > len(self.model_weights)-1:
                logging.info(f"Client {client_id} prev cluster {self.client_to_existing_cluster[client_id]} does not have model record, skip")
                continue
            client_prev_cluster_model = self.model_weights[self.client_to_existing_cluster[client_id]]
            if aggregate_client_weights is None:
                aggregate_client_weights = client_prev_cluster_model
            else:
                aggregate_client_weights = \
                    [weight + client_prev_cluster_model[i] for i, weight in enumerate(aggregate_client_weights)]
            num_clients_with_model_record += 1
        if aggregate_client_weights is not None:
            aggregate_client_weights = [np.divide(weight, num_clients_with_model_record)\
                                        for weight in aggregate_client_weights]
            logging.info(f"Cluster {cluster_id} get {num_clients_with_model_record} client average model")
        
        return aggregate_client_weights
    
    def init_new_cluster(self, new_cluster_id, closest_prev_cluster=0, load_from_checkpoint=False):
        logging.info(f"initing {new_cluster_id}")
        if new_cluster_id > len(self.model_wrapper)-1:
            self.resource_manager.add_cluster()
            if (not load_from_checkpoint):
                aggregate_client_weights = self.get_cluster_avg_model(new_cluster_id)
                if aggregate_client_weights is not None:
                    self.model_wrapper.append(copy.deepcopy(self.model_wrapper[closest_prev_cluster]))
                    self.model_weights.append(aggregate_client_weights)
                    logging.info(f"Cluster {new_cluster_id} increase self.model_wrapper length to {len(self.model_wrapper)}")
                    # just set weights, no need to apply optimizer (i.e., is_aggregator=False)
                    self.model_wrapper[new_cluster_id].set_weights(copy.deepcopy(aggregate_client_weights), is_aggregator=False)
                    self.model_wrapper[new_cluster_id].reset_optimizer_state()
                else:
                    # if not able to get client average model, use the closest previous cluster model
                    self.model_wrapper.append(copy.deepcopy(self.model_wrapper[closest_prev_cluster]))
                    self.model_weights.append(copy.deepcopy(self.model_weights[closest_prev_cluster]))
                    logging.info(f"Failed to get client average model for cluster {new_cluster_id}, use the closest previous cluster model")
            else:
                self.model_wrapper.append(copy.deepcopy(self.model_wrapper[closest_prev_cluster]))
                self.model_weights.append(copy.deepcopy(self.model_weights[closest_prev_cluster]))
            self.model_in_update.append(0)
            self.tasks_round.append(0)
            self.round.append(self.global_round)
            self.last_saved_round.append(self.last_saved_round[closest_prev_cluster])
            self.testing_history.append(copy.deepcopy(self.testing_history[closest_prev_cluster]))
            self.global_virtual_clock.append(self.max_global_virtual_clock)
            self.round_duration.append(0.)
            self.total_trained_samples.append(0)
            self.total_top1.append(0)
            self.per_client_top1.append([])
            self.total_top5.append(0)
            self.total_test_samples.append(0)
            self.total_test_top1.append(0)
            self.total_test_top5.append(0)
            self.per_client_test_top1.append([])
            self.model_update_aggregate_time.append(0.)
            for tracker in [self.test_result_accumulator, self.test_reported, self.sampled_participants, 
                            self.round_stragglers, self.stats_util_accumulator,
                            self.loss_accumulator, self.client_training_results,
                            self.virtual_client_clock]:
                tracker.append([])
                if new_cluster_id > len(tracker)-1:
                    logging.info(f"Cluster {new_cluster_id} increase {tracker} length to {len(tracker)}")
                tracker[new_cluster_id] = []
        else:
            self.resource_manager.clear_cluster(cluster_id=new_cluster_id)
            if (not load_from_checkpoint):
                aggregate_client_weights = self.get_cluster_avg_model(new_cluster_id)
                if aggregate_client_weights is not None:
                    self.model_weights[new_cluster_id] = aggregate_client_weights
                    # just set weights, no need to apply optimizer (i.e., is_aggregator=False)
                    self.model_wrapper[new_cluster_id].set_weights(copy.deepcopy(aggregate_client_weights), is_aggregator=False)
                    if self.need_optimizer_reset:
                        self.model_wrapper[new_cluster_id].reset_optimizer_state()
                else:
                    # if not able to get client average model, use the closest previous cluster model
                    self.model_wrapper[new_cluster_id] = copy.deepcopy(self.model_wrapper[closest_prev_cluster])
                    self.model_weights[new_cluster_id] = copy.deepcopy(self.model_weights[closest_prev_cluster])
                    logging.info(f"Failed to get client average model for cluster {new_cluster_id}, use the closest previous cluster model")
            else:
                self.model_wrapper[new_cluster_id] = copy.deepcopy(self.model_wrapper[closest_prev_cluster])
                self.model_weights[new_cluster_id] = copy.deepcopy(self.model_weights[closest_prev_cluster])
            self.model_in_update[new_cluster_id] = 0
            self.tasks_round[new_cluster_id] = 0
            self.round[new_cluster_id] = self.global_round
            self.last_saved_round[new_cluster_id] = self.last_saved_round[closest_prev_cluster]
            self.testing_history[new_cluster_id] = copy.deepcopy(self.testing_history[closest_prev_cluster])
            self.global_virtual_clock[new_cluster_id] = self.max_global_virtual_clock
            # self.global_virtual_clock[new_cluster_id] = copy.deepcopy(self.global_virtual_clock[closest_prev_cluster])
            self.round_duration[new_cluster_id] = 0.0
            self.total_trained_samples[new_cluster_id] = 0
            self.total_top1[new_cluster_id] = 0
            self.per_client_top1[new_cluster_id] = []
            self.total_top5[new_cluster_id] = 0
            self.total_test_samples[new_cluster_id] = 0
            self.total_test_top1[new_cluster_id] = 0
            self.total_test_top5[new_cluster_id] = 0
            self.per_client_test_top1[new_cluster_id] = []
            self.model_update_aggregate_time[new_cluster_id] = 0.0
            for tracker in [self.test_result_accumulator, self.test_reported, self.sampled_participants, 
                            self.round_stragglers, self.stats_util_accumulator,
                            self.loss_accumulator, self.client_training_results,
                            self.virtual_client_clock]:
                tracker[new_cluster_id] = []
        logging.info(f"inited {new_cluster_id}")

    def init_splits(self, new_clusters, load_from_checkpoint=False):
        for new_cluster_id in sorted(new_clusters):
            self.init_new_cluster(new_cluster_id=new_cluster_id, closest_prev_cluster=0, load_from_checkpoint=load_from_checkpoint)
            if load_from_checkpoint:
                checkpoint_load_path = os.path.join(self.args.checkpoint_dir, self.args.job_name, f'model_{self.args.model}_cluster{new_cluster_id}.pth')
                if os.path.exists(checkpoint_load_path):
                    self.model_wrapper[new_cluster_id].load_checkpoint(checkpoint_load_path)
                    self.model_weights[new_cluster_id] = self.model_wrapper[new_cluster_id].get_weights()
                    logging.info(f"Loaded model from {checkpoint_load_path}")
                else:
                    logging.info(f"Checkpoint file {checkpoint_load_path} not found, start from scratch")
                self.round[new_cluster_id] = self.saved_states['round']
                
        self.num_cluster = len(new_clusters)
        for new_cluster_id in new_clusters:
            if load_from_checkpoint:
                # when loading from checkpoint, we only need to broadcast CLUSTER_SPLIT event
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.CLUSTER_SPLIT, new_cluster_id))
            else:
                self.init_cluster_tasks(new_cluster_id)
                logging.info(f"init_cluster_tasks cluster {new_cluster_id}")
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.CLUSTER_SPLIT, new_cluster_id))
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.UPDATE_MODEL, new_cluster_id))
                time.sleep(1)
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.START_ROUND, new_cluster_id))
                # self.broadcast_aggregator_events(commons.encode_clusterid(commons.MODEL_TEST, new_cluster_id))
            logging.info(f"Create cluster {new_cluster_id}")

    def update_client_to_existing_cluster(self):
        unclustered_clients = self.client_manager.getAllClients() + self.client_manager.getAllInfeasibleClients()
        for cluster_id in range(1, self.num_cluster+1):
            cluster_clients = self.client_manager.getClusterFeasibleClients(cluster_id)
            for client_id in cluster_clients:
                self.client_to_existing_cluster[client_id] = cluster_id
            unclustered_clients = (set(unclustered_clients) - set(cluster_clients))
        for client_id in unclustered_clients:
            self.client_to_existing_cluster[client_id] = 0
        logging.info(f"client_to_existing_cluster updated to: {self.client_to_existing_cluster}")

    def round_completion_handler(self, cluster_id=0, after_test=False, after_loading=False):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        # clear self.curr_round_start_time
        self.curr_round_start_time = None
        # clear new_cluster_mapping
        self.new_cluster_mapping = {}
        if (not after_test) and (not self.need_loading_cluster_checkpoint):
            if self.last_update_clock_round < self.round[cluster_id]:
                # update the global max virtual clock time to be the max of all clusters in the last round
                self.max_global_virtual_clock = max(self.global_virtual_clock[:(self.num_cluster+1)])
                self.last_update_clock_round = self.round[cluster_id]
                logging.info(f"max_global_virtual_clock after round {self.last_update_clock_round}: {self.max_global_virtual_clock}")
            self.global_virtual_clock[cluster_id] += self.round_duration[cluster_id]
            self.round[cluster_id] += 1
            last_round_avg_util = sum(self.stats_util_accumulator[cluster_id]) / max(1, len(self.stats_util_accumulator[cluster_id]))

            if self.round[cluster_id] >= self.args.rounds:
                logging.info(f"Cluster {cluster_id} about to exit")
                if cluster_id == self.num_cluster or (cluster_id == 0 and self.num_cluster == 0):
                    # shutdown when all clusters have finished
                    self.broadcast_aggregator_events(commons.encode_clusterid(commons.SHUT_DOWN))
                return
            
            assign_default_reward = True
            if assign_default_reward:
                # assign avg reward to explored, but not ran workers
                for client_id in self.round_stragglers[cluster_id]:
                    self.client_manager.register_feedback(client_id, last_round_avg_util,
                                                        time_stamp=self.round[cluster_id],
                                                        duration=self.virtual_client_clock[cluster_id][client_id]['computation'] +
                                                                self.virtual_client_clock[cluster_id][client_id]['communication'],
                                                        success=False)

            avg_loss = sum(self.loss_accumulator[cluster_id]) / max(1, len(self.loss_accumulator[cluster_id]))
            logging.info(f"Cluster {cluster_id} Wall clock: {round(self.global_virtual_clock[cluster_id])} s, round: {self.round[cluster_id]}, Planned participants: " +
                        f"{len(self.sampled_participants[cluster_id])}, Succeed participants: {len(self.stats_util_accumulator[cluster_id])}, Training loss: {avg_loss}")


            # dump round completion information to tensorboard
            if len(self.loss_accumulator[cluster_id]):
                logging.info(f"Cluster {cluster_id} has {len(self.per_client_top1[cluster_id])} per client training top1 records")
                self.log_train_result_wandb(avg_loss, cluster_id)
                # reset round counters
                self.total_trained_samples[cluster_id] = 0
                self.total_top1[cluster_id] = 0
                self.per_client_top1[cluster_id] = []
                self.total_top5[cluster_id] = 0

            if self.round[cluster_id] % self.args.eval_interval == 0:
                self.init_cluster_tasks(cluster_id)
                #NOTE: evaluate no need to update lr
                self.get_test_client_set(cluster_id)
                # clear the per_client_test_top1, test top1 and top5
                self.per_client_test_top1[cluster_id] = []
                self.total_test_samples[cluster_id] = 0
                self.total_test_top1[cluster_id] = 0
                self.total_test_top5[cluster_id] = 0
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.UPDATE_MODEL, cluster_id))
                time.sleep(1)
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.MODEL_TEST, cluster_id))
                self.curr_round_start_time = time.time()
                self.global_testing = True
                return

        if (self.args.data_mode != "all") and self.num_cluster == 0 and cluster_id == 0:
            # when there is only the global cluster, we need to update the label distribution of all clients
            self.client_manager.global_client_update_label_counts(round=self.round[cluster_id])
        if cluster_id == 1 and (self.args.data_mode != "all"):
            # update client_to_existing_cluster mapping before recluster for cluster model initialization
            self.update_client_to_existing_cluster()
            self.new_cluster_mapping, self.need_optimizer_reset = \
                self.client_manager.clientReclusterAll(clusters=list(range(0, self.num_cluster+1)), \
                                                    use_distribution=True, \
                                                    default_global_recluster=self.default_global_recluster, \
                                                    round=self.round[cluster_id], \
                                                    use_global_model=self.use_global_model,\
                                                    force_incremental=self.force_incremental
                                                    )
            logging.info(f"cluster {cluster_id} invoked clientBasedReclusterAll")

        elif cluster_id == 0 and self.round[cluster_id] == self.split_round:
            # update client_to_existing_cluster mapping before recluster for cluster model initialization
            self.update_client_to_existing_cluster()
            logging.info(f"cluster {cluster_id} invoked global_clustering")
            new_clusters, _ = self.client_manager.global_clustering(
                curr_round=self.round[cluster_id], initial=True)
            self.init_splits(new_clusters)

            # cluster 0 should proceed as normal
            self.init_cluster_tasks(cluster_id)
            self.update_default_task_config(cluster_id)
            if not after_test:
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.UPDATE_MODEL, 0))
                time.sleep(1)
            self.broadcast_aggregator_events(commons.encode_clusterid(commons.START_ROUND, 0))
            self.curr_round_start_time = time.time()
            
        elif len(self.new_cluster_mapping) > 0:
            for new_cluster in sorted(self.new_cluster_mapping.keys()):
                closest_prev = self.new_cluster_mapping[new_cluster]
                self.init_new_cluster(new_cluster_id=new_cluster, closest_prev_cluster=closest_prev)
            self.num_cluster = len(self.new_cluster_mapping)
            for new_cluster_id in self.new_cluster_mapping.keys():
                self.init_cluster_tasks(new_cluster_id)
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.CLUSTER_SPLIT, new_cluster_id))
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.UPDATE_MODEL, new_cluster_id))
                time.sleep(1)
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.START_ROUND, new_cluster_id))
                logging.info(f"Reclustering resulted new cluster {new_cluster_id}")
            
        else:
            # initialize the saved cluster models if needed
            if self.need_loading_cluster_checkpoint:
                self.client_manager.load_saved_states(self.saved_states)
                self.init_splits(new_clusters=list(range(1, self.num_cluster+1)), load_from_checkpoint=True)
                self.need_loading_cluster_checkpoint = False
                logging.info(f"Clusters {list(range(1, self.num_cluster+1))} loaded from checkpoint")
                for i in range(0, self.num_cluster+1):
                    self.round_completion_handler(cluster_id=i, after_test=True, after_loading=True)
                    if len(self.new_cluster_mapping) > 0:
                        break
            else:
                self.init_cluster_tasks(cluster_id)
                self.update_default_task_config(cluster_id)
                if (not after_test) or after_loading:
                    self.broadcast_aggregator_events(commons.encode_clusterid(commons.UPDATE_MODEL, cluster_id))
                    time.sleep(1)
                self.broadcast_aggregator_events(commons.encode_clusterid(commons.START_ROUND, cluster_id))
                self.curr_round_start_time = time.time()

    def log_train_result_wandb(self, avg_loss, cluster_id=0):
        """Log training result on WanDB
        """
        if self.wandb != None:
            self.wandb.log({
                f'Train/round_to_loss_cluster{cluster_id}': avg_loss,
                f'Train/round_duration_cluster{cluster_id} (min)': self.round_duration[cluster_id]/60.,
                f'Train/time_to_round_cluster{cluster_id} (min)': self.global_virtual_clock[cluster_id]/60.,
                f'Train/round_to_top1_accuracy_cluster{cluster_id}': float(self.total_top1[cluster_id])/max(1, self.total_trained_samples[cluster_id]),
                f'Train/mean_top1_accuracy_cluster{cluster_id}': np.mean(self.per_client_top1[cluster_id]),
            }, step=self.round[cluster_id])
            if cluster_id == 0:
                self.wandb.log({
                    f'Train/time_to_round (min)': self.max_global_virtual_clock/60.,
                }, step=self.round[cluster_id])
        
    def log_all_clusters_mean_test_result(self):
        """Log the mean testing result of all clusters, and then for all clients
        """
        # dump record
        if self.args.dump_per_client_accuracy:
            with open(f"./workspace/records/{self.args.job_name}/per_client_accuracy/{self.args.time_stamp}.pkl", 'ab+') as f:
                pickle.dump(self.per_client_test_top1, f)
        if self.wandb != None:
            # log all clients test result
            # log the mean and median accuracy of clients on the global model
            mean_per_client_top1_accuracy_on_global_model = np.mean([x[1] for x in self.per_client_test_top1[0]])
            median_per_client_top1_accuracy_on_global_model = np.median([x[1] for x in self.per_client_test_top1[0]])
            self.wandb.log({
                'Test/all_clients_mean_top1_accuracy_on_global_model': mean_per_client_top1_accuracy_on_global_model,
                'Test/all_clients_median_top1_accuracy_on_global_model': median_per_client_top1_accuracy_on_global_model,
            }, step=self.round[0])
            logging.info(f"all_clients_mean_top1_accuracy_on_global_model: {mean_per_client_top1_accuracy_on_global_model}")
            logging.info(f"all_clients_median_top1_accuracy_on_global_model: {median_per_client_top1_accuracy_on_global_model}")

            if self.num_cluster == 0:
                return
            total_test_samples = 0
            total_test_top1 = 0
            all_per_client_top1_accuracy = []
            for cluster_id in range(1, self.num_cluster+1):
                total_test_samples += self.total_test_samples[cluster_id]
                total_test_top1 += self.total_test_top1[cluster_id]
                all_per_client_top1_accuracy += [x[1] for x in self.per_client_test_top1[cluster_id]]
                logging.info(f"Cluster {cluster_id} has {len(self.per_client_test_top1[cluster_id])} per client test top1 from clients: {[x[0] for x in self.per_client_test_top1[cluster_id]]}")
            mean_per_client_top1_accuracy = np.mean(all_per_client_top1_accuracy)
            median_per_client_top1_accuracy = np.median(all_per_client_top1_accuracy)
            self.wandb.log({
                'Test/all_clusters_mean_top1_accuracy': total_test_top1/max(1,total_test_samples),
                'Test/all_clustered_clients_mean_top1_accuracy': mean_per_client_top1_accuracy,
                'Test/all_clustered_clients_median_top1_accuracy': median_per_client_top1_accuracy,
            }, step=self.round[0])
            logging.info(f"all_clusters_mean_top1_accuracy (avg across all samples): {total_test_top1/max(1,total_test_samples)}")
            logging.info(f"all_clustered_clients_mean_top1_accuracy: {mean_per_client_top1_accuracy}")
            logging.info(f"all_clustered_clients_median_top1_accuracy: {median_per_client_top1_accuracy}")
            
            # get the number of clients only in cluster 0 and not in other clusters
            total_unclustered_clients = self.cluster_clients_to_test[0]
            for cluster_id in range(1, self.num_cluster+1):
                total_unclustered_clients = set(total_unclustered_clients) - \
                    set(self.cluster_clients_to_test[cluster_id])
            logging.info(f"Cluster 0 has {len(total_unclustered_clients)} (out of {len(self.cluster_clients_to_test[0])}) unclustered clients: {total_unclustered_clients}")
            unclustered_client_top1_accuracy = []
            clustered_client_top1_on_global_model = []
            for client_top1 in self.per_client_test_top1[0]:
                if client_top1[0] in total_unclustered_clients:
                    logging.info(f"Client {client_top1[0]} using the global model, top1 accuracy: {client_top1[1]}")
                    all_per_client_top1_accuracy.append(client_top1[1])
                    unclustered_client_top1_accuracy.append(client_top1[1])
                else:
                    clustered_client_top1_on_global_model.append(client_top1[1])
            logging.info(f"Cluster 0 has {len(unclustered_client_top1_accuracy)} per client test top1 recrods for unclustered clients using the global model")
            if len(all_per_client_top1_accuracy) != len(self.cluster_clients_to_test[0]):
                logging.info(f"WARNING: all_per_client_top1_accuracy has length {len(all_per_client_top1_accuracy)}, expected: {len(self.cluster_clients_to_test[0])}")
            logging.info(f"clustered_client_top1_on_global_model: {np.mean(clustered_client_top1_on_global_model)}")
            if len(unclustered_client_top1_accuracy) == 0:
                mean_per_unclustered_client_top1_accuracy = 0
                median_per_unclustered_client_top1_accuracy = 0
            else:
                mean_per_unclustered_client_top1_accuracy = np.mean(unclustered_client_top1_accuracy)
                median_per_unclustered_client_top1_accuracy = np.median(unclustered_client_top1_accuracy)
                logging.info(f"all_unclustered_clients_mean_top1_accuracy: {mean_per_unclustered_client_top1_accuracy}")
                logging.info(f"all_unclustered_clients_median_top1_accuracy: {median_per_unclustered_client_top1_accuracy}")
            self.wandb.log({
                'Test/all_unclustered_clients_mean_top1_accuracy': mean_per_unclustered_client_top1_accuracy,
                'Test/all_unclustered_clients_median_top1_accuracy': median_per_unclustered_client_top1_accuracy,
            }, step=self.round[0])
            
            mean_per_client_top1_accuracy = np.mean(all_per_client_top1_accuracy)
            median_per_client_top1_accuracy = np.median(all_per_client_top1_accuracy)
            self.wandb.log({
                'Test/all_clients_mean_top1_accuracy': mean_per_client_top1_accuracy,
                'Test/all_clients_median_top1_accuracy': median_per_client_top1_accuracy,
            }, step=self.round[0])
            logging.info(f"all_clients_mean_top1_accuracy: {mean_per_client_top1_accuracy}")
            logging.info(f"all_clients_median_top1_accuracy: {median_per_client_top1_accuracy}")
            
    
    def log_test_result_wandb(self, cluster_id=0):
        """Log testing result on WanDB
        """
        if self.round[cluster_id] % self.args.eval_interval == 0 and self.wandb != None:
            self.wandb.log({
                f'Test/round_to_mean_client_top1_accuracy_cluster{cluster_id}': np.mean([x[1] for x in self.per_client_test_top1[cluster_id]]),
                f'Test/round_to_top1_accuracy_cluster{cluster_id}': self.testing_history[cluster_id]['perf'][self.round[cluster_id]]['top_1'],
                f'Test_loss/round_to_loss_cluster{cluster_id}': self.testing_history[cluster_id]['perf'][self.round[cluster_id]]['loss'],
            }, step=self.round[cluster_id])

    def save_model(self, cluster_id=0):
        """Save model to the wandb server if enabled
        
        """
        if parser.args.save_checkpoint and self.last_saved_round[cluster_id] < self.round[cluster_id]:
            self.last_saved_round[cluster_id] = self.round[cluster_id]
            checkpoint_file_path = self.checkpoint_path + f'_cluster{cluster_id}.pth'
            try:
                self.model_wrapper[cluster_id].save_checkpoint(checkpoint_file_path)
                # when the global cluster calls this function, also save the current clusters and feasible clients
                if cluster_id == 0:
                    self.client_manager.saveFeasibleClients(file_path=self.checkpoint_path+'_feasible_clients.pkl',\
                                                            curr_round=self.round[0])
            except Exception as e:
                logging.info(f"Failed to save model checkpoint: {e}")

    def deserialize_response(self, responses):
        """Deserialize the response from executor

        Args:
            responses (byte stream): Serialized response from executor.

        Returns:
            string, bool, or bytes: The deserialized response object from executor.
        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """ Serialize the response to send to server upon assigned job completion

        Args:
            responses (ServerResponse): Serialized response from server.

        Returns:
            bytes: The serialized response object to server.

        """
        return pickle.dumps(responses)

    def testing_completion_handler(self, client_id, results, cluster_id=0):
        """Each executor will handle a subset of testing dataset

        Args:
            client_id (int): The client id.
            results (dictionary): The client test results.

        """
        logging.info(f"Cluster {cluster_id} finished testing: {results}")
        if client_id in self.test_reported[cluster_id]:
            logging.info(f"Cluster {cluster_id} already have reported testing results from executor {client_id}, skip")
            return False
        self.test_reported[cluster_id].append(client_id)
        per_client_record = results['per_client_record']
        results = results['results']

        # List append is thread-safe
        self.test_result_accumulator[cluster_id].append(results)
        self.total_test_top1[cluster_id] += results['top_1']
        self.total_test_top5[cluster_id] += results['top_5']
        self.total_test_samples[cluster_id] += results['test_len']
        self.per_client_test_top1[cluster_id].extend(per_client_record)

        # Have collected all testing results

        if len(self.test_result_accumulator[cluster_id]) == len(self.executors):
            self.aggregate_test_result(cluster_id)
            if len(self.per_client_test_top1[cluster_id]) != len(self.cluster_clients_to_test[cluster_id]):
                # there are likely some clients skipped due to not having enough data
                logging.info(f"Cluster {cluster_id} has {len(self.cluster_clients_to_test[cluster_id])-len(self.per_client_test_top1[cluster_id])} clients skipped in testing")
                self.cluster_clients_to_test[cluster_id] = [x[0] for x in self.per_client_test_top1[cluster_id]]
            self.log_test_result_wandb(cluster_id)

            self.save_model(cluster_id)
            return True
        return False

    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes by adding grpc request event
        (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.

        """
        self.broadcast_events_queue.append(event)

    def dispatch_client_events(self, event, clients=None):
        """Issue tasks (events) to clients

        Args:
            event (string): grpc event (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
            clients (list of int): target client ids for event.

        """
        if clients is None:
            clients = self.sampled_executors

        for client_id in clients:
            self.individual_client_events[client_id].append(event)

    def get_client_conf(self, client_id):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: TorchClient training config.

        """
        conf = {
            'learning_rate': self.args.learning_rate,
        }
        return conf

    def create_client_task(self, executor_id, cluster_id = 0, reconnect=False):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        if not reconnect:
            next_client_id = self.resource_manager.get_next_task(executor_id, cluster_id)
            self.executor_current_clientid[executor_id] = (next_client_id, cluster_id)
        else:
            logging.info(f"Reconnect Executor {executor_id} to run client {self.executor_current_clientid[executor_id][0]} of cluster {self.executor_current_clientid[executor_id][1]}")
            next_client_id = self.executor_current_clientid[executor_id][0]
            cluster_id = self.executor_current_clientid[executor_id][1]
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        if next_client_id is not None:
            config = self.get_client_conf(next_client_id)
            train_config = {'client_id': next_client_id, 'task_config': config, 'cluster_id': cluster_id, 
                            'round': self.round[cluster_id]}

        return train_config, self.model_wrapper[cluster_id].get_weights()

    def get_test_client_set(self, cluster_id=0):
        """Get the set of clients of a specific cluster for testing

        Args:
            cluster_id (int): The cluster id.

        Returns:
            None: update cluster_clients_to_test[cluster_id] with the selected clients for testing

        """
        if self.args.test_client_ratio == 1.0 or self.num_cluster == 0:
            client_list = self.select_participants(
                select_num_participants=self.args.num_participants, overcommitment=1.0, cluster_id=cluster_id,
                test=True, check_rank_avail=(self.args.data_mode != "all"))
            self.cluster_clients_to_test[cluster_id] = client_list
        else:
            if cluster_id == 0:
                self.cluster_clients_to_test[0] = []
                # first, select subset of clients from each cluster
                for cid in range(1, self.num_cluster+1):
                    client_list = self.select_participants(
                        select_num_participants=self.args.num_participants, overcommitment=1.0, cluster_id=cid,
                        test=True, check_rank_avail=(self.args.data_mode != "all"))
                    self.cluster_clients_to_test[cid] = client_list
                    self.cluster_clients_to_test[0] += client_list
                    logging.info(f"Cluster {cid} selected {len(self.cluster_clients_to_test[cid])} clients for testing: {client_list}")
                # then, if there are unclustered clients, select them
                unclustered_clients = self.client_manager.getUnclusteredClients()
                if len(unclustered_clients) > 0:
                    self.rng.shuffle(unclustered_clients)
                    # random.shuffle(unclustered_clients)
                    self.cluster_clients_to_test[0] += \
                        unclustered_clients[:max((min(50, len(unclustered_clients))), \
                                                 int(len(unclustered_clients)*self.args.test_client_ratio))]
                    logging.info(f"Cluster 0 selected {max(min(50, len(unclustered_clients)), int(len(unclustered_clients)*self.args.test_client_ratio))} additional \
unclustered clients for testing")
            # for other clusters, no need to reselect again cause their test clients are already selected 
            # when the global cluster calls this function

    def get_test_config(self, client_id, cluster_id = 0):
        """FL model testing on clients, developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: The testing config for new task.

        """
        client_list = self.cluster_clients_to_test[cluster_id]
        return {'client_id': client_list}

    def get_shutdown_config(self, client_id):
        """Shutdown config for client, developers can further define personalized client config here.

        Args:
            client_id (int): TorchClient id.

        Returns:
            dictionary: Shutdown config for new task.

        """
        return {'client_id': client_id}

    def add_event_handler(self, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first.

        Args:
            client_id (int): The client id.
            event (string): grpc event MODEL_TEST or UPLOAD_MODEL.
            meta (dictionary or string): Meta message for grpc communication, could be event.
            data (dictionary): Data transferred in grpc communication, could be model parameters, test result.

        """
        self.sever_events_queue.append((client_id, event, meta, data))

    def CLIENT_REGISTER(self, request, context):
        """FL TorchClient register to the aggregator

        Args:
            request (RegisterRequest): Registeration request info from executor.

        Returns:
            ServerResponse: Server response to registeration request

        """

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id uses the same executor_id (VMs) in simulations
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        if executor_id not in self.individual_client_events:
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info(f"Previous client: {executor_id} resumes connecting")

        # We can customize whether to admit the clients here
        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
                                          meta=dummy_data, data=dummy_data)

    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft()
            event_type, cluster_id = commons.decode_clusterid(current_event)
            if event_type == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    executor_id, cluster_id)
                if response_msg is None:
                    current_event = commons.encode_clusterid(commons.DUMMY_EVENT)
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN)
            elif event_type == commons.EXECUTOR_RECONNECT:
                response_msg, response_data = self.create_client_task(
                    executor_id, cluster_id, reconnect=True)
                if response_msg is None:
                    current_event = commons.encode_clusterid(commons.DUMMY_EVENT)
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(commons.CLIENT_TRAIN)
                else:
                    current_event = commons.encode_clusterid(commons.CLIENT_TRAIN, response_msg['cluster_id'])
            elif event_type == commons.CLUSTER_SPLIT:
                logging.info(f"new_cluster_id {cluster_id} to start with {self.new_cluster_mapping.get(cluster_id, 0)} model at round {self.global_round}")
                # round is self.global_round-1 as the per-cluster round then get updated in UPDATE_MODEL
                response_msg = {'new_cluster_id': cluster_id, \
                                'prev_cluster_id': self.new_cluster_mapping.get(cluster_id, 0), \
                                'round': self.global_round-1}
            elif event_type == commons.MODEL_TEST:
                response_msg = self.get_test_config(client_id, cluster_id)
            elif event_type == commons.UPDATE_MODEL:
                response_msg = {'round': self.global_round}
                response_data = self.model_wrapper[cluster_id].get_weights()
            elif event_type == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                              meta=response_msg, data=response_data)

        # if in simulation mode, check if the timer has expired (round takes longer than 10 minutes)
        if self.experiment_mode == commons.SIMULATION_MODE:
            if self.curr_round_start_time and \
                time.time() - self.curr_round_start_time > self.args.round_time_limit:
                logging.info(f"Round {self.round[0]} time limit reached, force to complete")
                if self.global_testing:
                    self.test_complete_cluster_count += 1
                    if self.test_complete_cluster_count == self.num_cluster + 1:
                        # clear all event queues
                        for i in range(0, self.num_cluster+1):
                            self.resource_manager.clear_cluster(cluster_id=i)
                        # log the mean testing result of all clusters
                        self.log_all_clusters_mean_test_result()
                        self.global_testing = False
                        self.test_complete_cluster_count = 0
                        for i in range(0, self.num_cluster+1):
                            self.round_completion_handler(cluster_id=i, after_test=True)
                            if len(self.new_cluster_mapping) > 0:
                                # just globally reclustered, old clusters don't need to proceed training
                                break
                        logging.info(f"forced all clusters done testing")
                else:
                    self.start_next_round(\
                        cluster_id=0, force_start=True)

        return response

    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task.

        Args:
            request (CompleteRequest): Complete request info from executor.

        Returns:
            ServerResponse: Server response to job completion request

        """

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result
        event_type, cluster_id = commons.decode_clusterid(event)

        if event_type == commons.CLIENT_TRAIN:
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            if execution_status is False:
                logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")

            if self.resource_manager.has_next_task(executor_id, cluster_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                if commons.encode_clusterid(commons.CLIENT_TRAIN, cluster_id) not in self.individual_client_events[executor_id]:
                    self.individual_client_events[executor_id].append(
                        commons.encode_clusterid(commons.CLIENT_TRAIN, cluster_id))
                    
        elif event_type == commons.EXECUTOR_RECONNECT:
            self.individual_client_events[executor_id].append(
                        commons.encode_clusterid(commons.EXECUTOR_RECONNECT, cluster_id))

        elif event_type in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            self.add_event_handler(
                executor_id, event, meta_result, data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")

        return self.CLIENT_PING(request, context)
    
    def start_next_round(self, cluster_id, force_start=False):
        """Start the next round of training and testing

        """
        # the global model cluster 0 should be free to continue
        if self.num_cluster == 0 and cluster_id == 0:
            coordinator_start_time = time.time()
            self.global_round += 1
            self.resource_manager.clear_cluster(cluster_id=cluster_id)
            self.round_completion_handler(cluster_id=cluster_id)
            logging.info(f"Coordinator time cost (select+create_task+recluster): {time.time() - coordinator_start_time} s")
        else:    
            self.round_completion_cluster_count += 1
            if force_start or self.round_completion_cluster_count == self.num_cluster + 1:
                coordinator_start_time = time.time()
                # clear all event queues
                for i in range(0, self.num_cluster+1):
                    self.resource_manager.clear_cluster(cluster_id=i)
                # clear counter
                self.round_completion_cluster_count = 0

                # all clusters have finished, let all to continue
                self.global_round += 1
                for i in range(0, self.num_cluster+1):
                    self.round_completion_handler(cluster_id=i)
                    if len(self.new_cluster_mapping) > 0:
                        # just globally reclustered, old clusters don't need to proceed training
                        break
                logging.info(f"Coordinator time cost (select+create_task+recluster): {time.time() - coordinator_start_time} s")


    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()
                event_type, cluster_id = commons.decode_clusterid(current_event)

                if event_type in (commons.UPDATE_MODEL, commons.MODEL_TEST, commons.CLUSTER_SPLIT):
                    self.dispatch_client_events(current_event)

                elif event_type == commons.START_ROUND:
                    self.dispatch_client_events(commons.encode_clusterid(commons.CLIENT_TRAIN, cluster_id))

                elif event_type == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()
                event_type, cluster_id = commons.decode_clusterid(current_event)

                if event_type == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data), cluster_id)
                    if len(self.stats_util_accumulator[cluster_id]) == self.tasks_round[cluster_id]:
                        self.start_next_round(cluster_id)

                elif event_type == commons.MODEL_TEST:
                    cluster_test_complete = self.testing_completion_handler(
                        client_id, self.deserialize_response(data), cluster_id)
                    if cluster_test_complete:
                        if self.num_cluster == 0 and cluster_id == 0:
                            self.resource_manager.clear_cluster(cluster_id=cluster_id)
                            # log the mean testing result of all clusters
                            self.log_all_clusters_mean_test_result()
                            self.round_completion_handler(cluster_id=cluster_id, after_test=True)
                        else:
                            self.test_complete_cluster_count += 1
                            if self.test_complete_cluster_count == self.num_cluster + 1:
                                # clear all event queues
                                for i in range(0, self.num_cluster+1):
                                    self.resource_manager.clear_cluster(cluster_id=i)
                                # log the mean testing result of all clusters
                                self.log_all_clusters_mean_test_result()
                                self.test_complete_cluster_count = 0
                                self.global_testing = False
                                for i in range(0, self.num_cluster+1):
                                    self.round_completion_handler(cluster_id=i, after_test=True)
                                    if len(self.new_cluster_mapping) > 0:
                                        break
                                logging.info(f"all clusters done testing")

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

    def stop(self):
        """Stop the aggregator
        """
        logging.info(f"Terminating the aggregator ...")
        if self.wandb != None:
            # # add wandb logs
            # for record in self.training_logs_for_wandb:
            #     self.wandb.log(record[0], step=record[1])
            # for record in self.testing_logs_for_wandb:
            #     self.wandb.log(record[0], step=record[1])
            # time.sleep(5)
            self.wandb.finish()
        time.sleep(5)


if __name__ == "__main__":
    aggregator = Aggregator(parser.args)
    aggregator.run()
