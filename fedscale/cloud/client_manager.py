import logging
import math
import pickle
from random import Random
import random
from typing import Dict, List

from fedscale.cloud.internal.client_metadata import ClientMetadata
from thirdparty.oort.oort import create_training_selector

import time
import os
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import copy
import heapq
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from fedscale.cloud.client_clustering import ClusterManager

class ClientManager:

    def __init__(self, mode, args, sample_seed=233):
        self.client_metadata = {}
        self.client_on_hosts = {}
        self.mode = mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more

        client_rank_to_avail_round_file = f"./workspace/records/{args.data_set}/client_rank_2_avail_rounds.pkl"
        # if data_mode is all, then all clients are available at all rounds
        if args.data_mode != "all" and os.path.isfile(client_rank_to_avail_round_file):
            with open(client_rank_to_avail_round_file, 'rb') as fin:
                self.client_rank_to_avail_round = pickle.load(fin)
                logging.info(f"loaded client_rank_to_avail_round from {client_rank_to_avail_round_file}")
        else:
            self.client_rank_to_avail_round = {}
            logging.info("WARNING: empty client_rank_to_avail_round")

        client_rank_to_distribution_at_round_file = f"./workspace/records/{args.data_set}/client_rank_2_distribution_at_round.pkl"
        if args.data_mode == "all":
            client_rank_to_distribution_at_round_file = f"./workspace/records/{args.data_set}/client_rank_2_distribution_at_round_allsamples.pkl"
        if os.path.isfile(client_rank_to_distribution_at_round_file):
            with open(client_rank_to_distribution_at_round_file, 'rb') as fin:
                self.client_rank_to_distribution_at_round = pickle.load(fin)
                logging.info(f"loaded client_rank_to_distribution_at_round_file from {client_rank_to_distribution_at_round_file}")
        else:
            self.client_rank_to_distribution_at_round = {}
            logging.info("empty client_rank_to_distribution_at_round")

        self.cluster_to_center = {}
        logging.info("created empty cluster_to_center")

        self.jensenshannon_distances = {}
        self.avg_center_jensenshannon = 0
        self.current_clusters = []
        self.client_rank_to_duration = {}
        self.fast_clients = {}
        self.slow_clients = {}
        self.client_rank_to_utility = {}
        self.delete_small_cluster = True

        self.ucb_sampler = None

        if self.mode == 'oort':
            self.ucb_sampler = [create_training_selector(args=args)]

        self.feasibleClients = [[]]
        self.not_yet_feasibleClients = []
        self.clients_with_gradient = set()
        self.data_drifted_clients = {}

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.numpy_rng = np.random.default_rng(seed=sample_seed)
        self.count = 0
        self.feasible_samples = 0
        self.user_trace = None
        self.args = args

        self.malicious_ratio = args.malicious_ratio
        self.malicious_clients = set()
        self.malicious_clients_true_distribution = {}

        self.cluster_manager = ClusterManager(args)
        logging.info(f"created cluster manager")

        if args.device_avail_file is not None:
            with open(args.device_avail_file, 'rb') as fin:
                self.user_trace = pickle.load(fin)
            self.user_trace_keys = list(self.user_trace.keys())

        self.euclidean_square = distance_metric(type_metric.EUCLIDEAN_SQUARE) # this is the default metric of pyclustering kmeans

    def load_saved_states(self, states):
        logging.info(f"Client manager loading saved states")
        try:
            logging.info(f"initilized number of feasibleClients: {len(self.feasibleClients[0])}, number of infeasibleClients: {len(self.not_yet_feasibleClients)}")
            prev_feasibleClients = self.feasibleClients[0]
            self.feasibleClients = states['feasibleClients']
            self.not_yet_feasibleClients = list(set(prev_feasibleClients + self.not_yet_feasibleClients) - set(self.feasibleClients[0]))
            logging.info(f"loaded number of feasibleClients: {len(self.feasibleClients[0])}, number of infeasibleClients: {len(self.not_yet_feasibleClients)}")
            self.current_clusters = states['current_clusters']
            self.malicious_clients = states['malicious_clients']
            self.malicious_clients_true_distribution = {}
            # the clusters saved are based on the round immediately before the testing round
            curr_round = states['round'] - 1
            distribution_prob = {}
            for client_id in self.feasibleClients[0] + self.not_yet_feasibleClients:
                # update the label distribution
                unique_id = self.getUniqueId(0, client_id)
                client_finish_round = max(self.client_rank_to_distribution_at_round[client_id].keys())
                if curr_round > client_finish_round:
                    new_label_counts = self.client_rank_to_distribution_at_round[client_id][client_finish_round]
                else:
                    for shift_round in sorted(self.client_rank_to_distribution_at_round[client_id].keys(), reverse=True):
                        if curr_round >= shift_round:
                            new_label_counts = self.client_rank_to_distribution_at_round[client_id][shift_round]
                            break
                if client_id in self.malicious_clients:
                    self.malicious_clients_true_distribution[client_id] = copy.deepcopy(new_label_counts)
                    self.rng.shuffle(new_label_counts)
                    logging.info(f"malicious client {client_id} permuted own distribution")
                self.client_metadata[unique_id].register_distribution(new_label_counts)
                if sum(new_label_counts) > 0:
                    if client_id in self.not_yet_feasibleClients:
                        logging.info(f"client {client_id} becomes feasible")
                        self.not_yet_feasibleClients.remove(client_id)
                        self.feasibleClients[0].append(client_id)
                    distribution_prob[client_id] = [x/sum(new_label_counts) for x in new_label_counts]
                elif client_id not in self.not_yet_feasibleClients:
                    logging.info(f"client {client_id} becomes infeasible")
                    self.not_yet_feasibleClients.append(client_id)
                    for cluster_id in [0] + self.current_clusters:
                        if client_id in self.feasibleClients[cluster_id]:
                            logging.info(f"remove client {client_id} from cluster {cluster_id}")
                            self.feasibleClients[cluster_id].remove(client_id)
            # update the cluster center
            self.cluster_to_center[0] = np.median(np.stack([distribution_prob[client_id] \
                                    for client_id in self.feasibleClients[0]]), axis=0).tolist()
            for cluster_id in self.current_clusters:
                self.cluster_to_center[cluster_id] = \
                    np.median(np.stack([distribution_prob[client_id] for client_id \
                        in self.feasibleClients[cluster_id]]), axis=0).tolist()
            logging.info(f"loaded feasibleClients and current_clusters")
        except Exception as e:
            logging.info(f"error in loading saved states: {e}")
    
    def register_client(self, host_id: int, client_id: int, size: int, speed: Dict[str, float],
                        duration: float = 1, curr_round=0) -> None:
        """Register client information to the client manager.

        Args:
            host_id (int): executor Id.
            client_id (int): client Id.
            size (int): number of samples on this client.
            speed (Dict[str, float]): device speed (e.g., compuutation and communication).
            duration (float): execution latency.

        """
        uniqueId = self.getUniqueId(host_id, client_id)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(
            client_id) % len(self.user_trace)]]
        user_distribution = None if client_id not in self.client_rank_to_distribution_at_round else \
            self.client_rank_to_distribution_at_round[client_id][0]
        if user_distribution is None:
            logging.info(f"client {client_id} has no distribution, filtered out")
            return

        self.client_metadata[uniqueId] = ClientMetadata(host_id, client_id, speed, user_trace, user_distribution)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            if self.args.data_mode != "all":
                if (user_distribution is not None) and sum(user_distribution) > 0:
                    self.feasibleClients[0].append(client_id)
                    logging.info(f"register client {client_id} with {size} samples into cluster 0")
                else:
                    self.not_yet_feasibleClients.append(client_id)
                    logging.info(f"register client {client_id} with {size} samples into not_yet_feasibleClients")
            else:
                self.feasibleClients[0].append(client_id)
                logging.info(f"All data always available, register client {client_id} with {size} samples into cluster 0")
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps * self.args.batch_size),
                             'duration': duration,
                             }
                self.ucb_sampler[0].register_client(client_id, feedbacks=feedbacks)

            if self.malicious_ratio > 0.0:
                # randomly determine whether this client is malicious
                if self.rng.random() < self.malicious_ratio:
                    self.malicious_clients.add(client_id)
                    self.malicious_clients_true_distribution[client_id] = copy.deepcopy(user_distribution)
        else:
            logging.info(f"client {client_id} has {size} samples and filtered out")
            del self.client_metadata[uniqueId]

    def getAllClients(self):
        return self.feasibleClients[0]
    
    def getAllInfeasibleClients(self):
        return self.not_yet_feasibleClients

    def getAllClientsLength(self):
        return len(self.feasibleClients[0])
    
    def getClusterClientsLength(self, cluster_id):
        return len(self.feasibleClients[cluster_id])
    
    def getClusterFeasibleClients(self, cluster_id):
        return self.feasibleClients[cluster_id]
    
    def getUnclusteredClients(self):
        unclustered_clients = set(self.feasibleClients[0])
        for cluster_id in self.current_clusters:
            if cluster_id == 0:
                logging.info(f"WARNING: cluster 0 appears in current_clusters, skip")
            elif cluster_id > len(self.feasibleClients) - 1:
                logging.info(f"WARNING: cluster {cluster_id} not in feasibleClients, skip")
            else:
                unclustered_clients = set(unclustered_clients) - set(self.feasibleClients[cluster_id])
        logging.info(f"{len(unclustered_clients)} unclustered clients")
        return list(unclustered_clients)

    def getClient(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)]
    
    def saveFeasibleClients(self, file_path, curr_round):
        with open(file_path, 'wb') as f:
            pickle.dump({'round': curr_round, 'current_clusters': self.current_clusters, \
                         'feasibleClients': self.feasibleClients, 'malicious_clients': self.malicious_clients}, f)
        logging.info(f"saved feasible clients into {file_path}")

    def registerDuration(self, client_id, batch_size, local_steps, upload_size, download_size):
        if self.getUniqueId(0, client_id) not in self.client_metadata:
            # this client has been filtered out due to lack of samples
            return
        exe_cost = self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time(
            batch_size=batch_size, local_steps=local_steps,
            upload_size=upload_size, download_size=download_size
        )
        if self.mode == "oort" and self.getUniqueId(0, client_id) in self.client_metadata:
            self.ucb_sampler[0].update_duration(
                client_id, exe_cost['computation'] + exe_cost['communication'])
        else:
            self.client_rank_to_duration[client_id] = exe_cost['computation'] + exe_cost['communication']

    def get_completion_time(self, client_id, batch_size, local_steps, upload_size, download_size):
        return self.client_metadata[self.getUniqueId(0, client_id)].get_completion_time(
            batch_size=batch_size, local_steps=local_steps,
            upload_size=upload_size, download_size=download_size
        )

    def registerSpeed(self, host_id, client_id, speed):
        uniqueId = self.getUniqueId(host_id, client_id)
        self.client_metadata[uniqueId].speed = speed

    def registerScore(self, client_id, reward, auxi=1.0, time_stamp=0, duration=1., success=True):
        self.register_feedback(client_id, reward, auxi=auxi, time_stamp=time_stamp, duration=duration, success=success)

    def register_feedback(self, client_id: int, reward: float, auxi: float = 1.0, time_stamp: float = 0,
                          duration: float = 1., success: bool = True, 
                          cluster_id = 0, prev_weight = None, new_weight=None, top1_accu=None, top5_accu=None,
                          representation=None, ignore_sampler=False) -> None:
        """Collect client execution feedbacks of last round.

        Args:
            client_id (int): client Id.
            reward (float): execution utilities (processed feedbacks).
            auxi (float): unprocessed feedbacks.
            time_stamp (float): current wall clock time.
            duration (float): system execution duration.
            success (bool): whether this client runs successfully.

        """
        if not ignore_sampler:
            # currently, we only use distance as reward
            if self.mode == "oort":
                feedbacks = {
                    'reward': reward,
                    'duration': duration,
                    'status': True,
                    'time_stamp': time_stamp
                }

                self.ucb_sampler[cluster_id].update_client_util(client_id, feedbacks=feedbacks)
            # register per-client current training loss
            if self.mode == "train_loss_reward":
                if client_id in self.client_rank_to_utility:
                    epsilon = 0.1
                    self.client_rank_to_utility[client_id] = (1-epsilon) * self.client_rank_to_utility[client_id] \
                        + epsilon * reward
                    logging.info(f"update client {client_id} reward to {self.client_rank_to_utility[client_id]}")
                else:
                    logging.info(f"register client {client_id} with reward {reward}")
                    self.client_rank_to_utility[client_id] = reward
            else:
                self.client_rank_to_utility[client_id] = reward

        if new_weight is not None:
            self.client_metadata[self.getUniqueId(0, client_id)].register_gradient(new_weight)
            logging.info(f"register client {client_id} gradient")
            self.clients_with_gradient.add(client_id)
        if top1_accu is not None:
            self.client_metadata[self.getUniqueId(0, client_id)].register_accuracy(top1_accu, top5_accu)
        if representation is not None:
            self.client_metadata[self.getUniqueId(0, client_id)].register_representation(representation)
            logging.info(f"register client {client_id} representation")

    def register_data_drifted_client(self, client_id, cluster_id = 0, recalculate_distance = False):
        if cluster_id not in self.data_drifted_clients:
            self.data_drifted_clients[cluster_id] = [client_id]
        else:
            self.data_drifted_clients[cluster_id].append(client_id)

    def registerClientScore(self, client_id, reward):
        self.client_metadata[self.getUniqueId(0, client_id)].register_reward(reward)

    def get_score(self, host_id, client_id):
        uniqueId = self.getUniqueId(host_id, client_id)
        return self.client_metadata[uniqueId].get_score()

    def getClientsInfo(self):
        clientInfo = {}
        for i, client_id in enumerate(self.client_metadata.keys()):
            client = self.client_metadata[client_id]
            clientInfo[client.client_id] = client.distance
        return clientInfo

    def next_client_id_to_run(self, host_id):
        init_id = host_id - 1
        lenPossible = len(self.feasibleClients)

        while True:
            client_id = str(self.feasibleClients[init_id])
            csize = self.client_metadata[client_id].size
            if csize >= self.filter_less and csize <= self.filter_more:
                return int(client_id)

            init_id = max(
                0, min(int(math.floor(self.rng.random() * lenPossible)), lenPossible - 1))

    def getUniqueId(self, host_id, client_id):
        return str(client_id)
        # return (str(host_id) + '_' + str(client_id))

    def clientSampler(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size

    def clientOnHost(self, client_ids, host_id):
        self.client_on_hosts[host_id] = client_ids

    def getCurrentclient_ids(self, host_id):
        return self.client_on_hosts[host_id]

    def getClientLenOnHost(self, host_id):
        return len(self.client_on_hosts[host_id])

    def getClientSize(self, client_id):
        return self.client_metadata[self.getUniqueId(0, client_id)].size
        
    def reset_sampler(self, cluster_id):
        if self.mode == "oort":
            if cluster_id >= len(self.ucb_sampler):
                while cluster_id >= len(self.ucb_sampler):
                    logging.info(f"NOTE: create a new sampler for cluster {cluster_id} in reset_samplers")
                    # start with a copy of the global model sampler
                    self.ucb_sampler.append(copy.deepcopy(self.ucb_sampler[0]))
                    # reset all clients in the new cluster
                    self.ucb_sampler[cluster_id].reset_all_client_util()
            else:
                if self.ucb_sampler[cluster_id] is None:
                    # replace the placeholder with a copy of the global model sampler
                    logging.info(f"NOTE: create a new sampler for cluster {cluster_id} in reset_samplers")
                    self.ucb_sampler[cluster_id] = copy.deepcopy(self.ucb_sampler[0])
                # reset all clients in the old sampler
                self.ucb_sampler[cluster_id].reset_all_client_util()

    def partition_fast_slow_client(self, cluster_id):
        cluster_clients = [(client_rank, self.client_rank_to_duration[client_rank]) \
                           for client_rank in self.feasibleClients[cluster_id]]
        # sort cluster_clients based on round duration
        cluster_clients.sort(key=lambda x : x[1])
        self.fast_clients[cluster_id] = [x[0] for x in cluster_clients[:len(cluster_clients)//2]]
        self.slow_clients[cluster_id] = [x[0] for x in cluster_clients[len(cluster_clients)//2:]]
    
    def global_clustering(self, curr_round=0, initial=False, use_kmeans=True):
        logging.info(f"In global_clustering")
        if initial:
            # need to first update client distribution
            self.global_client_update_label_counts(round=curr_round)
            # avoid keeping a record for cluster 0
            self.data_drifted_clients = {}
        start_time = time.time()
        all_clients = self.feasibleClients[0]
        
        # extract available clients
        if len(self.client_rank_to_avail_round) > 0:
            all_clients_to_cluster = [c for c in all_clients \
                                  if curr_round in self.client_rank_to_avail_round[c]]
        else:
            all_clients_to_cluster = all_clients
        # remove unavailable clients
        logging.info(f"In global_clustering, all_clients_to_cluster has {len(all_clients_to_cluster)} clients")
        distribution_prob = {}
        for client_id in all_clients_to_cluster:
            client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
            distribution_prob[client_id] = [x/sum(client_label_counts) for x in client_label_counts]
        current_clusters = {}

        current_clusters = self.cluster_manager.global_clustering_helper(\
            all_clients_to_cluster, self.client_metadata, distribution_prob, self.feasibleClients,\
            self.cluster_to_center, use_kmeans=use_kmeans)
    
        try:
            if len(self.feasibleClients) < len(current_clusters)+1:
                # expand self.feasibleClients if creating more clusters
                for _ in range(len(self.feasibleClients), len(current_clusters)+1):
                    logging.info(f"expand self.feasibleClients by 1")
                    self.feasibleClients.append([])
                    # expand Oort sampler if needed
                    if self.mode == "oort":
                        self.ucb_sampler.append(None)
            # update current clusters key to start from 1
            curr_cluster_idx = 1
            new_current_clusters = {}
            for v in current_clusters.values():
                new_current_clusters[curr_cluster_idx] = v
                self.feasibleClients[curr_cluster_idx] = v
                # reset Oort sampler if needed
                if self.mode == "oort":
                    self.reset_sampler(curr_cluster_idx)
                curr_cluster_idx += 1
            current_clusters = new_current_clusters
            # calculate the cluster centers
            overall_distribution_prob_all = np.stack([distribution_prob[client_id] for client_id in all_clients_to_cluster])
            self.cluster_to_center[0] = np.median(overall_distribution_prob_all, axis=0).tolist()
            for cluster_id, v in current_clusters.items():
                overall_distribution_prob = np.stack([distribution_prob[client_id] for client_id in v])
                self.cluster_to_center[cluster_id] = np.mean(overall_distribution_prob, axis=0).tolist()
            if len(current_clusters) == 1:
                logging.info(f"only one cluster, set avg_center_jensenshannon to 0")
                self.avg_center_jensenshannon = 0
            else:
                # calculate average wasserstein_distance between cluster centers
                sum_w_distances = []
                for i in range(1, len(current_clusters)):
                    if (not self.args.use_l1_distance) and sum(self.cluster_to_center[i]) <= 0:
                        logging.info(f"cluster {i} has non-positive sum ({self.cluster_to_center[i]}), skip calculating jensenshannon distance")
                        continue
                    for j in range(i + 1, len(current_clusters)+1):
                        if (not self.args.use_l1_distance) and sum(self.cluster_to_center[j]) <= 0:
                            logging.info(f"cluster {j} has non-positive sum ({self.cluster_to_center[j]}), skip calculating jensenshannon distance")
                            continue
                        if self.args.use_l1_distance:
                            sum_w_distances.append(\
                                np.linalg.norm(np.array(self.cluster_to_center[i]) - np.array(self.cluster_to_center[j]), ord=1))
                        else:
                            sum_w_distances.append(jensenshannon(self.cluster_to_center[i],self.cluster_to_center[j]))
                self.avg_center_jensenshannon = sum(sum_w_distances) / max(1,len(sum_w_distances))

            logging.info(f"global_clustering took {time.time() - start_time} seconds.")
            logging.info(f"created {len(current_clusters)} clusters with avg_center_jensenshannon {self.avg_center_jensenshannon}")
            logging.info(f"cluster sizes: {[(k, len(v)) for k, v in current_clusters.items()]}")
            
            self.current_clusters = list(current_clusters.keys())
            logging.info(f"current_clusters: {self.current_clusters}")
        except Exception as e:
            logging.info(f"error in global_clustering: {e}")
            return list(current_clusters.keys()), distribution_prob

        return list(current_clusters.keys()), distribution_prob
    
    def jl_transform(self, vectors, device, dim, individual=False):
        try:
            logging.info(f"jl_transform of {vectors.dtype} vectors size {vectors.size()} with dim {dim}")
            vectors = vectors.to(device)
            if not vectors.is_cuda:
                logging.info(f"jl_transform on cpu")
            res = torch.zeros((vectors.shape[0], dim), device=device)
            for i in range(dim):

                try:
                    direction = torch.randn(vectors.shape[1], device=device) / dim
                    
                    res[:,i] = vectors @ direction
                except Exception as e:
                    logging.info(f"jl_transform exception: {e}")
            if individual:
                return res.view(-1)
            return res

        except Exception as e:
            logging.info(f"jl_transform exception: {e}")
            return None


    def find_optimal_cluster_numbers(self, samples, ksearch_type="kmedians", kmin=2, kmax=10):
        if len(samples) <= 2:
            # just put all these clients into one cluster
            return 1
        amount = 0
        retry = 0
        while amount < kmin and retry < 100:
            search_instance = silhouette_ksearch(samples, kmin=kmin, kmax=min(kmax, len(samples)), 
                algorithm=silhouette_ksearch_type.KMEDIANS if ksearch_type=="kmedians" else silhouette_ksearch_type.KMEANS).process()
            amount = search_instance.get_amount()
            scores = search_instance.get_scores()
            retry += 1
        return max(1, amount)
    
    def need_gradient_based_global_recluster(self, client_features, force_incremental=False):
        try:
            logging.info(f"In need_gradient_based_global_recluster")
            if len(self.current_clusters) <= 1:
                logging.info(f"too few clusters, need to recluster")
                return True
            for cluster_id in self.current_clusters:
                if len(self.feasibleClients[cluster_id]) == 0:
                    logging.info(f"cluster {cluster_id} has no clients, need to recluster")
                    return True
                if len(self.feasibleClients[cluster_id]) > self.args.max_cluster_size_ratio * len(self.feasibleClients[0]):
                    logging.info(f"cluster {cluster_id} has too many clients, need to recluster")
                    return True
            observed_values = len(client_features[min(client_features.keys())])
            prev_cluster_to_jl_center = {}
            for cluster in self.current_clusters:
                all_jl_transformed_gradient = [0.0] * observed_values
                for client_id in self.feasibleClients[cluster]:
                    all_jl_transformed_gradient = [x+y for x, y in zip(all_jl_transformed_gradient, client_features[client_id])]
                mean_jl_transformed_gradient = [x/len(self.feasibleClients[cluster]) for x in all_jl_transformed_gradient]
                prev_cluster_to_jl_center[cluster] = mean_jl_transformed_gradient
            logging.info(f"prev_cluster_to_jl_center: {prev_cluster_to_jl_center}")
            
            # recluster deviating clients
            touched_clusters = set()
            
            for cluster_id in self.data_drifted_clients.keys():
                try:
                    deviate_clients = self.data_drifted_clients[cluster_id]
                    recluster_record = []
                    
                    if len(deviate_clients):
                        logging.info(f"cluster {cluster_id}, {len(deviate_clients)} clients reclustering")
                        
                        if cluster_id != 0:
                            touched_clusters.add(cluster_id)

                        for client_id in deviate_clients:
                            dist_to_cluster_centers = []
                            if force_incremental and len(self.feasibleClients[cluster_id]) == 1:
                                if not (cluster_id in [t[1] for t in recluster_record]):
                                    logging.info(f"force_incremental, cluster {cluster_id} has only one client, skip moving client {client_id}")
                                    break
                            # remove reclustered clients from the current cluster (unless the global cluster 0)
                            if cluster_id != 0:
                                self.feasibleClients[cluster_id].remove(client_id)
                            for cluster in self.current_clusters:
                                dist_to_cluster_centers.append((cluster, \
                                    sum((client_features[client_id][k] - prev_cluster_to_jl_center[cluster][k])**2 \
                                        for k in range(observed_values))))
                            # find the cluster with the closest center
                            dist_to_cluster_centers.sort(key = lambda x : x[1])
                            recluster_record.append((client_id, dist_to_cluster_centers[0][0]))

                        # add reclustered clients into new clusters accordingly
                        for t in recluster_record:
                            self.feasibleClients[t[1]].append(t[0])
                            touched_clusters.add(t[1])
                            logging.info(f"recluster client {t[0]} from {cluster_id} into {t[1]}")
                    

                    # reset cluster records
                    self.data_drifted_clients[cluster_id] = []
                except Exception as e:
                    logging.info(f"error in recluster cluster {cluster_id} deviating clients: {e}")

            # update cluster centers
            curr_cluster_to_jl_center = {}
            for cluster in self.current_clusters:
                all_jl_transformed_gradient = [0.0] * observed_values
                if len(self.feasibleClients[cluster]) == 0:
                    logging.info(f"cluster {cluster} become empty, assign zero center")
                    curr_cluster_to_jl_center[cluster] = all_jl_transformed_gradient
                else:
                    for client_id in self.feasibleClients[cluster]:
                        all_jl_transformed_gradient = [x+y for x, y in zip(all_jl_transformed_gradient, client_features[client_id])]
                    mean_jl_transformed_gradient = [x/len(self.feasibleClients[cluster]) for x in all_jl_transformed_gradient]
                    curr_cluster_to_jl_center[cluster] = mean_jl_transformed_gradient
                    logging.info(f"recalculate cluster {cluster} center: {mean_jl_transformed_gradient}")
            # calculate the distance shifted for each cluster
            cluster_shifted_distance = []
            for cluster in touched_clusters:
                cluster_shifted_distance.append(sum((prev_cluster_to_jl_center[cluster][k] - curr_cluster_to_jl_center[cluster][k])**2 \
                                                    for k in range(observed_values)))
            logging.info(f"cluster_shifted_distance: {cluster_shifted_distance}")
            # recalculate the average center distance
            sum_distances = []
            for i in range(1, len(self.current_clusters)):
                for j in range(i + 1, len(self.current_clusters)+1):
                    sum_distances.append(sum((curr_cluster_to_jl_center[i][k] - curr_cluster_to_jl_center[j][k])**2 \
                                            for k in range(observed_values)))
            avg_center_distance = sum(sum_distances) / max(len(sum_distances), 1)
            logging.info(f"avg_center_distance: {avg_center_distance}")

            need_global_recluster = False
            if force_incremental:
                need_global_recluster = False
            # check if any cluster center has shifted significantly
            elif max(cluster_shifted_distance) >= avg_center_distance / 3:
                need_global_recluster = True

        except Exception as e:
            logging.info(f"error in need_gradient_based_global_recluster: {e}")
            need_global_recluster = True
        return need_global_recluster
    
    def global_clustering_gradient_based(self, device, delete_small_cluster=False,
                                         increment=True, curr_round=0, initial=False):
        force_incremental = self.args.force_incremental
        logging.info(f"In global_clustering_gradient_based")
        if initial:
            # need to first update client distribution
            # for following rounds, this step is already applied in aggregator
            self.global_client_update_label_counts(round=curr_round)
            # avoid keeping a record for cluster 0
            self.data_drifted_clients = {}
        start_time = time.time()
        all_clients = self.feasibleClients[0]
        distribution_prob = {}
        for client_id in all_clients:
            client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
            distribution_prob[client_id] = [x/sum(client_label_counts) for x in client_label_counts]
        num_labels = len(distribution_prob[all_clients[0]])
        jl_transform_dimension = 5000

        try:
            gradient_record = {}
            accumulate_gradient = None
            for client_id in set(all_clients).intersection(self.clients_with_gradient):
                gradient_record[client_id] = torch.from_numpy(self.client_metadata[self.getUniqueId(0, client_id)].gradient).to('cpu')
                if accumulate_gradient is None:
                    accumulate_gradient = gradient_record[client_id]
                else:
                    accumulate_gradient = accumulate_gradient.add(gradient_record[client_id])
            # get the average gradient of clients in cluster
            avg_gradient = accumulate_gradient.div(len(self.clients_with_gradient))
            logging.info(f"{len(self.clients_with_gradient)} clients gradient mean finding time: {time.time() - start_time} seconds, shape {avg_gradient.size()}")
        except Exception as e:
            logging.info(f"error in loading gradients: {e}")
            return

        num_coordinates = len(avg_gradient)
        clustered_clients = sorted(gradient_record.keys())
        if num_coordinates > 10000:
            jl_transform_dimension = min(num_coordinates // 10, jl_transform_dimension)
            jl = self.jl_transform(
                torch.stack([gradient_record[client_id] for client_id in clustered_clients]), device, jl_transform_dimension)
            logging.info(f"jl_transform result shape {jl.size()}")
            jl_np = jl.cpu().detach().numpy()
        else:
            logging.info(f"use original gradients of dimension: {num_coordinates}")
            jl_transform_dimension = num_coordinates
            jl_np = np.array([gradient_record[client_id].flatten().numpy() for client_id in clustered_clients])
        
        current_clusters = {}
        need_global_recluster = True

        # remove clients that are unavailable from the feasibleClients
        for cluster_id in range(len(self.feasibleClients)):
            prev_clients = self.feasibleClients[cluster_id]
            self.feasibleClients[cluster_id] = [client for client in prev_clients if client in clustered_clients]
        # construct the clustered client feature dictionary
        clustered_client_features = {}
        for idx in range(len(clustered_clients)):
            clustered_client_features[clustered_clients[idx]] = jl_np[idx].tolist()
       
        if increment:
            need_global_recluster = self.need_gradient_based_global_recluster(\
                clustered_client_features, force_incremental=force_incremental)
                
        if need_global_recluster:
            k_clusters = self.find_optimal_cluster_numbers(jl_np, ksearch_type="kmeans")
            logging.info(f"optimal k_clusters: {k_clusters}")
            max_cluster_size = len(jl_np)
            retry = 0
            while max_cluster_size > len(jl_np) * self.args.max_cluster_size_ratio and retry < 100:
                # Prepare initial centers using K-Means++ method.
                initial_centers = kmeans_plusplus_initializer(jl_np, max(self.args.min_num_cluster, k_clusters)).initialize()
                # Create instance of K-Means algorithm with prepared centers.
                kmeans_instance = kmeans(jl_np, initial_centers, metric=self.euclidean_square)
                # Run cluster analysis and obtain results.
                kmeans_instance.process()
                kclusters = kmeans_instance.get_clusters()
                max_cluster_size = max([len(k) for k in kclusters])
                logging.info(f"num_cluster: {len(kclusters)}, max_cluster_size: {max_cluster_size}")
                retry += 1
            
            # extract cluster assignments
            current_clusters = {}
            for i in range(len(kclusters)):
                current_clusters[i+1] = []
                for idx in kclusters[i]:
                    current_clusters[i+1].append(clustered_clients[idx])
            logging.info(f"k clusters: {[(k, len(v)) for k, v in current_clusters.items()]}")
            
            
            if delete_small_cluster:
                # delete small clusters
                current_clusters = self.cluster_manager.move_clients_by_gain(
                    clustered_client_features, current_clusters, ksearch_type="kmeans")

        else:
            current_clusters = {}
            for cluster_id in self.current_clusters:
                if len(self.feasibleClients[cluster_id]) > 0:
                    current_clusters[cluster_id] = self.feasibleClients[cluster_id]
        
        if len(self.feasibleClients) < len(current_clusters)+1:
            # expand self.feasibleClients if creating more clusters
            for _ in range(len(self.feasibleClients), len(current_clusters)+1):
                logging.info(f"expand self.feasibleClients by 1")
                self.feasibleClients.append([])
                # expand Oort sampler if needed
                if self.mode == "oort":
                    self.ucb_sampler.append(None)

        # update current clusters key to start from 1
        curr_cluster_idx = 1
        new_current_clusters = {}
        for v in current_clusters.values():
            new_current_clusters[curr_cluster_idx] = v
            self.feasibleClients[curr_cluster_idx] = v
            # reset Oort sampler if needed
            if self.mode == "oort":
                self.reset_sampler(curr_cluster_idx)
            curr_cluster_idx += 1
        current_clusters = new_current_clusters

        # calculate the center of each cluster
        per_cluster_to_center_distance_sum_and_avg = {}
        for cluster_id, cluster in current_clusters.items():
            cluster_center = np.mean([clustered_client_features[client_id] for client_id in cluster], axis=0)
            cluster_to_center_distance_sum = 0
            for client_id in cluster:
                cluster_to_center_distance_sum += sum((clustered_client_features[client_id][k] - cluster_center[k])**2 \
                                                        for k in range(jl_transform_dimension))
            per_cluster_to_center_distance_sum_and_avg[cluster_id] = \
                (cluster_to_center_distance_sum, cluster_to_center_distance_sum / len(cluster))
        logging.info(f"per cluster pairwise distance sum and avg: {per_cluster_to_center_distance_sum_and_avg}")
        
        # calculate the cluster centers
        overall_distribution_prob_all = np.stack([distribution_prob[client_id] for client_id in all_clients])
        # self.cluster_to_center[0] = np.median(overall_distribution_prob_all, axis=0).tolist()
        self.cluster_to_center[0] = np.mean(overall_distribution_prob_all, axis=0).tolist()
        for cluster_id, v in current_clusters.items():
            overall_distribution_prob = np.stack([distribution_prob[client_id] for client_id in v])
            # self.cluster_to_center[cluster_id] = np.median(overall_distribution_prob, axis=0).tolist()
            self.cluster_to_center[cluster_id] = np.mean(overall_distribution_prob, axis=0).tolist()
        
        logging.info(f"cluster sizes: {[(k, len(v)) for k, v in current_clusters.items()]}")
        
        self.current_clusters = list(current_clusters.keys())
        logging.info(f"current_clusters: {self.current_clusters}")

        return list(current_clusters.keys()), distribution_prob, need_global_recluster
    
    def global_clustering_representation_based(self, delete_small_cluster=False,
                                         increment=True, curr_round=0, initial=False):
        force_incremental = self.args.force_incremental
        logging.info(f"In global_clustering_representation_based")
        if initial:
            # need to first update client distribution
            # for following rounds, this step is already applied in aggregator
            self.global_client_update_label_counts(round=curr_round)
            # avoid keeping a record for cluster 0
            self.data_drifted_clients = {}
        start_time = time.time()
        all_clients = self.feasibleClients[0]
        distribution_prob = {}
        for client_id in all_clients:
            client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
            distribution_prob[client_id] = [x/sum(client_label_counts) for x in client_label_counts]

        try:
            representaion_record = {}
            for client_id in all_clients:
                representaion_record[client_id] = self.client_metadata[self.getUniqueId(0, client_id)].representation
            logging.info(f"{len(all_clients)} clients representation finding time: {time.time() - start_time} second")
        except Exception as e:
            logging.info(f"error in loading representaion of client {client_id}: {e}")
            return
        representation_dimension = len(representaion_record[all_clients[0]])
        logging.info(f"representation_dimension: {representation_dimension}")
        
        # construct the clustered client feature dictionary
        clustered_client_features = {}
        for idx in range(len(all_clients)):
            clustered_client_features[all_clients[idx]] = representaion_record[all_clients[idx]]

        need_global_recluster = True
        
        if increment:
            # reuse the need_gradient_based_global_recluster function, 
            # except that we use representation instead of jl-transformed gradient
            need_global_recluster = self.need_gradient_based_global_recluster(\
                clustered_client_features, force_incremental=force_incremental)

        use_kmeans = True
        if need_global_recluster:
            A = np.array([representaion_record[c] for c in all_clients])
            max_cluster_size = len(A)
            retry = 0
            while max_cluster_size > len(A) * self.args.max_cluster_size_ratio and retry < 100:
                
                k_clusters = max(self.args.min_num_cluster, self.find_optimal_cluster_numbers(A, ksearch_type="kmeans"))
                # k_clusters = self.find_optimal_cluster_numbers(A, ksearch_type="kmeans")
                logging.info(f"optimal kmeans_clusters: {k_clusters}")
                initial_centers = kmeans_plusplus_initializer(A, k_clusters).initialize()
                # Create instance of K-Means algorithm with prepared centers.
                kmeans_instance = kmeans(A, initial_centers, metric=self.euclidean_square)
                # Run cluster analysis and obtain results.
                kmeans_instance.process()
                kclusters = kmeans_instance.get_clusters()
                max_cluster_size = max([len(k) for k in kclusters])
                logging.info(f"num_cluster: {len(kclusters)}, max_cluster_size: {max_cluster_size}")
                retry += 1
            
            # extract cluster assignments
            current_clusters = {}
            for i in range(len(kclusters)):
                current_clusters[i+1] = []
                for idx in kclusters[i]:
                    current_clusters[i+1].append(all_clients[idx])
            logging.info(f"k clusters: {[(k, len(v)) for k, v in current_clusters.items()]}")
            
            if delete_small_cluster:
                # delete small clusters
                current_clusters = self.cluster_manager.move_clients_by_gain(
                    clustered_client_features, current_clusters, ksearch_type="kmeans")
        else:
            current_clusters = {}
            for cluster_id in self.current_clusters:
                if len(self.feasibleClients[cluster_id]) > 0:
                    current_clusters[cluster_id] = self.feasibleClients[cluster_id]
        
        if len(self.feasibleClients) < len(current_clusters)+1:
            # expand self.feasibleClients if creating more clusters
            for _ in range(len(self.feasibleClients), len(current_clusters)+1):
                logging.info(f"expand self.feasibleClients by 1")
                self.feasibleClients.append([])
                # expand Oort sampler if needed
                if self.mode == "oort":
                    self.ucb_sampler.append(None)

        # update current clusters key to start from 1
        curr_cluster_idx = 1
        new_current_clusters = {}
        for v in current_clusters.values():
            new_current_clusters[curr_cluster_idx] = v
            self.feasibleClients[curr_cluster_idx] = v
            # reset Oort sampler if needed
            self.reset_sampler(curr_cluster_idx)
            curr_cluster_idx += 1
        current_clusters = new_current_clusters

        # calculate the center of each cluster
        per_cluster_to_center_distance_sum_and_avg = {}
        for cluster_id, cluster in current_clusters.items():
            cluster_center = np.mean([clustered_client_features[client_id] for client_id in cluster], axis=0)
            cluster_to_center_distance_sum = 0
            for client_id in cluster:
                cluster_to_center_distance_sum += sum((clustered_client_features[client_id][k] - cluster_center[k])**2 \
                                                        for k in range(representation_dimension))
            per_cluster_to_center_distance_sum_and_avg[cluster_id] = \
                (cluster_to_center_distance_sum, cluster_to_center_distance_sum / len(cluster))
        logging.info(f"per cluster pairwise distance sum and avg: {per_cluster_to_center_distance_sum_and_avg}")
        
        # calculate the cluster centers
        overall_distribution_prob_all = np.stack([distribution_prob[client_id] for client_id in all_clients])
        # self.cluster_to_center[0] = np.median(overall_distribution_prob_all, axis=0).tolist()
        self.cluster_to_center[0] = np.mean(overall_distribution_prob_all, axis=0).tolist()
        for cluster_id, v in current_clusters.items():
            overall_distribution_prob = np.stack([distribution_prob[client_id] for client_id in v])
            # self.cluster_to_center[cluster_id] = np.median(overall_distribution_prob, axis=0).tolist()
            self.cluster_to_center[cluster_id] = np.mean(overall_distribution_prob, axis=0).tolist()
        
        logging.info(f"cluster sizes: {[(k, len(v)) for k, v in current_clusters.items()]}")
        
        self.current_clusters = list(current_clusters.keys())
        logging.info(f"current_clusters: {self.current_clusters}")

        return list(current_clusters.keys()), distribution_prob, need_global_recluster

    def client_update_label_counts(self, client_id, new_label_counts):
        unique_id = self.getUniqueId(0, client_id)
        # logging.info(f"client {client_id} previous label counts: {self.client_metadata[unique_id].label_distribution}")
        if client_id in self.malicious_clients:
            prev_label_counts = self.malicious_clients_true_distribution[client_id]
        else:
            prev_label_counts = self.client_metadata[unique_id].label_distribution
        if new_label_counts != prev_label_counts:
            logging.info(f"client {client_id} previous label counts: {prev_label_counts}")
            if client_id in self.malicious_clients:
                self.malicious_clients_true_distribution[client_id] = copy.deepcopy(new_label_counts)
                self.rng.shuffle(new_label_counts)
            self.client_metadata[unique_id].register_distribution(new_label_counts)
            # if this client now doesn't have any samples, move it to not_yet_feasibleClients
            if sum(new_label_counts) == 0:
                self.feasibleClients[0].remove(client_id)
                logging.info(f"client {client_id} moved to not_yet_feasibleClients")
                for cluster in self.current_clusters:
                    if client_id in self.feasibleClients[cluster]:
                        self.feasibleClients[cluster].remove(client_id)
                        logging.info(f"client {client_id} removed from cluster {cluster}")
                self.not_yet_feasibleClients.append(client_id)
            else:
                # add this client into the data drifted clients
                cluster_id = 0
                for cluster in self.current_clusters:
                    if client_id in self.feasibleClients[cluster]:
                        cluster_id = cluster
                        break
                self.register_data_drifted_client(client_id, cluster_id)
                logging.info(f"client {client_id} of cluster {cluster_id} drifted, different idx and count: \
{[(i, (prev_label_counts[i], new_label_counts[i])) for i in range(len(prev_label_counts)) if prev_label_counts[i] != new_label_counts[i]]}")

    def verify_unfeasible_clients(self, round):
        has_empty_cluster = False
        # go through the clients in self.not_yet_feasibleClients, add any clients with samples into feasibleClients[0]
        for client_id in self.not_yet_feasibleClients:
            unique_id = self.getUniqueId(0, client_id)
            if client_id in self.malicious_clients:
                prev_label_counts = self.malicious_clients_true_distribution[client_id]
            else:
                prev_label_counts = self.client_metadata[unique_id].label_distribution
            client_finish_round = max(self.client_rank_to_distribution_at_round[client_id].keys())
            if round > client_finish_round:
                new_label_counts = self.client_rank_to_distribution_at_round[client_id][client_finish_round]
            else:
                for shift_round in sorted(self.client_rank_to_distribution_at_round[client_id].keys(), reverse=True):
                    if round >= shift_round:
                        new_label_counts = self.client_rank_to_distribution_at_round[client_id][shift_round]
                        break
            if new_label_counts != prev_label_counts:
                if client_id in self.malicious_clients:
                    self.malicious_clients_true_distribution[client_id] = copy.deepcopy(new_label_counts)
                    self.rng.shuffle(new_label_counts)
                    logging.info(f"malicious client {client_id} permuted own distribution")
                self.client_metadata[unique_id].register_distribution(new_label_counts)
                logging.info(f"unfeasible client {client_id} nown online, updated to {new_label_counts} in round {round}")
            if sum(new_label_counts) > 0:
                self.not_yet_feasibleClients.remove(client_id)
                self.feasibleClients[0].append(client_id)
                self.register_data_drifted_client(client_id, 0)
                logging.info(f"client {client_id} moved from not_yet_feasibleClients to feasibleClients0")
        # go through the clients in clusters, remove any clients without any samples from feasibleClients[0] and trigger global_recluster
        for cluster_id in self.current_clusters:
            client_to_remove = []
            for client_id in self.feasibleClients[cluster_id]:
                unique_id = self.getUniqueId(cluster_id, client_id)
                client_finish_round = max(self.client_rank_to_distribution_at_round[client_id].keys())
                if round > client_finish_round:
                    new_label_counts = self.client_rank_to_distribution_at_round[client_id][client_finish_round]
                else:
                    for shift_round in sorted(self.client_rank_to_distribution_at_round[client_id].keys(), reverse=True):
                        if round >= shift_round:
                            new_label_counts = self.client_rank_to_distribution_at_round[client_id][shift_round]
                            break
                if sum(new_label_counts) == 0:
                    client_to_remove.append(client_id)
                    logging.info(f"client {client_id} removed from cluster {cluster_id}")
                    self.not_yet_feasibleClients.append(client_id)
            for client_id in client_to_remove:
                self.feasibleClients[cluster_id].remove(client_id)
                self.feasibleClients[0].remove(client_id)
            if len(self.feasibleClients[cluster_id]) == 0:
                has_empty_cluster = True
        return has_empty_cluster

    def global_client_update_label_counts(self, round):
        total_clients = 0
        drift_clients = 0
        for client_id in self.feasibleClients[0] + self.not_yet_feasibleClients:
            cluster_id = 0
            for cluster in self.current_clusters:
                if client_id in self.feasibleClients[cluster]:
                    cluster_id = cluster
                    break
            total_clients += 1
            unique_id = self.getUniqueId(0, client_id)
            if client_id in self.malicious_clients:
                prev_label_counts = self.malicious_clients_true_distribution[client_id]
            else:
                prev_label_counts = self.client_metadata[unique_id].label_distribution
            client_finish_round = max(self.client_rank_to_distribution_at_round[client_id].keys())
            if round > client_finish_round:
                new_label_counts = self.client_rank_to_distribution_at_round[client_id][client_finish_round]
            else:
                for shift_round in sorted(self.client_rank_to_distribution_at_round[client_id].keys(), reverse=True):
                    if round >= shift_round:
                        new_label_counts = self.client_rank_to_distribution_at_round[client_id][shift_round]
                        break
            if new_label_counts != prev_label_counts:
                if client_id in self.malicious_clients:
                    self.malicious_clients_true_distribution[client_id] = copy.deepcopy(new_label_counts)
                    self.rng.shuffle(new_label_counts)
                    logging.info(f"malicious client {client_id} permuted own distribution")
                self.client_metadata[unique_id].register_distribution(new_label_counts)
                # check if any client in self.not_yet_feasibleClients now has samples
                if client_id in self.not_yet_feasibleClients:
                    if sum(new_label_counts) > 0:
                        self.not_yet_feasibleClients.remove(client_id)
                        self.feasibleClients[0].append(client_id)
                        self.register_data_drifted_client(client_id, 0)
                        logging.info(f"client {client_id} moved from not_yet_feasibleClients to feasibleClients0")
                        drift_clients += 1
                    else:
                        continue
                else:
                    # if this client now doesn't have any samples, move it to not_yet_feasibleClients
                    if sum(new_label_counts) == 0:
                        self.feasibleClients[0].remove(client_id)
                        logging.info(f"client {client_id} moved to not_yet_feasibleClients")
                        for cluster in self.current_clusters:
                            if client_id in self.feasibleClients[cluster]:
                                self.feasibleClients[cluster].remove(client_id)
                                logging.info(f"client {client_id} removed from cluster {cluster}")
                        self.not_yet_feasibleClients.append(client_id)
                    else:
                        # add this client into the data drifted clients
                        self.register_data_drifted_client(client_id, cluster_id)
                        logging.info(f"client {client_id} of cluster {cluster_id} drifted")
                        drift_clients += 1
        logging.info(f"round {round} total clients {total_clients}, drifted clients {drift_clients}")

    def clientRecluster(self, cluster_id=0, use_distribution=True, force_incremental=False):

        if cluster_id not in self.data_drifted_clients:
            return set()
        deviate_clients = self.data_drifted_clients[cluster_id]

        prev_cluster_to_center = copy.deepcopy(self.cluster_to_center)
        # recluster deviating clients
        recluster_record = []
        touched_clusters = set()
        
        if len(deviate_clients):
            logging.info(f"cluster {cluster_id}, {len(deviate_clients)} clients reclustering")
            
            touched_clusters = {cluster_id}

            if use_distribution:
                for i in deviate_clients:
                    if force_incremental and len(self.feasibleClients[cluster_id]) == 1:
                        if not (cluster_id in [t[1] for t in recluster_record]):
                            logging.info(f"force_incremental, cluster {cluster_id} has only one client, skip moving client {self.feasibleClients[cluster_id]}")
                            break
                    dist_to_cluster_centers = []
                    client_id = i
                    # remove reclustered clients from the current cluster (unless it is in the global cluster)
                    if cluster_id != 0:
                        self.feasibleClients[cluster_id].remove(client_id)
                    client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
                    client_distribution = [x / sum(client_label_counts) for x in client_label_counts] 
                    for cluster in self.current_clusters:
                        if self.args.use_l1_distance:
                            dist_to_cluster_centers.append((cluster, \
                                np.linalg.norm(np.array(client_distribution)-np.array(prev_cluster_to_center[cluster]), ord=1)))
                        else:
                            dist_to_cluster_centers.append((cluster, \
                                    jensenshannon(prev_cluster_to_center[cluster],client_distribution)))
                    # find the cluster with the closest center
                    dist_to_cluster_centers.sort(key = lambda x : x[1])
                    # distance to the center of the previous cluster this client belongs to
                    if self.args.use_l1_distance:
                        dist_to_prev_cluster = np.linalg.norm(np.array(client_distribution)-\
                                                              np.array(prev_cluster_to_center[cluster_id]), ord=1)
                    else:
                        dist_to_prev_cluster = jensenshannon(prev_cluster_to_center[cluster_id],client_distribution)
                    if cluster_id == 0:
                        # if client was in the global cluster, recluster it into the closest cluster
                        recluster_record.append((client_id, dist_to_cluster_centers[0][0]))
                    elif dist_to_cluster_centers[0][1] < dist_to_prev_cluster:
                        recluster_record.append((client_id, dist_to_cluster_centers[0][0]))
                    else:
                        # if the closest cluster is not close enough and it is not the global cluster, recluster it into the previous cluster
                        recluster_record.append((client_id, cluster_id))
            
            # add reclustered clients into new clusters accordingly
            for t in recluster_record:
                self.feasibleClients[t[1]].append(t[0])
                touched_clusters.add(t[1])
                logging.info(f"recluster client {t[0]} from {cluster_id} into {t[1]}")

        self.data_drifted_clients[cluster_id] = []
        return touched_clusters
    
    def hasDriftedClients(self):
        for k, v in self.data_drifted_clients.items():
            if len(v) > 0:
                logging.info(f"has drifted clients in cluster {k}")
                return True
        return False
    
    def getDriftedClients(self):
        drifted_clients = set()
        for k, v in self.data_drifted_clients.items():
            if len(v) > 0:
                drifted_clients.update(v)
        return drifted_clients
    
    def clientReclusterAllGradientBased(self, device, clusters=[0], use_global_model=False, curr_round=0, default_global_recluster=False):
        logging.info(f"reclustering clusters using gradient {clusters}")

        prev_cluster_to_center = copy.deepcopy(self.cluster_to_center)
        prev_max_cluster_id = max(prev_cluster_to_center.keys())
        model_mapping = {}

        _, distribution_prob, need_global_recluster = \
            self.global_clustering_gradient_based(device, delete_small_cluster=self.delete_small_cluster,
                                                  curr_round=curr_round, increment=(not default_global_recluster))
        # find the closest previous cluster for each new cluster
        for new_cluster in self.current_clusters:       
            if use_global_model:
                model_mapping[new_cluster] = 0
                logging.info(f"new cluster {new_cluster} with {len(self.feasibleClients[new_cluster])} clients use global model by default")
            elif not need_global_recluster:
                # if no global recluster, just continue with the previous model
                # if new_cluster > prev_max_cluster_id, then this is a new singleton cluster, should use global model
                if new_cluster > prev_max_cluster_id:
                    model_mapping[new_cluster] = 0
                else:
                    model_mapping[new_cluster] = new_cluster
            else:
                logging.info(f"new cluster {new_cluster} with {len(self.feasibleClients[new_cluster])} clients: {self.feasibleClients[new_cluster]}")
                dist_to_previous_center = []
                for prev_cluster, prev_center in prev_cluster_to_center.items():
                    distance_sum = 0
                    for client_id in self.feasibleClients[new_cluster]:
                        if client_id in distribution_prob:
                            if self.args.use_l1_distance:
                                distance_sum += np.linalg.norm(np.array(distribution_prob[client_id])-np.array(prev_center), ord=1)
                            else:
                                distance_sum += jensenshannon(distribution_prob[client_id],prev_center)
                        else:
                            logging.info(f"{client_id} not in distribution_prob")
                    dist_to_previous_center.append((prev_cluster, distance_sum))
                # record the closest previous cluster
                closest_record = sorted(dist_to_previous_center, key=lambda x:x[1])[0]
                model_mapping[new_cluster] = closest_record[0]
                logging.info(f"new cluster {new_cluster} should start with model of old cluster {model_mapping[new_cluster]}, cumulated distance {closest_record[1]}")
        # remove old clusters
        cluster_to_remove = []
        for cluster in self.cluster_to_center:
            if cluster != 0 and (cluster not in self.current_clusters):
                cluster_to_remove.append(cluster)
        for cluster in cluster_to_remove:
            logging.info(f"remove old cluster {cluster}")
            del self.cluster_to_center[cluster]
    
        # clear the data drifted clients if we do default global reclustering
        self.data_drifted_clients = {}
        return model_mapping
    
    def clientReclusterAllRepresentationBased(self, device, clusters=[0], use_global_model=False, curr_round=0, default_global_recluster=False):
        logging.info(f"reclustering clusters using representation {clusters}")

        prev_cluster_to_center = copy.deepcopy(self.cluster_to_center)
        prev_max_cluster_id = max(prev_cluster_to_center.keys())
        model_mapping = {}

        _, distribution_prob, need_global_recluster = \
            self.global_clustering_representation_based(\
                delete_small_cluster=self.delete_small_cluster,
                increment=(not default_global_recluster), curr_round=curr_round)
        # find the closest previous cluster for each new cluster
        for new_cluster in self.current_clusters:       
            if use_global_model:
                model_mapping[new_cluster] = 0
                logging.info(f"new cluster {new_cluster} with {len(self.feasibleClients[new_cluster])} clients use global model by default")
            elif not need_global_recluster:
                # if no global recluster, just continue with the previous model
                # if new_cluster > prev_max_cluster_id, then this is a new singleton cluster, should use global model
                if new_cluster > prev_max_cluster_id:
                    model_mapping[new_cluster] = 0
                else:
                    model_mapping[new_cluster] = new_cluster
            else:
                logging.info(f"new cluster {new_cluster} with {len(self.feasibleClients[new_cluster])} clients: {self.feasibleClients[new_cluster]}")
                dist_to_previous_center = []
                for prev_cluster, prev_center in prev_cluster_to_center.items():
                    distance_sum = 0
                    for client_id in self.feasibleClients[new_cluster]:
                        if client_id in distribution_prob:
                            distance_sum += np.linalg.norm(np.array(distribution_prob[client_id])-np.array(prev_center), ord=1)
                        else:
                            logging.info(f"{client_id} not in distribution_prob")
                    dist_to_previous_center.append((prev_cluster, distance_sum))
                # record the closest previous cluster
                closest_record = sorted(dist_to_previous_center, key=lambda x:x[1])[0]
                model_mapping[new_cluster] = closest_record[0]
                logging.info(f"new cluster {new_cluster} should start with model of old cluster {model_mapping[new_cluster]}, cumulated distance {closest_record[1]}")
        # remove old clusters
        cluster_to_remove = []
        for cluster in self.cluster_to_center:
            if cluster != 0 and (cluster not in self.current_clusters):
                cluster_to_remove.append(cluster)
        for cluster in cluster_to_remove:
            logging.info(f"remove old cluster {cluster}")
            del self.cluster_to_center[cluster]
    
        # clear the data drifted clients if we do default global reclustering
        self.data_drifted_clients = {}
        return model_mapping
    
    def clientReclusterAll(self, clusters=[0], use_distribution=True, default_global_recluster=False, round=0,
                           use_global_model=False, force_incremental=False):
        logging.info(f"reclustering clusters {clusters}")
        if self.args.recluster_at_drift:
            self.global_client_update_label_counts(round=round)
        else:
            # need to always handle newly available clients 
            # and trigger global reclustering if any cluster becomes empty
            default_global_recluster = self.verify_unfeasible_clients(round=round) \
                or default_global_recluster
        # if no clients drifted, skip
        if not self.hasDriftedClients():
            logging.info(f"no clients drifted, skip reclustering")
            return {}, False
        
        prev_cluster_to_center = copy.deepcopy(self.cluster_to_center)
        logging.info(f"prev_cluster_to_center: {prev_cluster_to_center}")
        shifted_cluster = set()
        empty_cluster = set()
        model_mapping = {}
        global_recluster = False
        distribution_prob = {}
        if not default_global_recluster:
            try:
                # update the cluster centers after client drifting
                for cluster in clusters:
                    # if this cluster becomes empty due to sliding window, skip
                    if len(self.feasibleClients[cluster]) == 0:
                        logging.info(f"cluster {cluster} became empty, skip")
                        empty_cluster.add(cluster)
                        continue
                    for client_id in self.feasibleClients[cluster]:
                        client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
                        distribution_prob[client_id] = [x/sum(client_label_counts) for x in client_label_counts]
                    overall_distribution_prob = np.stack([distribution_prob[client_id] for client_id in self.feasibleClients[cluster]])
                    self.cluster_to_center[cluster] = np.mean(overall_distribution_prob, axis=0).tolist()
                for cluster_id in clusters:
                    if len(self.feasibleClients[cluster_id]) > 0:
                        shifted_cluster.update(\
                            self.clientRecluster(cluster_id=cluster_id,
                                                use_distribution=use_distribution,
                                                force_incremental=force_incremental))
            except Exception as ex:
                logging.info(f"error in clientReclusterAll: {ex}")
            # recalculate the centers of clusters
            for cluster in shifted_cluster:
                if len(self.feasibleClients[cluster]) == 0:
                    logging.info(f"cluster {cluster} became empty, skip")
                    empty_cluster.add(cluster)
                    continue
                overall_distribution_prob = np.stack([distribution_prob[client_id] for client_id in self.feasibleClients[cluster]])
                self.cluster_to_center[cluster] = np.mean(overall_distribution_prob, axis=0).tolist()
        
        if force_incremental:
            # for logging purpose only
            self.cluster_manager.label_based_need_global_recluster(\
                prev_cluster_to_center, shifted_cluster, self.cluster_to_center, self.current_clusters) 
            global_recluster = False
        elif default_global_recluster or len(empty_cluster) > 0:
            global_recluster = True
        else:
            if self.args.use_pairwise_delta_threshold:
                global_recluster = self.cluster_manager.label_based_need_global_recluster_pairwise_threshold(\
                    self.feasibleClients, distribution_prob, self.current_clusters)
            else:
                global_recluster = self.cluster_manager.label_based_need_global_recluster(\
                    prev_cluster_to_center, shifted_cluster, self.cluster_to_center, self.current_clusters)

        if global_recluster:
            _, distribution_prob = self.global_clustering(curr_round=round)
            # find the closest previous cluster for each new cluster
            for new_cluster in self.current_clusters:
                if use_global_model:
                    model_mapping[new_cluster] = 0
                    logging.info(f"new cluster {new_cluster} with {len(self.feasibleClients[new_cluster])} clients use global model by default")
                else:
                    logging.info(f"new cluster {new_cluster} with {len(self.feasibleClients[new_cluster])} clients: {self.feasibleClients[new_cluster]}")
                    dist_to_previous_center = []
                    for prev_cluster, prev_center in prev_cluster_to_center.items():
                        distance_sum = 0
                        for client_id in self.feasibleClients[new_cluster]:
                            if client_id in distribution_prob:
                                if self.args.use_l1_distance:
                                    distance_sum += np.linalg.norm(np.array(distribution_prob[client_id])-np.array(prev_center), ord=1)
                                else:
                                    distance_sum += jensenshannon(distribution_prob[client_id],prev_center)
                            else:
                                logging.info(f"{client_id} not in distribution_prob")
                        dist_to_previous_center.append((prev_cluster, distance_sum))
                    # record the closest previous cluster
                    closest_record = sorted(dist_to_previous_center, key=lambda x:x[1])[0]
                    model_mapping[new_cluster] = closest_record[0]
                    logging.info(f"new cluster {new_cluster} should start with model of old cluster {model_mapping[new_cluster]}, cumulated distance {closest_record[1]}")
            # remove old clusters
            cluster_to_remove = []
            for cluster in self.cluster_to_center:
                if cluster != 0 and (cluster not in self.current_clusters):
                    cluster_to_remove.append(cluster)
            for cluster in cluster_to_remove:
                logging.info(f"remove old cluster {cluster}")
                del self.cluster_to_center[cluster]
        
        elif (not force_incremental):
            # delete clusters with high intra-cluster heterogeneity
            new_clusters = {}
            for cluster_id in self.current_clusters:
                if cluster_id != 0:
                    new_clusters[cluster_id] = self.feasibleClients[cluster_id]
                    logging.info(f"cluster {cluster_id} with {len(new_clusters[cluster_id])} clients")
            high_hetero_clusters, rebalanced_clusters = \
                self.cluster_manager.check_intra_heterogeneity_for_cluster_deletion(\
                new_clusters=new_clusters, distribution_prob=distribution_prob, 
                client_metadata=self.client_metadata, feasibleClients=self.feasibleClients)
            if len(high_hetero_clusters) > 0:
                logging.info(f"delete high intra-cluster heterogeneity clusters {high_hetero_clusters}")
                curr_cluster_idx = 1
                # reorder the clusters to start from 1 and update the feasibleClients, cluster_to_center, and model_mapping
                for cluster_id in sorted(self.current_clusters):
                    if cluster_id != 0 and (not cluster_id in high_hetero_clusters):
                        self.feasibleClients[curr_cluster_idx] = self.feasibleClients[cluster_id]
                        # update Oort sampler if needed
                        if self.mode == "oort":
                            self.ucb_sampler[curr_cluster_idx] = copy.deepcopy(self.ucb_sampler[cluster_id])
                        self.cluster_to_center[curr_cluster_idx] = self.cluster_to_center[cluster_id]
                        model_mapping[curr_cluster_idx] = cluster_id
                        curr_cluster_idx += 1
                # update the current_clusters
                self.current_clusters = list(range(1, curr_cluster_idx))
                # remove old clusters
                cluster_to_remove = []
                for cluster in self.cluster_to_center:
                    if cluster != 0 and (cluster not in self.current_clusters):
                        cluster_to_remove.append(cluster)
                for cluster in cluster_to_remove:
                    logging.info(f"remove old cluster {cluster}")
                    try:
                        del self.cluster_to_center[cluster]
                        del self.data_drifted_clients[cluster]
                        self.feasibleClients[cluster] = []
                        if self.mode == "oort":
                            self.ucb_sampler[cluster] = None
                    except Exception as ex:
                        logging.info(f"error in deleting cluster {cluster}: {ex}")
            else:
                if len(rebalanced_clusters) > 0:
                    # first, keep the existing clusters
                    for cluster_id in sorted(self.current_clusters):
                        if cluster_id != 0:
                            model_mapping[cluster_id] = cluster_id
                    for cluster_id, cluster_clients in rebalanced_clusters.items():
                        if cluster_id < len(self.feasibleClients):
                            self.feasibleClients[cluster_id] = cluster_clients
                            model_mapping[cluster_id] = cluster_id
                            logging.info(f"rebalanced cluster {cluster_id} with {len(cluster_clients)} clients to reduce intra-cluster heterogeneity")
                        else:
                            # keep expanding the feasibleClients
                            while cluster_id >= len(self.feasibleClients):
                                self.feasibleClients.append([])
                                # expand the Oort sampler if needed
                                self.reset_sampler(cluster_id)
                            self.feasibleClients[cluster_id] = cluster_clients
                            # start with the global model for new singleton clusters
                            model_mapping[cluster_id] = 0
                            logging.info(f"expanded and rebalanced cluster {cluster_id} with {len(cluster_clients)} clients to reduce intra-cluster heterogeneity")
                        # update the cluster center
                        client_label_counts = [self.client_metadata[self.getUniqueId(0, client_id)].label_distribution\
                                                for client_id in self.feasibleClients[cluster_id]]
                        distribution_prob = [[x/sum(l) for x in l] for l in client_label_counts]
                        overall_distribution_prob = np.stack(distribution_prob)
                        self.cluster_to_center[cluster_id] = np.median(overall_distribution_prob, axis=0).tolist()
                        # update current_clusters
                        if cluster_id not in self.current_clusters:
                            self.current_clusters.append(cluster_id)

        # clear the data drifted clients
        self.data_drifted_clients = {}

        return model_mapping, global_recluster

    def getFeasibleClients(self, cur_time, cluster_id=0, curr_round=0):
        logging.info(f"Cluster {cluster_id} {len(self.feasibleClients[cluster_id])} FeasibleClients now: {str(self.feasibleClients[cluster_id])}")
        if self.mode == "speed":
            # if there is only one client, return it
            if len(self.feasibleClients[cluster_id]) == 1:
                return self.feasibleClients[cluster_id]
            # make sure fast and slow clients are partitioned and up to date
            self.partition_fast_slow_client(cluster_id=cluster_id)

            if curr_round % 2 != 0:
                logging.info(f"Cluster {cluster_id} round {curr_round} fast client selection mode")
                clients_online = self.fast_clients[cluster_id]
            else:
                clients_online = self.feasibleClients[cluster_id]
           
            clients_online = list(set(clients_online).intersection(set(self.feasibleClients[cluster_id])))
        else:
            if self.user_trace is None:
                clients_online = self.feasibleClients[cluster_id]
            else:
                clients_online = [client_id for client_id in self.feasibleClients[cluster_id] if self.client_metadata[self.getUniqueId(
                    0, client_id)].is_active(cur_time)]

        return clients_online

    def isClientActive(self, client_id, cur_time):
        return self.client_metadata[self.getUniqueId(0, client_id)].is_active(cur_time)
    
    def find_cluster_distribution_center(self, cluster_id, use_mean=False):
        distribution_prob = {}
        for client_id in self.feasibleClients[cluster_id]:
            client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
            distribution_prob[client_id] = [x/sum(client_label_counts) for x in client_label_counts]
        overall_distribution_prob = np.stack([distribution_prob[client_id] for client_id in self.feasibleClients[cluster_id]])
        if use_mean:
            return np.mean(overall_distribution_prob, axis=0).tolist()
        return np.median(overall_distribution_prob, axis=0).tolist()

    def select_participants(self, num_of_clients: int, cur_time: float = 0, cluster_id=0, test=False,
                            curr_round=0, check_client_avail=False, get_global_gradient=False) -> List[int]:
        """Select participating clients for current execution task.

        Args:
            num_of_clients (int): number of participants to select.
            cur_time (float): current wall clock time.

        Returns:
            List[int]: indices of selected clients.

        """
        if test:
            clients_online = self.feasibleClients[cluster_id].copy()
            if check_client_avail and len(self.client_rank_to_avail_round) > 0:
                logging.info(f"check_client_avail in selecting subset of clients to test")
                clients_online = [c for c in clients_online \
                                  if curr_round in self.client_rank_to_avail_round[c]]
            # also filter out the malicious clients
            if len(self.malicious_clients) > 0:
                prev_clients_to_test = len(clients_online)
                clients_online = [c for c in clients_online if (not c in self.malicious_clients)]
                logging.info(f"cluster {cluster_id} testing filtered out {prev_clients_to_test-len(clients_online)} malicious clients")
            self.rng.shuffle(clients_online)
            if self.args.test_client_ratio == 1.0:
                return clients_online
            # use part of clients for testing
            num_client_to_test = max(min(len(clients_online), 100), int(len(clients_online)*self.args.test_client_ratio))
            return clients_online[:num_client_to_test]
        
        self.count += 1

        # use all clients when we need to get global properties
        clients_online = self.getFeasibleClients(cur_time, cluster_id=0 if get_global_gradient else cluster_id, \
                                                 curr_round=curr_round)
        if check_client_avail and len(self.client_rank_to_avail_round) > 0:
            clients_online = [c for c in clients_online \
                              if curr_round in self.client_rank_to_avail_round[c]]
        
        logging.info(f"Cluster {cluster_id} Round {curr_round}, Wall clock time: {round(cur_time)}, {len(clients_online)} clients online, " +
                     f"{len(self.feasibleClients[cluster_id]) - len(clients_online)} clients offline")

        if len(clients_online) <= num_of_clients or get_global_gradient:
            return clients_online

        pickled_clients = None
        clients_online_set = set(clients_online)
        client_len = num_of_clients

        if self.mode == "oort" and self.ucb_sampler[cluster_id].getAllMetricsLength() > 0:
            logging.info(f"Cluster {cluster_id} Oort selection mode")
            try:
                pickled_clients = self.ucb_sampler[cluster_id].select_participant(
                    client_len, feasible_clients=clients_online_set)
            except Exception as ex:
                logging.info(f"error in Oort selection: {ex}")
                self.rng.shuffle(clients_online)
                pickled_clients = clients_online[:client_len]

        elif (self.mode == "train_loss" or self.mode == "train_loss_reward") and len(self.client_rank_to_utility) > 0:
            # do weighted sampling based on training loss, higher priority to clients with larger loss
            # NOTE: for now, assign clients without a utility the minimum non-zero recorded utility value to ensure they might be picked
            min_recorded_utility = 0.1
            max_recorded_utility = max(self.client_rank_to_utility.values())
            for u in sorted(self.client_rank_to_utility.values()):
                if u > 0:
                    min_recorded_utility = u
                    break
            for k, v in self.client_rank_to_utility.items():
                if v <= 0.0:
                    self.client_rank_to_utility[k] = min_recorded_utility
            logging.info(f"max_recorded_utility: {max_recorded_utility}")
            clients_weight = [min(self.client_rank_to_utility.get(c, max_recorded_utility), max_recorded_utility)\
                               for c in clients_online]

            # normalize weights so they sum to 1
            clients_weight = [w / sum(clients_weight) for w in clients_weight]
            pickled_clients = self.numpy_rng.choice(clients_online, client_len, replace=False, p=clients_weight)
            # pickled_clients = np.random.choice(clients_online, client_len, replace=False, p=clients_weight)
            pickled_clients = pickled_clients.tolist()

        elif self.mode == "distribution_distance" or self.mode == "distribution_uniform":
            cluster_distribution_center = self.find_cluster_distribution_center(cluster_id, use_mean=True)
            clients_weight = []
            max_distance = 2.0
            for client_id in clients_online:
                client_label_counts = self.client_metadata[self.getUniqueId(0, client_id)].label_distribution
                client_distribution = [x/sum(client_label_counts) for x in client_label_counts]
                if self.args.use_l1_distance:
                    client_weight = np.linalg.norm(np.array(client_distribution)-np.array(cluster_distribution_center), ord=1)
                else:
                    client_weight = jensenshannon(cluster_distribution_center,client_distribution)
                if client_weight > max_distance:
                    logging.info(f"WARNING: client {client_id} with distance {client_weight} to cluster center")
                clients_weight.append(client_weight)
            # do weighted sampling based on the distribution difference between client and the cluster center
            # higher priority to clients with smaller distance
            if self.mode == "distribution_distance":
                median_weight = np.median(clients_weight)
                if median_weight == 0:
                    # all clients are equally close to the cluster center, just return a random subset
                    logging.info(f"distribution_distance selection, all clients are equally close to the cluster center")
                    self.rng.shuffle(clients_online)
                    pickled_clients = clients_online[:client_len]
                else:
                    # replace weights of the farthest 50% clients with the median weight
                    clients_weight = [1/w if w < median_weight else 1/median_weight for w in clients_weight]
                    logging.info(f"distribution_distance selection median_weight: {median_weight}")
                    
                    # normalize weights so they sum to 1
                    clients_weight = [w / sum(clients_weight) for w in clients_weight]
                    pickled_clients = self.numpy_rng.choice(clients_online, client_len, replace=False, p=clients_weight)
                    pickled_clients = pickled_clients.tolist()

            # only select the top 50% clients who are closest to the cluster center
            else:
                clients_idx_sorted_by_distance = sorted(enumerate(clients_weight), key=lambda x:x[1])[:len(clients_online)//2]
                logging.info(f"distribution_uniform selection, select among {len(clients_idx_sorted_by_distance)} clients")
                pickled_clients = [clients_online[i] for i, _ in clients_idx_sorted_by_distance]
                self.rng.shuffle(pickled_clients)
                pickled_clients = pickled_clients[:client_len]
        else:
            self.rng.shuffle(clients_online)
            pickled_clients = clients_online[:client_len]

        return pickled_clients

    def resampleClients(self, num_of_clients, cur_time=0):
        return self.select_participants(num_of_clients, cur_time)

    def getAllMetrics(self, cluster_id=0):
        if self.mode == "oort":
            return self.ucb_sampler[cluster_id].getAllMetrics()
        return {}

    def getDataInfo(self):
        return {'total_feasible_clients': len(self.feasibleClients[0]), 'total_num_samples': self.feasible_samples}

    def getClientReward(self, client_id, cluster_id=0):
        return self.ucb_sampler[cluster_id].get_client_reward(client_id)

    def get_median_reward(self, cluster_id=0):
        if self.mode == 'oort':
            return self.ucb_sampler[cluster_id].get_median_reward()
        return 0.
