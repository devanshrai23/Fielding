import logging
import numpy as np
from scipy.stats import wasserstein_distance

from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmedians import kmedians
import copy
import torch

class ClusterManager:
    def __init__(self, args):
        self.args = args
        logging.info("created empty cluster_to_center")

    def getUniqueId(self, host_id, client_id):
        return str(client_id)

    def move_clients_by_gain(self, client_features, current_clusters, ksearch_type="kmedians"):
        # calculate pairwise distance between clients
        num_clients = len(client_features)
        sorted_clients = sorted(client_features.keys())
        pairwise_distance = {}
        for i in sorted_clients:
            pairwise_distance[i] = {}
            pairwise_distance[i][i] = 0
        observed_values = len(client_features[min(client_features.keys())])
        if ksearch_type == 'kmedians':
            observed_values = np.arange(observed_values)
        logging.info(f"observed_values: {observed_values}")
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                client_i = sorted_clients[i]
                client_j = sorted_clients[j]
                try:
                    pairwise_distance[client_i][client_j] = \
                            np.linalg.norm(np.array(client_features[client_i]) - np.array(client_features[client_j]), ord=1)
                    pairwise_distance[client_j][client_i] = pairwise_distance[client_i][client_j]
                except Exception as e:
                    logging.info(f"error in calculating distance between {client_i} and {client_j}, {e}")
        
        initial_pairwise_distance_sum = 0
        for cluster in current_clusters.values():
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    initial_pairwise_distance_sum += pairwise_distance[cluster[i]][cluster[j]]

        current_gain_record = [None, 1] # [(client_id, prev_cluster_id, move_to_cluster_id), gain]

        moved_clients = 0
        while current_gain_record[1] > 0 and moved_clients < 100:
            # reset the current gain record
            current_gain_record = [None, 0]
            for c in client_features.keys():
                remove_gain = 0
                add_gains = {}
                cluster_of_c = None
                for cluster in current_clusters:
                    # find the change in pairwise distance if c is moved to cluster
                    if c in current_clusters[cluster]:
                        cluster_of_c = cluster
                        for client in current_clusters[cluster]:
                            remove_gain -= pairwise_distance[c][client]
                    else:
                        new_gain = 0
                        for client in current_clusters[cluster]:
                            new_gain += pairwise_distance[c][client]
                        add_gains[cluster] = new_gain
                # calculate the gain if c is moved to each cluster
                for cluster in add_gains.keys():
                    add_gains[cluster] = -(remove_gain + add_gains[cluster])
                # find the maximum gain
                max_gain_record = max(add_gains.items(), key=lambda x : x[1])
                # update the current gain record if necesary
                if max_gain_record[1] > current_gain_record[1]:
                    current_gain_record = [(c, cluster_of_c, max_gain_record[0]), max_gain_record[1]]
            # move the client with the maximum gain
            if current_gain_record[1] > 0:
                client_id, prev_cluster_id, move_to_cluster_id = current_gain_record[0]
                # keep at least 2 clusters
                if len(current_clusters) == 2 and len(current_clusters[prev_cluster_id]) == 1:
                    logging.info(f"stop moving client {client_id} from cluster {prev_cluster_id} to cluster {move_to_cluster_id} to preserve 2 clusters")
                    break
                current_clusters[prev_cluster_id].remove(client_id)
                if len(current_clusters[prev_cluster_id]) == 0:
                    del current_clusters[prev_cluster_id]
                current_clusters[move_to_cluster_id].append(client_id)
                logging.info(f"move client {client_id} from cluster {prev_cluster_id} to cluster {move_to_cluster_id}")
                moved_clients += 1

        after_pairwise_distance_sum = 0
        for _, cluster in current_clusters.items():
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    after_pairwise_distance_sum += pairwise_distance[cluster[i]][cluster[j]]

        logging.info(f"pairwise distance sum before: {initial_pairwise_distance_sum}, after: {after_pairwise_distance_sum}")
        
        return current_clusters

    def reduce_intra_heterogeneity(self, new_cluster, target_global_avg_distance, distribution_prob, \
                                    observed_values, pairwise_distance=None):
        """
        Given a new cluster, keeps removing the client with the largest distance to the
        cluster center until the intra heterogeneity is reduced to the target level
        or the cluster has only one client left"""

        cluster_avg_distance = 0
        max_distance = 0
        max_distance_client = None

        client_id_to_distance = {}
        # for each cluster, calculate the avg pairwise distance between each pair of its clients
        sorted_cluster_clients = sorted(new_cluster)
        for i in range(len(sorted_cluster_clients) - 1):
            for j in range(i+1, len(sorted_cluster_clients)):
                client_i = sorted_cluster_clients[i]
                client_j = sorted_cluster_clients[j]
                cluster_avg_distance += pairwise_distance[(client_i, client_j)]
                client_id_to_distance[client_i] = client_id_to_distance.get(client_i, 0) + pairwise_distance[(client_i, client_j)]
                client_id_to_distance[client_j] = client_id_to_distance.get(client_j, 0) + pairwise_distance[(client_i, client_j)]
        if len(new_cluster) > 1:
            cluster_avg_distance /= (len(new_cluster) * (len(new_cluster) - 1) // 2)
            max_distance_client = max(client_id_to_distance, key=client_id_to_distance.get)
            max_distance = client_id_to_distance[max_distance_client]

        removed_clients = []
        while cluster_avg_distance > target_global_avg_distance and len(new_cluster) > 1:
            # remove the client with the largest distance to the cluster center
            new_cluster.remove(max_distance_client)
            removed_clients.append(max_distance_client)
            logging.info(f"remove client {max_distance_client} from cluster, avg distance was {cluster_avg_distance}, target is {target_global_avg_distance}")

            # remove distance of the removed client
            client_id_to_distance.pop(max_distance_client)
            # recalculate the cluster_avg_distace
            # we removed a client, so the number of pairs previously was (len(new_cluster) + 1 choose 2)
            cluster_avg_distance = \
                (cluster_avg_distance * ((len(new_cluster)+1) * len(new_cluster) // 2) - max_distance) / (len(new_cluster) * (len(new_cluster) - 1) // 2)
            # recalculate the sum of pairwise distance for the remaining clients
            for client_id in new_cluster:
                if client_id < max_distance_client:
                    client_id_to_distance[client_id] -= pairwise_distance[(client_id, max_distance_client)]
                else:
                    client_id_to_distance[client_id] -= pairwise_distance[(max_distance_client, client_id)]
            max_distance_client = max(client_id_to_distance, key=client_id_to_distance.get)
            max_distance = client_id_to_distance[max_distance_client]

        return new_cluster, removed_clients
    
    def check_intra_heterogeneity_for_cluster_deletion(self, new_clusters, distribution_prob, client_metadata,
                                                       feasibleClients):
        # get the distribution center of the global model and each new cluster
        cluster_centers = {}
        try:
            # add missing distribution_prob for clients in new_clusters if needed
            for client_id in feasibleClients[0]:
                if client_id not in distribution_prob:
                    client_label_counts = client_metadata[self.getUniqueId(0, client_id)].label_distribution
                    distribution_prob[client_id] = [x/sum(client_label_counts) for x in client_label_counts]
                    logging.info(f"NOTE: client {client_id} added distribution: {distribution_prob[client_id]} in check_intra_heterogeneity_for_cluster_deletion")
            # calculate cluster centers
            for cluster_id, cluster in new_clusters.items():
                logging.info(f"cluster {cluster_id} has {len(cluster)} clients")
                cluster_centers[cluster_id] = np.median(np.stack([distribution_prob[client_id] for client_id in cluster]), axis=0).tolist()
            global_center = np.median(np.stack([distribution_prob[client_id] for client_id in feasibleClients[0]]), axis=0).tolist()
            observed_values = np.arange(len(global_center))
            
            cluster_to_remove = []
            rebalanced_clusters = {}
            pairwise_distance = {}
            new_cluster_id = max(new_clusters.keys()) + 1
            min_cluster_avg_distance = None
            cluster_id_to_avg_distance = {}

            # calculate pairwise distance between clients
            sorted_clients = sorted(distribution_prob.keys())
            for i in range(len(sorted_clients) - 1):
                for j in range(i+1, len(sorted_clients)):
                    client_i = sorted_clients[i]
                    client_j = sorted_clients[j]
                    pairwise_distance[(client_i, client_j)] = \
                            np.linalg.norm(np.array(distribution_prob[client_i]) - np.array(distribution_prob[client_j]), ord=1)
            global_avg_distance = sum(pairwise_distance.values()) / len(pairwise_distance)
            # for each cluster, calculate the avg wdistance between each pair of its clients
            for cluster_id, cluster in new_clusters.items():
                cluster_avg_distance = 0
                sorted_cluster_clients = sorted(cluster)
                for i in range(len(sorted_cluster_clients) - 1):
                    for j in range(i+1, len(sorted_cluster_clients)):
                        cluster_avg_distance += pairwise_distance[(sorted_cluster_clients[i], sorted_cluster_clients[j])]
                if len(cluster) > 1:
                    cluster_avg_distance /= (len(cluster) * (len(cluster) - 1) / 2)
                if (min_cluster_avg_distance is None) or cluster_avg_distance < min_cluster_avg_distance:
                    min_cluster_avg_distance = cluster_avg_distance
                cluster_id_to_avg_distance[cluster_id] = cluster_avg_distance
            
            # intra_cluster_distance_thres = max(min_cluster_avg_distance, global_avg_distance * 0.8) if rebalance else global_avg_distance
            intra_cluster_distance_thres = global_avg_distance
            for cluster_id, cluster_avg_distance in cluster_id_to_avg_distance.items():
                if cluster_avg_distance > intra_cluster_distance_thres:
                    cluster = new_clusters[cluster_id]
                    rebalanced_clusters[cluster_id], removed_clients = \
                        self.reduce_intra_heterogeneity(cluster, intra_cluster_distance_thres, distribution_prob, observed_values,\
                                                        pairwise_distance=pairwise_distance)

        except Exception as e:
            logging.info(f"error in check_intra_heterogeneity_for_cluster_deletion: {e}")
            
        return cluster_to_remove, rebalanced_clusters
    
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
    
    def label_based_need_global_recluster(self, prev_cluster_to_center, shifted_cluster,
                                          cluster_to_center, current_clusters):
        cluster_center_shift_distance = []
        for cluster in shifted_cluster:
            cluster_center_shift_distance.append(\
                    np.linalg.norm(np.array(prev_cluster_to_center[cluster]) - np.array(cluster_to_center[cluster]), ord=1))
            
            logging.info(f"recalculate cluster {cluster} center, shifted {cluster_center_shift_distance[-1]}")
        
        if len(cluster_center_shift_distance) > 0:
            if len(current_clusters) > 1:
                # recalculate the average w_distances between cluster centers
                w_distances = []
                for i in range(1, len(current_clusters)):
                    for j in range(i + 1, len(current_clusters)+1):
                        w_distances.append(\
                                np.linalg.norm(np.array(cluster_to_center[i]) - np.array(cluster_to_center[j]), ord=1))
                self.avg_center_wasserstein_distance = sum(w_distances) / len(w_distances)
                logging.info(f"avg_center_wasserstein_distance becomes {self.avg_center_wasserstein_distance}")
            else:
                self.avg_center_wasserstein_distance = 0
                logging.info(f"avg_center_wasserstein_distance becomes 0 as only one cluster left")
            # check if any cluster's center moved a lot, if so, do global reclustering
            
            for dist in cluster_center_shift_distance:
                if dist >= self.avg_center_wasserstein_distance / (self.args.global_recluster_thres_ratio):
                    # trigger global reclustering
                    return True
        return False

    def global_clustering_helper(self, all_clients_to_cluster, client_metadata, distribution_prob,
            feasibleClients, cluster_to_center, use_kmeans=True):
        
        current_clusters = {}
            
        A = np.array([distribution_prob[c] for c in all_clients_to_cluster])
        if use_kmeans:
            num_cluster = 0
            start_num_cluster = max(self.args.min_num_cluster, self.find_optimal_cluster_numbers(A, ksearch_type="kmeans"))
            max_cluster_size = len(A)
            while num_cluster < start_num_cluster or max_cluster_size > len(A) * self.args.max_cluster_size_ratio:
                # reuse existing cluster centers if we have enough
                if len(current_clusters) == start_num_cluster:
                    initial_centers = np.array([cluster_to_center[cid] for cid in \
                                                current_clusters])
                else:
                    # Prepare initial centers using K-Means++ method.
                    initial_centers = kmeans_plusplus_initializer(A, start_num_cluster).initialize()
                # Create instance of K-Means algorithm with prepared centers.
                kmeans_instance = kmeans(A, initial_centers)
                # Run cluster analysis and obtain results.
                kmeans_instance.process()
                kclusters = kmeans_instance.get_clusters()
                num_cluster = len(kclusters)
                max_cluster_size = max([len(v) for v in kclusters])
                logging.info(f"start_num_cluster: {start_num_cluster}, num_cluster: {num_cluster}, max_cluster_size: {max_cluster_size}")
        else:
            # use kmedians
            # start with 10 k-median clusters
            num_cluster = 0
            start_num_cluster = max(self.args.min_num_cluster, self.find_optimal_cluster_numbers(A, ksearch_type="kmedians"))
            max_cluster_size = len(A)
            while num_cluster < start_num_cluster or max_cluster_size > len(A) * self.args.max_cluster_size_ratio:
                initial_medians = A[np.random.permutation(A.shape[0])[:start_num_cluster],:]
                kmedians_instance = kmedians(A, initial_medians)
                kmedians_instance.process()
                kclusters = kmedians_instance.get_clusters()
                num_cluster = len(kclusters)
                max_cluster_size = max([len(v) for v in kclusters])
                logging.info(f"start_num_cluster: {start_num_cluster}, num_cluster: {num_cluster}, max_cluster_size: {max_cluster_size}")
            
        # if any cluster's size is > 1/2 of the total clients, redo clustering for the clients not in the largest cluster
        largest_cluster_size = 0
        largest_cluster_idx = None
        for i in range(len(kclusters)):
            if len(kclusters[i]) > largest_cluster_size:
                largest_cluster_size = len(kclusters[i])
                largest_cluster_idx = i
        # if there is one large cluster and less than 4 clusters for the remaining clients, redo clustering
        if largest_cluster_size > len(all_clients_to_cluster) // 2 and len(kclusters) < 5:
            indices_of_all_clients_to_recluster = [idx for idx in range(len(all_clients_to_cluster)) if idx not in kclusters[largest_cluster_idx]]
            all_clients_to_recluster = [all_clients_to_cluster[idx] for idx in indices_of_all_clients_to_recluster]
            A = np.array([distribution_prob[c] for c in all_clients_to_recluster])
            logging.info(f"largest cluster size {largest_cluster_size} is too large, redo clustering for the rest {len(all_clients_to_recluster)} clients")
            if use_kmeans:
                optimal_num_cluster = self.find_optimal_cluster_numbers(A, ksearch_type="kmeans")
                start_num_cluster = min(4, optimal_num_cluster)
                if start_num_cluster == 0:
                    logging.info(f"start_num_cluster is 0, skip re-clustering")
                else:
                    num_cluster = 0
                    while num_cluster < start_num_cluster:
                        logging.info(f"redo kmeans clustering for the rest clients, target cluster number: {start_num_cluster}, optimal cluster number: {optimal_num_cluster}")
                        initial_centers = kmeans_plusplus_initializer(A, start_num_cluster).initialize()
                        kmeans_instance = kmeans(A, initial_centers)
                        kmeans_instance.process()
                        kclusters_redo = kmeans_instance.get_clusters()
                        num_cluster = len(kclusters_redo)
            else:
                optimal_num_cluster = self.find_optimal_cluster_numbers(A, ksearch_type="kmedians")
                start_num_cluster = min(4, optimal_num_cluster)
                if start_num_cluster == 0:
                    logging.info(f"start_num_cluster is 0, skip re-clustering")
                else:
                    num_cluster = 0
                    while num_cluster < start_num_cluster:
                        logging.info(f"redo kmedians clustering for the rest clients, target cluster number: {start_num_cluster}, optimal cluster number: {optimal_num_cluster}")
                        initial_medians = A[np.random.permutation(A.shape[0])[:start_num_cluster],:]
                        kmedians_instance = kmedians(A, initial_medians)
                        kmedians_instance.process()
                        kclusters_redo = kmedians_instance.get_clusters()
                        num_cluster = len(kclusters_redo)
            # recombine everything into kclusters
            new_kclusters = [kclusters[largest_cluster_idx]]
            for i in range(len(kclusters_redo)):
                new_kclusters.append([indices_of_all_clients_to_recluster[idx] for idx in kclusters_redo[i]])
            kclusters = new_kclusters

        for i in range(len(kclusters)):
            current_clusters[i] = []
            for idx in kclusters[i]:
                current_clusters[i].append(all_clients_to_cluster[idx])
        logging.info(f"initial clusters: {[(k, len(v)) for k, v in current_clusters.items()]}")

        current_clusters = \
            self.move_clients_by_gain(\
                distribution_prob, current_clusters, ksearch_type="kmeans" if use_kmeans else "kmedians")

        # remove clusters with high intra-heterogeneity
        high_intra_heterogeneity_clusters, rebalanced_clusters = \
            self.check_intra_heterogeneity_for_cluster_deletion(\
                current_clusters, distribution_prob, client_metadata, 
                feasibleClients)
        for cluster_id in high_intra_heterogeneity_clusters:
            logging.info(f"remove cluster {cluster_id} due to high intra-heterogeneity")
            del current_clusters[cluster_id]
        for cluster_id, cluster in rebalanced_clusters.items():
            current_clusters[cluster_id] = cluster
            logging.info(f"rebalanced cluster {cluster_id} with {len(cluster)} clients to reduce high intra-heterogeneity")
    
        return current_clusters
                                    