"""Auxo reward-table clustering adapted to FIELDING's streaming runtime."""
import copy
import logging
import numpy as np
from sklearn.preprocessing import normalize

from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.auxo.vendor.clustering import QTable


class AuxoClientManager(ClientManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxo_updates = {}
        self.auxo_table = None

    def global_clustering(self, curr_round=0, initial=False, **kwargs):
        if not initial:
            raise RuntimeError('Auxo does not use FIELDING global reclustering')
        self.global_client_update_label_counts(curr_round)
        ids = sorted(int(m.client_id) for m in self.client_metadata.values())
        self.auxo_ids = ids
        self.auxo_index = {c: i for i, c in enumerate(ids, start=1)}
        ratio = self.args.num_participants / max(1, len(ids))
        self.auxo_table = QTable(len(ids)+1, known_clt=max(1, len(ids)//20),
                                train_ratio=ratio, avg_train_times=self.args.rounds*ratio,
                                split_round=None, elbow_constant=.97,
                                sample_seed=self.args.sample_seed, metric='cosine')
        while len(self.feasibleClients) < 2:
            self.feasibleClients.append([])
        self.feasibleClients[1] = list(self.feasibleClients[0])
        self.current_clusters = [1]
        self.reset_sampler(1)
        distributions = self._prepare_active_distributions()
        self._centers(distributions)
        self.data_drifted_clients = {}
        return [1], distributions

    def _centers(self, distributions):
        for k in [0] + self.current_clusters:
            members = self.feasibleClients[k]
            if members:
                self.cluster_to_center[k] = np.mean([distributions[c] for c in members], axis=0).tolist()

    def record_update(self, client, cluster, new_weights, old_weights):
        if cluster == 0 or self.auxo_table is None:
            return
        def array(x):
            return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        new = list(new_weights.values()) if isinstance(new_weights, dict) else new_weights
        old = list(old_weights.values()) if isinstance(old_weights, dict) else old_weights
        if len(new) != len(old):
            raise ValueError('Incompatible Auxo model updates')
        delta = np.concatenate([(array(a)-array(b)).ravel() for a,b in zip(old,new)])
        # Same tail-coordinate compression used by upstream Auxo metadata.
        delta = delta[-max(1, len(delta)//10):].astype(np.float32)
        if not np.isfinite(delta).all():
            raise ValueError('Non-finite Auxo client update')
        self.auxo_updates.setdefault(cluster, {})[client] = delta

    def clientReclusterAll(self, round=0, **kwargs):
        self.global_client_update_label_counts(round)
        distributions = self._prepare_active_distributions()
        active = set(distributions)
        before = {k: set(self.feasibleClients[k]) for k in self.current_clusters}
        table = self.auxo_table
        mapping = {k: k for k in self.current_clusters}
        changed = False
        for cluster, updates in list(self.auxo_updates.items()):
            clients = sorted(c for c in updates if c in active and c in before.get(cluster, set()))
            if len(clients) < 5:
                continue
            # Observe every eligible batch; gate only changes to cohort membership.
            candidate = copy.deepcopy(table)
            gradients = np.stack([updates[c] for c in clients])
            centered = gradients - gradients.mean(axis=0)
            indices = [self.auxo_index[c] for c in clients]
            candidate.knn_update_subR(cluster-1, normalize(centered), indices, round <= 500)
            split = candidate.update_mainR(cluster-1, centered, indices, round <= 500)
            pending = self.data_drifted_clients
            self.data_drifted_clients = {cluster: clients}
            # Auxo gating is cohort-update based, independent of label-drift triggers.
            allowed = table.num_model == 1 or self._check_model_impact()
            self.data_drifted_clients = pending
            if not allowed:
                logging.info('AUXO_GATE round=%s cohort=%s decision=defer', round, cluster)
                continue
            # Reject empty proposed cohorts: the runtime cannot train an empty model.
            labels = {c: int(candidate.y_kmeans[self.auxo_index[c]])+1 for c in active}
            if set(labels.values()) != set(range(1, candidate.num_model+1)):
                logging.info('AUXO_DEFER_EMPTY_COHORT round=%s', round)
                continue
            if split:
                mapping[candidate.num_model] = cluster
            table = candidate
            changed = changed or split or any(c not in before.get(k, set()) for c,k in labels.items())
            self.auxo_updates[cluster] = {}
        self.auxo_table = table
        while len(self.feasibleClients) <= table.num_model:
            self.feasibleClients.append([])
            self.reset_sampler(len(self.feasibleClients)-1)
        self.current_clusters = list(range(1, table.num_model+1))
        for k in self.current_clusters:
            self.feasibleClients[k] = [c for c in sorted(active)
                                      if int(table.y_kmeans[self.auxo_index[c]])+1 == k]
        self._centers(distributions)
        self.data_drifted_clients = {}
        logging.info('AUXO_ADAPT round=%s cohorts=%s changed=%s', round, table.num_model, changed)
        return (mapping if changed else {}), False
