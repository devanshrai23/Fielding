import ast
import copy
import logging
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

source = ast.parse(Path('fedscale/cloud/client_manager.py').read_text())
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ClientManager')
names = {'_prepare_active_distributions', 'getDriftedClients', 'register_data_drifted_client',
         'clientReclusterAll', 'hasDriftedClients'}
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = dict(np=np, math=math, logging=logging, copy=copy)
exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actual client manager>', 'exec'), ns)

class ActiveClientTests(unittest.TestCase):
    def manager(self):
        m = ns['ClientManager']()
        m.getUniqueId = lambda host, client: client
        m.feasibleClients = [[1, 2, 296], [1, 296], [2]]
        m.current_clusters = [1, 2]
        m.not_yet_feasibleClients = []
        m.data_drifted_clients = {1: [1, 296], 2: [2]}
        m.client_metadata = {c: SimpleNamespace(label_distribution=v) for c,v in
                             {1:[2,2], 2:[1,3], 296:[0,0]}.items()}
        return m

    def test_zero_client_removed_everywhere_and_pending_survives(self):
        m = self.manager()
        distributions = m._prepare_active_distributions()
        self.assertEqual(distributions, {1:[.5,.5], 2:[.25,.75]})
        self.assertEqual(m.feasibleClients, [[1,2],[1],[2]])
        self.assertEqual(m.not_yet_feasibleClients, [296])
        self.assertEqual(m.getDriftedClients(), {1,2})
        self.assertEqual(m._prepare_active_distributions(), distributions)
        self.assertEqual(m.not_yet_feasibleClients, [296])

    def test_missing_global_membership_is_reconciled(self):
        m = self.manager()
        m.feasibleClients[0] = [1]
        self.assertEqual(set(m._prepare_active_distributions()), {1,2})
        self.assertEqual(m.feasibleClients[0], [1,2])

    def test_invalid_counts_fail_before_mutation(self):
        for counts in ([float('nan'),1], [-1,2], [1,2,3], []):
            m = self.manager()
            m.client_metadata[2].label_distribution = counts
            before = copy.deepcopy(m.feasibleClients)
            with self.assertRaises(ValueError):
                m._prepare_active_distributions()
            self.assertEqual(m.feasibleClients, before)

    def test_all_inactive_clears_pending(self):
        m = self.manager()
        for metadata in m.client_metadata.values():
            metadata.label_distribution = [0,0]
        self.assertEqual(m._prepare_active_distributions(), {})
        self.assertEqual(m.feasibleClients, [[],[],[]])
        self.assertEqual(m.data_drifted_clients, {})

    def test_migration_failure_does_not_reach_heterogeneity(self):
        m = self.manager()
        m.args = SimpleNamespace(recluster_at_drift=True)
        m.global_client_update_label_counts = lambda **kw: None
        m._check_model_impact = lambda: True
        m.cluster_to_center = {0:[.5,.5],1:[.5,.5],2:[.5,.5]}
        def fail(**kwargs):
            raise RuntimeError('migration failed')
        m.clientRecluster = fail
        # No cluster_manager: reaching heterogeneity would raise AttributeError.
        with self.assertRaisesRegex(RuntimeError, 'migration failed'):
            m.clientReclusterAll(clusters=[0,1,2])

class HeterogeneityTests(unittest.TestCase):
    def check(self, counts, clusters):
        tree = ast.parse(Path('fedscale/cloud/client_clustering.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClusterManager')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and
                      n.name == 'check_intra_heterogeneity_for_cluster_deletion')
        scope = dict(np=np, logging=logging)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual heterogeneity>', 'exec'), scope)
        manager = SimpleNamespace(getUniqueId=lambda host, client: client)
        metadata = {c:SimpleNamespace(label_distribution=v) for c,v in counts.items()}
        return scope[method.name](manager, clusters, {}, metadata, [list(counts)])

    def test_single_client_has_no_pairwise_division(self):
        self.assertEqual(self.check({1:[2,2]}, {1:[1]}), ([],{}))

    def test_orphan_member_raises_explicit_error(self):
        with self.assertRaisesRegex(ValueError, 'missing from active'):
            self.check({1:[2,2]}, {1:[1,296]})

    def test_zero_distribution_cannot_continue(self):
        with self.assertRaisesRegex(ValueError, 'Invalid active distribution'):
            self.check({296:[0,0]}, {1:[296]})

if __name__ == '__main__':
    unittest.main()
