import ast
import copy
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import normalize
from fedscale.cloud.auxo.vendor.clustering import QTable

module=ast.parse(Path('fedscale/cloud/auxo/client_manager.py').read_text())
cls=next(n for n in module.body if isinstance(n,ast.ClassDef)); cls.bases=[]
ns=dict(np=np,copy=copy,logging=logging,QTable=QTable,normalize=normalize)
exec(compile(ast.Module(body=[cls],type_ignores=[]),'<auxo adapter>','exec'),ns)

class AuxoTests(unittest.TestCase):
 def manager(self):
  m=object.__new__(ns['AuxoClientManager'])
  m.auxo_table=QTable(303,known_clt=15,train_ratio=100/302,split_round=None)
  m.auxo_index={c:c for c in range(1,303)}
  m.feasibleClients=[list(range(1,303)),list(range(1,303))]
  m.current_clusters=[1];m.cluster_to_center={};m.data_drifted_clients={}
  rng=np.random.default_rng(2)
  m.auxo_updates={1:{c:rng.normal(size=20) for c in range(1,101)}}
  m.global_client_update_label_counts=lambda r:None
  m._prepare_active_distributions=lambda:{c:[.5,.5] for c in range(1,303)}
  m.reset_sampler=lambda k:None
  m._check_model_impact=lambda:True
  return m
 def test_baseline_split_partitions_every_real_client(self):
  m=self.manager();mapping,_=m.clientReclusterAll(round=2)
  self.assertEqual(m.auxo_table.num_model,2)
  self.assertEqual(mapping,{1:1,2:1})
  members=[c for k in m.current_clusters for c in m.feasibleClients[k]]
  self.assertEqual(sorted(members),list(range(1,303)))
 def test_rejected_candidate_does_not_modify_table(self):
  m=self.manager();m.clientReclusterAll(round=2)
  k=max(m.current_clusters,key=lambda k:len(m.feasibleClients[k]))
  clients=m.feasibleClients[k][:20]
  m.auxo_updates={k:{c:np.random.default_rng(c).normal(size=20) for c in clients}}
  m._check_model_impact=lambda:False
  before=copy.deepcopy(m.auxo_table.y_kmeans)
  m.clientReclusterAll(round=3)
  np.testing.assert_array_equal(before,m.auxo_table.y_kmeans)
  self.assertTrue(m.auxo_updates[k])
 def test_zero_divergence_reward_is_finite(self):
  q=QTable(10)
  self.assertTrue(np.isfinite(q.dist_to_reward([0.,0.],1)).all())
 def test_global_reference_updates_are_excluded(self):
  m=self.manager();before=copy.deepcopy(m.auxo_updates)
  m.record_update(1,0,[np.ones(20)],[np.zeros(20)])
  np.testing.assert_equal(before,m.auxo_updates)

if __name__=='__main__': unittest.main()
