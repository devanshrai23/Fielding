import sys
import types

class MockArgs:
    def __init__(self):
        self.model_impact_aware_reclustering = True
        self.model_impact_smoothing = False
        self.model_impact_threshold = 0.05
        self.temporal_epsilon_reclustering = True
        self.temporal_epsilon_window = 5
        self.epsilon_percentile = 95
        self.persistence_window = 5
        self.persistence_threshold = 0.8

class MockMetadata:
    def __init__(self):
        self.top1_accuracies = [0.8, 0.7]

class MockClientManager:
    def __init__(self):
        self.args = MockArgs()
        self.data_drifted_clients = {1: [101]}
        self.client_metadata = {101: MockMetadata()}
    def getDriftedClients(self):
        return [101]
    def getUniqueId(self, host, client):
        return client

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import re
with open('fedscale/cloud/client_manager.py', 'r') as f:
    content = f.read()
match = re.search(r'(?s)(    def _check_model_impact\(self\):.*?)(    def clientReclusterAllGradientBased)', content)
code = match.group(1).replace('    def _check_model_impact(self):', 'def _check_model_impact(self):')

import numpy as np
scipy = types.ModuleType('scipy')
spatial = types.ModuleType('spatial')
distance = types.ModuleType('distance')
distance.jensenshannon = lambda p, q: 0.5
spatial.distance = distance
scipy.spatial = spatial
sys.modules['scipy'] = scipy
sys.modules['scipy.spatial'] = spatial
sys.modules['scipy.spatial.distance'] = distance

exec(code, globals())
MockClientManager._check_model_impact = _check_model_impact

mgr = MockClientManager()

print("\n--- Test D: Warm-up (history_size < 5) ---")
for i in range(4):
    mgr.client_metadata[101].prev_label_distribution = [0.1, 0.9]
    mgr.client_metadata[101].label_distribution = [0.1 + i*0.01, 0.9 - i*0.01]
    mgr._check_model_impact()
    
print("\n--- Test E: Activation (history_size >= 5) ---")
for i in range(4, 6):
    mgr.client_metadata[101].prev_label_distribution = [0.1, 0.9]
    mgr.client_metadata[101].label_distribution = [0.1 + i*0.01, 0.9 - i*0.01]
    mgr._check_model_impact()
    
print("\n--- Test F & G: delta > epsilon and delta <= epsilon ---")
mgr.client_metadata[101].prev_label_distribution = [1.0, 0.0]
mgr.client_metadata[101].label_distribution = [0.0, 1.0]
distance.jensenshannon = lambda p, q: 1.0
print("Triggering massive delta...")
mgr._check_model_impact()

mgr.client_metadata[101].prev_label_distribution = [0.5, 0.5]
mgr.client_metadata[101].label_distribution = [0.5, 0.5]
distance.jensenshannon = lambda p, q: 0.0
print("Triggering zero delta...")
mgr._check_model_impact()

print("\n--- Test H: Disable Temporal Epsilon ---")
mgr.args.temporal_epsilon_reclustering = False
mgr._check_model_impact()

