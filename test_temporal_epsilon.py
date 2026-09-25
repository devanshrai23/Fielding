import sys
sys.path.append('c:\\fielding')
from fedscale.cloud.client_manager import ClientManager
from fedscale.cloud.internal.client_metadata import ClientMetadata

class MockArgs:
    def __init__(self):
        self.model_impact_aware_reclustering = True
        self.model_impact_smoothing = False
        self.model_impact_threshold = 0.05
        self.temporal_epsilon_reclustering = True
        self.temporal_epsilon_window = 10  # use smaller window for quick test
        self.epsilon_percentile = 95
        self.persistence_window = 5
        self.persistence_threshold = 0.8

class MockClientManager(ClientManager):
    def __init__(self):
        self.args = MockArgs()
        self.data_drifted_clients = {1: [101]}
        self.client_metadata = {101: ClientMetadata(0, 101, {'computation': 1, 'communication': 1})}
        self.client_metadata[101].top1_accuracies = [0.8, 0.7] # mock acc drop to trigger impact = 0.1 > 0.05
    def getDriftedClients(self):
        return [101]
    def getUniqueId(self, host, client):
        return client

import logging
logging.basicConfig(level=logging.INFO)

mgr = MockClientManager()

# B, C, D: Warmup test
print("\n--- Test D: Warm-up (history_size < 10) ---")
for i in range(5):
    # simulate some drift distributions
    mgr.client_metadata[101].prev_label_distribution = [0.1, 0.9]
    mgr.client_metadata[101].label_distribution = [0.1 + i*0.01, 0.9 - i*0.01]
    mgr._check_model_impact()
    
print("\n--- Test E: Activation (history_size >= 10) ---")
for i in range(5, 10):
    mgr.client_metadata[101].prev_label_distribution = [0.1, 0.9]
    mgr.client_metadata[101].label_distribution = [0.1 + i*0.01, 0.9 - i*0.01]
    mgr._check_model_impact()
    
print("\n--- Test F & G: delta > epsilon and delta <= epsilon ---")
mgr.client_metadata[101].prev_label_distribution = [1.0, 0.0]
mgr.client_metadata[101].label_distribution = [0.0, 1.0]
# Should trigger massive delta
print("Triggering massive delta...")
mgr._check_model_impact()

mgr.client_metadata[101].prev_label_distribution = [0.5, 0.5]
mgr.client_metadata[101].label_distribution = [0.5, 0.5]
print("Triggering zero delta...")
mgr._check_model_impact()

print("\n--- Test H: Disable Temporal Epsilon ---")
mgr.args.temporal_epsilon_reclustering = False
mgr._check_model_impact()

