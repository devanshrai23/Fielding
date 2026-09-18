from fedscale.cloud.client_manager import ClientManager

class MockArgs:
    def __init__(self):
        self.model_impact_aware_reclustering = True
        self.model_impact_smoothing = True
        self.model_impact_smoothing_window = 3
        self.model_impact_threshold = 0.05

class MockMetadata:
    def __init__(self, top1):
        self.top1_accuracies = top1

class MockClientManager(ClientManager):
    def __init__(self, clients_data):
        self.args = MockArgs()
        self.data_drifted_clients = {1: list(clients_data.keys())}
        self.client_metadata = {}
        for k, v in clients_data.items():
            self.client_metadata[k] = MockMetadata(v)
            
    def getDriftedClients(self):
        return list(self.client_metadata.keys())

    def getUniqueId(self, host, client):
        return client

import logging
logging.basicConfig(level=logging.INFO)

print("\n--- Test 1: Window 3, [74, 73, 72, 72, 71, 65] ---")
# prev = [74, 73, 72] avg = 73
# recent = [72, 71, 65] avg = 69.333
# impact = 3.666
# 3.66 < 5, should SKIP
mgr1 = MockClientManager({101: [0.74, 0.73, 0.72, 0.72, 0.71, 0.65]})
mgr1._check_model_impact()

print("\n--- Test 2: RECLUSTER [80, 80, 80, 60, 60, 60] ---")
# prev = 80, recent = 60, impact = 20 > 5, should RECLUSTER
mgr2 = MockClientManager({102: [0.80, 0.80, 0.80, 0.60, 0.60, 0.60]})
mgr2._check_model_impact()

print("\n--- Test 3: Insufficient History [72, 71] ---")
mgr3 = MockClientManager({103: [0.72, 0.71]})
mgr3._check_model_impact()

print("\n--- Test 4: Missing Values [] ---")
mgr4 = MockClientManager({104: []})
mgr4._check_model_impact()

print("\n--- Test 5: Multiple clients, independent history ---")
mgr5 = MockClientManager({
    201: [0.74, 0.73, 0.72, 0.72, 0.71, 0.65], # impact 3.66%
    202: [0.80, 0.80, 0.80, 0.60, 0.60, 0.60]  # impact 20%
})
# average impact = 11.83% > 5%, RECLUSTER
mgr5._check_model_impact()

print("\n--- Test 6: Baseline (Smoothing Disabled) ---")
mgr6 = MockClientManager({101: [0.74, 0.73, 0.72, 0.72, 0.71, 0.65]})
mgr6.args.model_impact_smoothing = False
mgr6._check_model_impact()

