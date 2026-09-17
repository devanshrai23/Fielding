from fedscale.cloud.client_manager import ClientManager

class MockArgs:
    def __init__(self):
        self.model_impact_aware_reclustering = True
        self.model_impact_threshold = 0.05
        self.model_impact_smoothing = False

class MockMetadata:
    def __init__(self, top1):
        self.top1_accuracies = top1

class MockClientManager(ClientManager):
    def __init__(self):
        self.args = MockArgs()
        self.data_drifted_clients = {1: [101, 102]}
        self.client_metadata = {}
        self.total_data_drift_events = 0

    def getDriftedClients(self):
        return [101, 102]

    def getUniqueId(self, host, client):
        return client

mgr = MockClientManager()
mgr.client_metadata[101] = MockMetadata([])
mgr.client_metadata[102] = MockMetadata([0.72])
print("Test 1 (missing/insufficient data):", mgr._check_model_impact())

mgr.client_metadata[101] = MockMetadata([0.72, 0.65]) # impact = 0.07
mgr.client_metadata[102] = MockMetadata([0.65, 0.72]) # impact = 0
print("Test 2 (avg impact 0.035 < 0.05):", mgr._check_model_impact())

mgr.client_metadata[101] = MockMetadata([0.72, 0.65]) # impact = 0.07
mgr.client_metadata[102] = MockMetadata([0.80, 0.70]) # impact = 0.10
print("Test 3 (avg impact 0.085 >= 0.05):", mgr._check_model_impact())

mgr.args.model_impact_aware_reclustering = False
print("Test 4 (disabled):", mgr._check_model_impact())
