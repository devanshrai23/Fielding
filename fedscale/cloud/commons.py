# Define Basic Experiment Setup
from enum import Enum

SIMULATION_MODE = 'simulation'
DEPLOYMENT_MODE = 'deployment'

# Define Basic FL Events
UPDATE_MODEL = 'update_model'
MODEL_TEST = 'model_test'
SHUT_DOWN = 'terminate_executor'
START_ROUND = 'start_round'
CLIENT_CONNECT = 'client_connect'
CLIENT_TRAIN = 'client_train'
DUMMY_EVENT = 'dummy_event'
UPLOAD_MODEL = 'upload_model'
CLUSTER_SPLIT = 'cluster_split'
EXECUTOR_RECONNECT = 'executor_reconnect'
GLOBAL_GRADIENT = 'global_gradient'
GLOBAL_GRADIENT_COMPLETE = 'global_gradient_complete'
PICK_BEST_MODEL = 'pick_best_model'

# PLACEHOLD
DUMMY_RESPONSE = 'N'


TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'

def decode_clusterid(msg):
    """Decode message into event type and cluster id

    Args:
        msg (string): message from client
    """
    msg_split = msg.split('-')
    if len(msg_split) > 1:
        return msg_split[0], int(msg_split[1])
    else:
        return msg, 0


def encode_clusterid( msg_type, cluster_id=0):
    return f'{msg_type}-{cluster_id}'