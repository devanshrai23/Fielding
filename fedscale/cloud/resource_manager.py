import threading

from fedscale.cloud import commons


class ResourceManager(object):
    """Schedule training tasks across GPUs/CPUs"""

    def __init__(self, experiment_mode):

        self.client_run_queue = [[]]
        self.client_run_queue_idx = [0]
        self.experiment_mode = experiment_mode
        self.update_lock = threading.Lock()

    def register_tasks(self, clientsToRun, cluster_id=0):
        # TODO: append new checkin client
        self.client_run_queue[cluster_id] = clientsToRun.copy()
        self.client_run_queue_idx[cluster_id] = 0
    
    def add_cluster(self):
        self.client_run_queue.append([])
        self.client_run_queue_idx.append(0)
    
    def clear_cluster(self, cluster_id):
        self.client_run_queue[cluster_id] = []
        self.client_run_queue_idx[cluster_id] = 0

    def get_task_length(self, cluster_id) -> int:
        """Number of tasks left in the queue

        Returns:
            int: Number of tasks left in the queue
        """
        self.update_lock.acquire()
        remaining_task_num: int = len(self.client_run_queue[cluster_id]) - self.client_run_queue_idx[cluster_id]
        self.update_lock.release()
        return remaining_task_num

    def remove_client_task(self, client_id, cluster_id=0):
        assert(client_id in self.client_run_queue[cluster_id],
               f"client task {client_id} is not in task queue")
        pass

    def has_next_task(self, client_id=None, cluster_id=0):
        # TODO: always has next task
        exist_next_task = False
        if self.experiment_mode == commons.SIMULATION_MODE:
            exist_next_task = self.client_run_queue_idx[cluster_id] < len(
                self.client_run_queue[cluster_id])
        else:
            exist_next_task = client_id in self.client_run_queue[cluster_id]
        return exist_next_task

    def get_next_task(self, client_id=None, cluster_id=0):
        # TODO: remove client id when finish
        next_task_id = None
        self.update_lock.acquire()
        if self.experiment_mode == commons.SIMULATION_MODE:
            if self.has_next_task(client_id, cluster_id):
                next_task_id = self.client_run_queue[cluster_id][self.client_run_queue_idx[cluster_id]]
                self.client_run_queue_idx[cluster_id] += 1
        else:
            if client_id in self.client_run_queue[cluster_id]:
                next_task_id = client_id
                self.client_run_queue[cluster_id].remove(next_task_id)

        self.update_lock.release()
        return next_task_id
