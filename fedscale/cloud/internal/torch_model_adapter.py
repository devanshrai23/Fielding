from typing import List

import numpy as np
import torch
import copy
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.internal.model_adapter_base import ModelAdapterBase


class TorchModelAdapter(ModelAdapterBase):
    """
    Adapts functions to pytorch models.
    """
    def __init__(self, model: torch.nn.Module, optimizer: TorchServerOptimizer = None):
        """
        Initializes a TorchModelAdapter.
        :param model: the PyTorch model to adapt
        :param optimizer: the optimizer to apply weights, when specified.
        """
        self.model = model
        self.optimizer = optimizer

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load the model from a checkpoint file.
        :param checkpoint_path: the path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # TorchServerOptimizer doesn't have a load_state_dict method

    def save_checkpoint(self, checkpoint_path: str):
        """
        Save the model to a checkpoint file.
        :param checkpoint_path: the path to the checkpoint file.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        # TorchServerOptimizer doesn't have a state_dict method
        torch.save(checkpoint, checkpoint_path)

    def reset_optimizer_state(self):
        """
        Reset the optimizer state.
        """
        if self.optimizer:
            self.optimizer.reset_state()
    
    def set_weights(self, weights: List[np.ndarray], is_aggregator=True, client_training_results=None,
                    current_learning_rate=None):
        """
        Set the model's weights to the numpy weights array.
        :param weights: numpy weights array
        :param is_aggregator: boolean indicating whether the caller is the aggregator
        :param client_training_results: list of gradients from every clients, for q-fedavg
        """
        last_grad_weights = [param.data.clone() for param in self.model.state_dict().values()]
        new_state_dict = {
            name: torch.from_numpy(np.asarray(weights[i], dtype=np.float32))
            for i, name in enumerate(self.model.state_dict().keys())
        }
        self.model.load_state_dict(new_state_dict)
        if self.optimizer and is_aggregator:
            weights_origin = copy.deepcopy(weights)
            weights = [torch.tensor(x) for x in weights_origin]
            self.optimizer.update_round_gradient(\
                last_grad_weights, weights, self.model, client_training_results, current_learning_rate) 

    def get_weights(self) -> List[np.ndarray]:
        """
        Get the model's weights as a numpy weights array. Note that it doesn't contain layer names. Rather, index 0
        contains the model's first layer weights, and index N contains the N+1 layer's weights.
        :return: A numpy array
        """
        return [params.data.clone() for params in self.model.state_dict().values()]

    def get_model(self):
        """
        Get the instantiated framework specific model including the architecture.
        """
        return self.model
    