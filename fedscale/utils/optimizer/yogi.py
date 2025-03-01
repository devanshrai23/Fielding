import torch
import numpy as np
import logging


class YoGi:
    def __init__(self, eta=1e-2, tau=1e-3, beta=0.9, beta2=0.99):
        self.eta = eta
        self.tau = tau
        self.beta = beta

        self.beta2 = beta2
        self.v_t = None
        self.m_t = None

    def update(self, gradients):
        update_gradients = []
        # first, flatten the gradients
        gradients_flatten = torch.concat([g.flatten() for g in gradients])
        if self.v_t is None:
            self.v_t = torch.full_like(gradients_flatten, self.tau)
            self.m_t = torch.full_like(gradients_flatten, 0.0)
        
        gradient_square = gradients_flatten**2
        self.m_t = self.beta * self.m_t + (1.0 - self.beta) * gradients_flatten
        self.v_t = self.v_t - (1.0 - self.beta2) * gradient_square * torch.sign(self.v_t - gradient_square)
        yogi_learning_rate = self.eta / (torch.sqrt(self.v_t) + self.tau)
        logging.info(f"yogi_learning_rate: {yogi_learning_rate}")

        update_flatten = yogi_learning_rate * self.m_t
        for idx, gradient in enumerate(gradients):
            num_elements = gradient.numel()
            update_gradients.append(update_flatten[:num_elements].reshape(gradient.shape))
            update_flatten = update_flatten[num_elements:]
        return update_gradients
