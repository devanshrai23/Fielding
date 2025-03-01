import numpy as np
import torch
import logging


class TorchServerOptimizer(object):
    """This is a abstract server optimizer class

    Args:
        mode (string): mode of gradient aggregation policy
        args (distionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
        device (string): Runtime device type
        sample_seed (int): Random seed

    """

    def __init__(self, mode, args, device, sample_seed=233):
        self.mode = mode
        self.args = args
        self.device = device

        logging.info(f"Use optimizer: {mode}")
        if mode == "fed-yogi":
            logging.info(f"Use fed-yogi")
            from fedscale.utils.optimizer.yogi import YoGi

            self.gradient_controller = YoGi(
                eta=args.yogi_eta,
                tau=args.yogi_tau,
                beta=args.yogi_beta,
                beta2=args.yogi_beta2,
            )

    def reset_state(self):
        """Reset the optimizer state when a new cluster is created"""
        if self.mode == "fed-yogi":
            logging.info("Reset fed-yogi state")
            self.gradient_controller.v_t = None
            self.gradient_controller.m_t = None

    def update_round_gradient(
        self, last_model, current_model, target_model, client_training_results=None, learning_rate=None
    ):
        """update global model based on different policy

        Args:
            last_model (list of tensor weight): A list of global model weight in last round.
            current_model (list of tensor weight): A list of global model weight in this round.
            target_model (PyTorch or TensorFlow nn module): Aggregated model.
            client_training_results list of gradients from every clients, for q-fedavg

        """
        if self.mode == "fed-yogi":
            """
            "Adaptive Federated Optimizations",
            Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, H. Brendan McMahan,
            ICLR 2021.
            """
            if last_model[0].is_cuda:
                last_model = [x.cpu() for x in last_model]
            if current_model[0].is_cuda:
                current_model = [x.cpu() for x in current_model]

            diff_weight = self.gradient_controller.update(
                [pb - pa for pa, pb in zip(last_model, current_model)]
            )

            new_state_dict = {
                name: torch.from_numpy(
                    np.array(last_model[idx] + diff_weight[idx], dtype=np.float32)
                )
                for idx, name in enumerate(target_model.state_dict().keys())
            }

            target_model.load_state_dict(new_state_dict)

        elif self.mode == "q-fedavg":
            """
            "Fair Resource Allocation in Federated Learning", Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith, ICLR 2020.
            """
            if learning_rate is None:
                learning_rate = self.args.learning_rate
            qfedq = self.args.qfed_q
            num_clients = len(client_training_results)
            logging.info(f"Use q-fedavg with learning rate: {learning_rate}, q: {qfedq}, over {num_clients} clients")

            Deltas, hs = None, 0.0
            last_model = [x.to(device=self.device) for x in last_model]
            coefficients = []
            
            for result in client_training_results:
                # plug in the weight updates into the gradient
                update_weights = result["update_weight"]
                if type(update_weights) is dict:
                    update_weights = [x for x in update_weights.values()]

                weights = [
                    torch.tensor(x).to(device=self.device) for x in update_weights
                ]
                grads = [
                    (u - v) * 1.0 / learning_rate for u, v in zip(last_model, weights)
                ]
                loss = result["moving_loss"]

                if Deltas is None:
                    Deltas = [
                        np.float_power(loss + 1e-10, qfedq) * grad for grad in grads
                    ]
                else:
                    for idx in range(len(Deltas)):
                        Deltas[idx] += np.float_power(loss + 1e-10, qfedq) * grads[idx]

                # estimation of the local Lipchitz constant
                hs += qfedq * np.float_power(loss + 1e-10, (qfedq - 1)) * torch.sum(
                    torch.stack([torch.square(grad).sum() for grad in grads])
                ) + (1.0 / learning_rate) * np.float_power(loss + 1e-10, qfedq)
                coefficients.append((np.float_power(loss + 1e-10, qfedq), hs.item()))

            # update global model
            new_state_dict = {
                name: last_model[idx] - (Deltas[idx] / hs) for idx, name in enumerate(target_model.state_dict().keys())
            }
            target_model.load_state_dict(new_state_dict)
            logging.info(f"coefficients: {coefficients}")

        else:
            # The default optimizer, FedAvg, has been applied in aggregator.py on the fly
            pass