import logging
import math
import time

import torch
from torch.autograd import Variable
from overrides import overrides
from torch.nn import CTCLoss

from fedscale.cloud.execution.client_base import ClientBase
from fedscale.cloud.execution.optimizers import ClientOptimizer
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.dataloaders.nlp import mask_tokens
from fedscale.utils.model_test_module import test_pytorch_model, accuracy


class TorchClient(ClientBase):
    """Implements a PyTorch-based client for training and evaluation."""

    def __init__(self, args):
        """
        Initializes a torch client.
        :param args: Job args
        """
        self.args = args
        self.optimizer = ClientOptimizer()
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')
        if args.task == "detection":
            self.im_data = Variable(torch.FloatTensor(1).cuda())
            self.im_info = Variable(torch.FloatTensor(1).cuda())
            self.num_boxes = Variable(torch.LongTensor(1).cuda())
            self.gt_boxes = Variable(torch.FloatTensor(1).cuda())

        self.epoch_train_loss = 1e-4
        self.completed_steps = 0
        self.loss_squared = 0

        self.correct = 0
        self.top_5 = 0
        self.train_len = 0
        self.train_labels = []
        self.projection = None

    @overrides
    def train(self, client_data, model, conf):
        """
        Perform a training task.
        :param client_data: client training dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: training results
        """
        client_id = conf.client_id
        # logging.info(f"Start to train (CLIENT: {client_id}) ... ")
        tokenizer = conf.tokenizer

        model = model.to(device=self.device)
        model.train()

        if self.args.run_epochs:
            trained_unique_samples = len(client_data.dataset) * conf.local_steps
        else:
            trained_unique_samples = min(
                len(client_data.dataset), conf.local_steps * conf.batch_size)
        self.global_model = None

        if conf.gradient_policy == 'fed-prox':
            logging.info(f"use fed-prox gradient policy")
            # could be move to optimizer
            self.global_model = [param.data.clone() for param in model.parameters()]

        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None

        target_steps = conf.local_steps * len(client_data) if self.args.run_epochs else min(conf.local_steps, len(client_data))
        while self.completed_steps < target_steps:
            try:
                trainRes = self.train_step(client_data, conf, model, optimizer, criterion, target_steps)
            except Exception as ex:
                error_type = ex
                break

        state_dicts = model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts}
        results = {'client_id': client_id, 'moving_loss': self.epoch_train_loss,
                   'trained_size': trainRes["train_len"],
                       'success': self.completed_steps >= target_steps
                    }

        if error_type is None:
            logging.info(f"Training of (CLIENT: {client_id}) completes, success {results['success']}, trained_unique_samples {trained_unique_samples}, all labels {trainRes['train_labels']}")
        else:
            logging.info(f"Training of (CLIENT: {client_id}) failed as {error_type}")

        results['utility'] = math.sqrt(
            self.loss_squared) * float(trained_unique_samples)
        results['update_weight'] = model_param
        results['wall_duration'] = 0

        results['top_1'] = trainRes["top_1"]
        results['top_5'] = trainRes["top_5"]

        return results

    def get_optimizer(self, model, conf):
        optimizer = None
        if conf.task == "detection":
            lr = conf.learning_rate
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        params += [{'params': [value], 'lr': lr,
                                    'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        elif conf.task == 'nlp':

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": conf.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            # Bert pre-training setup
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters, lr=conf.learning_rate, weight_decay=1e-2)
            logging.info("use Adam")
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
            # logging.info(f"use SGD with lr {conf.learning_rate}")
        return optimizer

    def get_criterion(self, conf):

        criterion = None
        if conf.task == 'voice':
            from torch_baidu_ctc import CTCLoss
            criterion = CTCLoss(reduction='none').to(device=self.device)
        else:
            criterion = torch.nn.CrossEntropyLoss(
                reduction='none').to(device=self.device)
        return criterion

    def train_step(self, client_data, conf, model, optimizer, criterion, target_steps):
        train_start = time.time()

        for data_pair in client_data:
            if time.time() - train_start > 90:
                logging.info(f"Training of (CLIENT: {conf.client_id}) exit because of timeout")
                break
            try:
                if conf.task == 'nlp':
                    (data, _) = data_pair
                    data, target = mask_tokens(
                        data, tokenizer, conf, device=self.device)
                elif conf.task == 'voice':
                    (data, target, input_percentages,
                    target_sizes), _ = data_pair
                    input_sizes = input_percentages.mul_(
                        int(data.size(3))).int()
                elif conf.task == 'detection':
                    temp_data = data_pair
                    target = temp_data[4]
                    data = temp_data[0:4]
                else:
                    (data, target) = data_pair

                if conf.task == "detection":
                    self.im_data.resize_(data[0].size()).copy_(data[0])
                    self.im_info.resize_(data[1].size()).copy_(data[1])
                    self.gt_boxes.resize_(data[2].size()).copy_(data[2])
                    self.num_boxes.resize_(data[3].size()).copy_(data[3])
                elif conf.task == 'speech':
                    data = torch.unsqueeze(data, 1).to(device=self.device)
                elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                    (data, masks) = data
                    data, masks = Variable(data).to(
                        device=self.device), Variable(masks).to(device=self.device)

                else:
                    data = Variable(data).to(device=self.device)

                target = Variable(target).to(device=self.device)

                if conf.task == 'nlp':
                    outputs = model(data, labels=target)
                    loss = outputs[0]
                elif conf.task == 'voice':
                    outputs, output_sizes = model(data, input_sizes)
                    outputs = outputs.transpose(0, 1).float()  # TxNxH
                    loss = criterion(
                        outputs, target, output_sizes, target_sizes)
                elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                    outputs = model(
                        data, attention_mask=masks, labels=target)
                    loss = outputs.loss
                    output = outputs.logits
                elif conf.task == "detection":
                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label = model(
                        self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

                    loss = rpn_loss_cls + rpn_loss_box \
                        + RCNN_loss_cls + RCNN_loss_bbox

                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()

                else:
                    # cos=torch.nn.CosineSimilarity(dim=-1)
                    outputs = model(data)
                    if len(outputs) == 2 and outputs[1].shape[0] == target.shape[0]:
                        proj, output = outputs
                    else:
                        output = outputs
                    
                    loss = criterion(output, target)
                    # record test accuracy

                    topk=(1,)
                    # topk=(1, 5)
                    acc = accuracy(output, target, topk=topk)
                    self.correct += acc[0].item()
                    # self.top_5 += acc[1].item()
                    self.train_len += len(target)
                    if len(acc) > len(topk):
                        self.train_labels += acc[len(topk)]

                # ======== collect training feedback for other decision components [e.g., oort selector] ======

                if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2'):
                    loss_list = [loss.item()]  # [loss.mean().data.item()]

                elif conf.task == "detection":
                    loss_list = [loss.tolist()]
                    loss = loss.mean()
                else:
                    loss_list = loss.tolist()
                    loss = loss.mean()

                temp_loss = sum(loss_list) / float(len(loss_list))
                self.loss_squared += sum([l ** 2 for l in loss_list]
                                        ) / float(len(loss_list))
                # only measure the loss of the first epoch
                if self.completed_steps < len(client_data):
                    if self.epoch_train_loss == 1e-4:
                        self.epoch_train_loss = temp_loss
                    else:
                        self.epoch_train_loss = (
                                                        1. - conf.loss_decay) * self.epoch_train_loss + conf.loss_decay * temp_loss

                # ========= Define the backward loss ==============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ========= Weight handler ========================
                self.optimizer.update_client_weight(
                    conf, model, self.global_model if self.global_model is not None else None)
            except Exception as ex:
                logging.info(f"Training of (CLIENT: {conf.client_id}) failed as {ex}, skip")

            self.completed_steps += 1
            # logging.info(f"Training of (CLIENT: {conf.client_id}) completes {self.completed_steps} steps")

            if self.completed_steps == target_steps:
                break
            
        trainRes = {'top_1': self.correct, 'top_5': self.top_5, 'train_len': self.train_len, 'train_labels': self.train_labels}
        return trainRes

    @overrides
    def test(self, client_data, model, conf):
        """
        Perform a testing task.
        :param client_data: client evaluation dataset
        :param model: the framework-specific model
        :param conf: job config
        :return: testing results
        """
        # evalStart = time.time()
        try:
            # logging.info(f"Start to test for executor {conf.rank} ... ")
            if self.args.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=self.device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=self.device)
            test_loss, acc, acc_5, test_results = test_pytorch_model(conf.rank, model, client_data,
                                                                    device=self.device, criterion=criterion,
                                                                    tokenizer=conf.tokenizer)
        except Exception as ex:
            logging.info(f"Test of executor {conf.rank} failed as {ex}")
            test_results = {'top_1': 0, 'top_5': 0,
               'test_loss': 0, 'test_len': 0, 'wrong_predictions': []}
        return test_results

    @overrides
    def get_model_adapter(self, model) -> TorchModelAdapter:
        """
        Return framework-specific model adapter.
        :param model: the model
        :return: a model adapter containing the model
        """
        return TorchModelAdapter(model)
