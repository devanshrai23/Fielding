# -*- coding: utf-8 -*-
import csv
import logging
import random
import time
from collections import defaultdict
from random import Random

import numpy as np
from torch.utils.data import DataLoader
import pickle
import copy

#from argParser import args


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, args, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.numOfLabels = numOfClass
        self.client_label_cnt = defaultdict(set)
        self.flipped = set()
        self.sample_idx_avail_frame = {}

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def getClientLen(self):
        return len(self.partitions)

    def getClientLabel(self):
        return [len(self.client_label_cnt[i]) for i in range(self.getClientLen())]
    
    def getAllLabels(self):
        return self.labels

    def trace_partition(self, data_map_file, data_frame_file=None, client_start_end_rounds=None):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        client_id_maps = {}
        unique_client_ids = {}
        if data_frame_file:
            logging.info(f"with data frame mapping {data_frame_file}")
            with open(data_frame_file, "rb") as pkl_file:
                data_frame_mapping = pickle.load(pkl_file)
        else:
            data_frame_mapping = {}
        if self.args.data_mode == 'sliding_window_cumulate':
            if not client_start_end_rounds:
                logging.info("WARNING: sliding_window_cumulate mode but client_start_end_rounds is not provided, switch to sliding_window mode")
                self.args.data_mode = 'sliding_window'
            else:
                with open(client_start_end_rounds, "rb") as pkl_file:
                    self.clientid_start_end_rounds = pickle.load(pkl_file)
                logging.info(f"with client start end rounds mapping {client_start_end_rounds}")
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_client_ids:
                        unique_client_ids[client_id] = len(unique_client_ids)

                    client_id_maps[sample_id] = unique_client_ids[client_id]
                    self.client_label_cnt[unique_client_ids[client_id]].add(
                        row[-1])
                    
                    self.sample_idx_avail_frame[sample_id] = data_frame_mapping.get(sample_id, 0)
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_client_ids))]

        for idx in range(sample_id):
            self.partitions[client_id_maps[idx]].append(idx)

        logging.info(f"{len(unique_client_ids)} unique_client_ids: {unique_client_ids}")
        logging.info(f"partition sizes: {[len(p) for p in self.partitions]}")

    def partition_data_helper(self, num_clients, data_map_file=None, data_frame_file=None,
                              client_start_end_rounds_file=None):

        # read mapping file to partition trace
        if data_map_file is not None:
            self.trace_partition(data_map_file, data_frame_file, client_start_end_rounds_file)
        else:
            self.uniform_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        # random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition, istest, select_ratio=1, round=500):
        resultIndex = self.partitions[partition % len(self.partitions)]

        if self.args.data_mode == 'streaming':
            # filter out samples that are not available yet
            availIndex = list(filter(lambda x: self.sample_idx_avail_frame[x] <= round, resultIndex))
        elif self.args.data_mode == 'sliding_window':    
            # filter out samples that are not available yet and are not in the current window
            availIndex = list(filter(lambda x: self.sample_idx_avail_frame[x] <= round\
                                     and round - self.sample_idx_avail_frame[x] < 100, resultIndex))
        elif self.args.data_mode == 'sliding_window_cumulate':
            # filter out samples that are not available yet and are not in the current window
            client_start_round = self.clientid_start_end_rounds[partition+1][0]
            client_end_round = self.clientid_start_end_rounds[partition+1][1]
            if round < client_start_round:
                availIndex = []
            elif round <= client_end_round:
                availIndex = list(filter(lambda x: self.sample_idx_avail_frame[x] <= (round+98)\
                                     and (round+98) - self.sample_idx_avail_frame[x] < 100, resultIndex))
            else:
                availIndex = list(filter(lambda x: self.sample_idx_avail_frame[x] >= client_end_round-1, \
                                         resultIndex))
        else:
            # use all samples
            availIndex = copy.deepcopy(resultIndex)
        
        resultIndex = availIndex

        exeuteLength = int(len(resultIndex) * select_ratio) \
            if not istest else int(len(resultIndex) * self.args.test_ratio)
        if istest:
            # for testing data, shuffle before truncating
            self.rng.shuffle(resultIndex)

        resultIndex = resultIndex[:exeuteLength]
        # get the target labels of training samples in availIndex
        label_counts = [0] * self.getNumOfLabels()
        if len(resultIndex) > 0:
            label_records = {}
            all_labels = self.getAllLabels()
            for idx in resultIndex:
                sample_label = all_labels[idx]
                if sample_label not in label_records:
                    label_records[sample_label] = []
                label_counts[sample_label] += 1
                label_records[sample_label].append(idx)

        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex), label_counts

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}

def select_dataset(rank, partition, batch_size, args, isTest=False, collate_fn=None, select_ratio=1, client_test=False, round=500):
    """Load data given client Id"""
    data_partition, label_counts = partition.use(rank - 1, (isTest or client_test), select_ratio, round)
    
    if len(data_partition) == 0:
        return [], label_counts
    
    dropLast = False
    if (isTest or client_test):
        num_loaders = 0
    else:
        num_loaders = min(int(len(data_partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(data_partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn), label_counts
    return DataLoader(data_partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast), label_counts
