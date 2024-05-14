from glob import glob
import json
import re
import torch
from functools import wraps
import yaml
import rdkit
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from torch_geometric.utils import to_undirected, remove_self_loops
from sys import path
import importlib

with open("config.json") as config_file:
    config = json.loads(config_file.read())


class NodeFeaturizer:
    """
    In order make graphs PDB or molecular data, we need to featurize protein atoms to make nodes, i.e. develop a vocab.
    This class provides methods for both updating/storing a graph node vocab, and featurizing nodes based on this vocab.
    """

    def __init__(self, is_categorical: list, vocab: list = None) -> None:
        """
        Parameters:
            is_categorical: One-hot vector to indicate if feature is categorical (requires a vocab)
            vocab: previously collated vocab data for updating or featurizing
        """
        self.is_categorical = is_categorical

        if vocab:
            self.vocab = vocab
            # template n-hot vectors for categorical features.
            self.zero_vectors = [[0 for x in y] if y else None for y in self.vocab]
        else:
            self.vocab = [[] if x else None for x in is_categorical]

    def sort_and_update(self):
        """
        Updates template zero vectors and sorts vocabularies.
        To be invoked after any updates to vocab
        """
        for f_i in range(len(self.vocab)):
            if self.vocab[f_i]:
                self.vocab[f_i] = sorted(self.vocab[f_i])

        self.zero_vectors = [[0 for x in y] if y else None for y in self.vocab]

    def update_vocab(self, data_list):
        """
        accepts a list of data rows
        updates vocabulary at each categorical index
        """
        for row in data_list:
            for f_i in range(len(row)):
                if self.vocab[f_i] is None:
                    continue

                if row[f_i] not in self.vocab[f_i]:
                    self.vocab[f_i].append(row[f_i])

        self.sort_and_update()

    def featurize_data(self, data_list):
        """
        accepts a list of data rows
        turns each categorical index into an n-hot sequence
        returns resulting list of featurized data rows
        """
        featurized_data = []

        for row in data_list:
            featurized_row = []

            for f_index in range(len(row)):
                f_vocab = self.vocab[f_index]
                f_val = row[f_index]

                if self.is_categorical[f_index]:
                    one_hot = self.zero_vectors[f_index][:]

                    if f_val in f_vocab:
                        one_hot[f_vocab.index(f_val)] = 1

                    featurized_row += one_hot
                else:
                    featurized_row.append(f_val)
            featurized_data.append(featurized_row)

        return featurized_data

    def add_class(self, feature_index, class_value):
        """
        Manually insert a class into vocabulary.
        """
        self.vocab[feature_index].append(class_value)
        self.sort_and_update()

    def dump(self):
        return self.vocab
