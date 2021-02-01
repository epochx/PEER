#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

from skorch import NeuralNet

from wikigen.settings import SPLITS_PATH, DATASET_NAMES

np.random.seed(2)
torch.cuda.manual_seed(2)

parser = argparse.ArgumentParser(description="Sample from a trained RVAE")

parser.add_argument("json_file_path", type=str, help="Path to data")

parser.add_argument(
    "--device", type=str, help="Device", choices=["gpu", "cpu"], default="gpu"
)


class NeuralNetMultilabelClassifier(NeuralNet):
    def __init__(self, *args, **kwargs):
        super(NeuralNetMultilabelClassifier, self).__init__(*args, **kwargs)

    def score(self, X, y):
        probas = self.forward(X)
        predictions = (probas > 0.5).long()
        return accuracy_score(y.cpu().numpy(), predictions.cpu().numpy())


class TorchNeuralNetMultilabelClassifier(nn.Module):
    def __init__(self, num_units=300, nonlin=nn.ReLU()):
        super(TorchNeuralNetMultilabelClassifier, self).__init__()

        self.dense0 = nn.Linear(input_size, num_units)
        self.nonlin = nonlin
        # self.dense1 = torch.nn.Linear(num_units, 200)
        self.output = nn.Linear(num_units, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        # X = self.nonlin(self.dense1(X))
        X = self.sigmoid(self.output(X))
        return X


if __name__ == "__main__":

    args = parser.parse_args()

    multi_label_binarizer = MultiLabelBinarizer()

    file_name = os.path.basename(args.json_file_path)

    matches = [dataset_name in file_name for dataset_name in DATASET_NAMES]
    assert any(matches)
    assert sum(matches) == 1

    name = DATASET_NAMES[matches.index(True)]

    with open(args.json_file_path) as f:
        examples = json.load(f)

    base_indices_file_path = os.path.join(SPLITS_PATH, name)

    train_indices_file_path = base_indices_file_path + ".train.txt"
    with open(train_indices_file_path) as f:
        train_indices = np.array(
            list(map(lambda x: int(x.strip()), f.readlines()))
        )

    valid_indices_file_path = base_indices_file_path + ".valid.txt"
    with open(valid_indices_file_path) as f:
        valid_indices = np.array(
            list(map(lambda x: int(x.strip()), f.readlines()))
        )

    test_indices_file_path = base_indices_file_path + ".test.txt"
    with open(test_indices_file_path) as f:
        test_indices = np.array(
            list(map(lambda x: int(x.strip()), f.readlines()))
        )

    X = np.asarray([example["edit_representation"] for example in examples])

    key = "tgt_classes"
    labels = list(examples[0][key].keys())
    label2index = {label: index for index, label in enumerate(labels)}

    Y_labels = [
        [
            label2index[label]
            for label, value in example[key].items()
            if value == 1
        ]
        for example in examples
    ]

    Y = multi_label_binarizer.fit_transform(Y_labels)

    if args.device == "cpu":
        neural_net = MLPClassifier(max_iter=3000, random_state=2,)
        param_grid = {"hidden_layer_sizes": [[100,], [200,], [300,]]}

    else:

        X = torch.from_numpy(X).float()
        Y = torch.tensor(Y).squeeze().float()

        input_size = X.size(1)
        output_size = Y.size(1)

        neural_net = NeuralNetMultilabelClassifier(
            TorchNeuralNetMultilabelClassifier,
            max_epochs=3000,
            criterion=nn.BCELoss,
            iterator_train__shuffle=True,
            train_split=False,
            device=torch.device("cuda"),
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=0.0001,
            optimizer__lr=0.008,
            batch_size=min(1000, X.shape[0]),
        )

        param_grid = {
            "module__num_units": [100, 200, 300],
        }

    model = GridSearchCV(
        neural_net,
        param_grid,
        cv=[(train_indices, valid_indices)],
        verbose=1,
        n_jobs=3,
    )

    model.fit(X, Y)

    print(f"Train Acc: {model.score(X[train_indices], Y[train_indices])}")
    print(f"Valid Acc: {model.score(X[valid_indices], Y[valid_indices])}")
    print(f"Test Acc: {model.score(X[test_indices], Y[test_indices])}")

