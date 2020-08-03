import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csr import csr_matrix


class DataLoader(object):
    def __init__(self, path):
        super(DataLoader, self).__init__()

        self.path = path
        self.n_items = self._n_items()

    def _n_items(self):
        tr_items = np.load(os.path.join(self.path, "skill2id.npy"), allow_pickle=True)
        return len(tr_items)

    def load_data(self, datatype: str = "train"):
        if datatype == "train":
            return self._load_train_data()
        elif datatype == "full_data":
            return self._load_full_data()
        else:
            return self._load_valid_test_data(datatype)

    def _load_train_data(self) -> csr_matrix:
        df = pd.read_csv(os.path.join(self.path, "train.csv"))
        users, skills, rates = df["user"], df["skill"], df["rate"]
        n_users = len(np.load(os.path.join(self.path, "user2id.npy"), allow_pickle=True))
        data = sparse.csr_matrix(
            (rates, (users, skills)),
            dtype="float32",
            shape=(n_users, self.n_items),
        )
        return data

    def _load_full_data(self) -> csr_matrix:
        df = pd.read_csv(os.path.join(self.path, "full_enc_data.csv"))
        users, skills, rates = df["user"], df["skill"], df["rate"]
        n_users = len(np.load(os.path.join(self.path, "user2id.npy"), allow_pickle=True))
        data = sparse.csr_matrix(
            (rates, (users, skills)),
            dtype="float32",
            shape=(n_users, self.n_items),
        )
        return data

    def _load_valid_test_data(self, datatype: str) -> Tuple[csr_matrix, csr_matrix]:
        tp_tr = pd.read_csv(os.path.join(self.path, "{}_tr.csv".format(datatype)))
        tp_te = pd.read_csv(os.path.join(self.path, "{}_te.csv".format(datatype)))

        start_idx = min(tp_tr["user"].min(), tp_te["user"].min())
        end_idx = max(tp_tr["user"].max(), tp_te["user"].max())

        rows_tr, cols_tr, rates_tr = tp_tr["user"] - start_idx, tp_tr["skill"], tp_tr["rate"]
        rows_te, cols_te, rates_te = tp_te["user"] - start_idx, tp_te["skill"], tp_te["rate"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float32",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float32",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        return data_tr, data_te