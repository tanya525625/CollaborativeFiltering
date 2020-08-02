import os
import json

import torch
import pandas as pd

from utils.models import MultiVAE
from utils.parser import parse_args
from utils.data_loader import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def process_args(n_items):
    p_dims = eval(args.p_dims)
    q_dims = eval(args.q_dims)
    if q_dims:
        assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
        assert (
                q_dims[-1] == p_dims[0]
        ), "Latent dimension for p- and q- network mismatches."
    else:
        q_dims = p_dims[::-1]
    q_dims = [n_items] + q_dims
    p_dims = p_dims + [n_items]
    dropout_enc = eval(args.dropout_enc)
    dropout_dec = eval(args.dropout_dec)

    return p_dims, q_dims, dropout_enc, dropout_dec


def make_prediction(model, matrix, path_to_dicts):
    arr = matrix.toarray()
    X = torch.FloatTensor(arr).to(device)
    X_out, mu, logvar = model(X)
    sampled_z, mu, logvar = model.forward(X_out)

    skill2id = read_json(os.path.join(path_to_dicts, 'dict_of_skills.json'))
    user2id = read_json(os.path.join(path_to_dicts, 'dict_of_users.json'))
    user_list = list(user2id.keys())
    skills_list = list(skill2id.keys())
    df = pd.DataFrame(data=sampled_z, index=user_list, columns=skills_list)

    return df


def read_json(path):
    with open(path, 'r') as file:
        data = file.read()

    return json.loads(data)


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = "data"
    data_path = os.path.join(DATA_DIR,"_".join([args.dataset, "processed"]))
    model_name = str("_".join(["pt", args.model]))
    log_dir = args.log_dir
    model_weights = os.path.join(log_dir, 'weights')
    data_loader = DataLoader(data_path)
    n_items = data_loader.n_items
    p_dims, q_dims, dropout_enc, dropout_dec = process_args(n_items)
    model = MultiVAE(
        p_dims=p_dims,
        q_dims=q_dims,
        dropout_enc=dropout_enc,
        dropout_dec=dropout_dec,
    )

    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_weights, model_name + ".pt")))
    loader = DataLoader(data_path)
    data_tr, data_te = loader.load_data("test")
    res = make_prediction(model, data_te, DATA_DIR)
    print(res)


