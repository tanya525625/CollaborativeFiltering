import os
import json
from copy import copy

import torch
import pandas as pd
import numpy as np

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
    skills_list = np.load(os.path.join(path_to_dicts, 'skill2id.npy'))
    user_list = np.load(os.path.join(DATA_DIR, 'user2id.npy'))
    sampled_z = sampled_z.cpu().detach().numpy()
    df = pd.DataFrame(data=sampled_z, index=user_list, columns=skills_list)
    return df


def read_json(path):
    with open(path, 'r') as file:
        data = file.read()

    return json.loads(data)


def process_results(df, key):
    res_rows = []
    user_skills = []
    for i, row in enumerate(df.values):
        user_skills.clear()
        for j, val in enumerate(row):
            if val > float(key):
                user_skills.append(df.columns[j])
        res_rows.append({str(df.index[i]): copy(user_skills)})
    return res_rows


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = "data"
    out_path = "results"
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
    data_tr = loader.load_data("train")
    res_df = make_prediction(model, data_tr, DATA_DIR)
    # res.to_csv(os.path.join(out_path, f"prediction_{args.n_epochs}.csv"), sep=';')
    key = 3
    recoms = process_results(res_df, key)

    with open(os.path.join(out_path, f'prediction_{args.n_epochs}.json'), 'w') as json_file:
        for row in recoms:
            json_str = json.dumps(row)
            json_file.write(json_str + '\n')


