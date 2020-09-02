import os

import torch
import numpy as np
import pandas as pd
from scipy import sparse

from utils.models import MultiVAE


def make_one_hot_embedding(curr_skills, *all_skills):
  skills_indexes = [all_skills.index(skill) for skill in curr_skills]
  curr_skills_count = len(curr_skills)
  vect = [1] * curr_skills_count
  rows_indexes = [0] * curr_skills_count
  data = sparse.csr_matrix(
          (vect, (rows_indexes, skills_indexes)),
          dtype="int",
          shape=(1, len(all_skills)),
      )
  return data.toarray()[0]


def make_embedding(model, df):
    emb_vacancies_matrix = df['skills_dist'].apply(make_one_hot_embedding, args=(all_skills))
    X = torch.FloatTensor(emb_vacancies_matrix).to(device)
    X_out, mu, logvar = model(X)
    emb_vector = X_out.cpu().detach().numpy()
    return emb_vector


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_name = 'vae_on_filtered_dataset'
    log_dir = "results"
    model_weights = os.path.join(log_dir, 'weights')
    data_dir = "data"
    dataset_path = os.path.join(data_dir, "filtered_dataset.json")
    output_path = os.path.join(data_dir, "DBSKAN_embeddings.npy")

    p_dims, q_dims, dropout_enc, dropout_dec = [200, 600, 51], [51, 600, 200], [0.5, 0.0], [0.0, 0.0]
    model = MultiVAE(
        p_dims=p_dims,
        q_dims=q_dims,
        dropout_enc=dropout_enc,
        dropout_dec=dropout_dec,
    )
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_weights, model_name + ".pt")))

    all_skills = np.load(os.path.join(data_dir, 'skill2id.npy'),  allow_pickle=True).tolist()
    dataset = pd.read_json(dataset_path, lines=True)
    emb_matr = make_embedding(model, dataset)
    np.save(output_path, emb_matr)

