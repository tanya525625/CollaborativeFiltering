import os

import torch
import numpy as np
import pandas as pd
from scipy import sparse
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings

from utils.models import MultiVAE


flair_forward = FlairEmbeddings('news-forward-fast')
count = 0
embeddings_dict = {}


def make_one_hot_embedding(curr_skills, all_skills):
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


def make_flair_emb(x):
    x = x.replace(']', '').replace('[', '').replace("'", '').split(', ')
    x = ' '.join(x)
    global count
    print(count)
    count += 1
    if x:
        sentence = Sentence(x)
        flair_forward.embed(sentence)
        new_list = []
        for token in sentence:
            new_list.append(np.array(token.embedding.tolist()))
        return np.array(sentence[0].embedding.tolist())
    else:
        return None


def make_mean_embedding(curr_skills, *args, vects_count=10):
    """
    Function for finding the mean value of embedding to make predictions
    more accurate and stable
    """
    global count
    count += 1
    print(count)
    all_skills = args[1]
    model = args[0]

    if str(curr_skills) not in embeddings_dict.keys():
        embeddings = []
        arr = make_one_hot_embedding(curr_skills, all_skills)
        arr = np.array(arr).reshape([1, 1304])
        X = torch.FloatTensor(arr).to(device)
        for i in range(vects_count):
            X_out, mu, logvar = model(X)
            emb_vector = X_out.cpu().detach().numpy()[0]
            embeddings.append(emb_vector)
        emb_matrix = np.array(embeddings)
        emb_matrix = np.transpose(emb_matrix).tolist()
        vect = [np.mean(lst) for lst in emb_matrix]
        embeddings_dict.update({str(curr_skills): vect})
    else:
        vect = embeddings_dict[str(curr_skills)]
    return vect


def make_embedding(model, df):
    # df['skills_dist'] = df['skills_dist'].apply(make_one_hot_embedding, args=all_skills)
    emb_vacancies_matrix = df['skills_dist'].apply(make_mean_embedding, args=list([model, all_skills]))
    # emb_vacancies_matrix = np.array(df['skills_dist'].apply(make_flair_emb))
    # emb_vacancies_matrix = torch.from_numpy(emb_vacancies_matrix)
    # X = torch.FloatTensor(df['skills_dist']).to(device)
    # X_out, mu, logvar = model(X)
    # emb_vector = X_out.cpu().detach().numpy()
    return np.array(emb_vacancies_matrix)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_name = 'vae_on_filtered_dataset'
    log_dir = "results"
    model_weights = os.path.join(log_dir, 'weights')
    data_dir = "data"
    dataset_path = os.path.join(data_dir, "filtered_dataset.json")
    output_path = os.path.join(data_dir, "DBSKAN_embeddings.npy")

    p_dims, q_dims, dropout_enc, dropout_dec = [200, 600, 1304], [1304, 600, 200], [0.5, 0.0], [0.0, 0.0]
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

