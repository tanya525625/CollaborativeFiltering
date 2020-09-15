import os
import json
import random
from tqdm import tqdm
from copy import copy

import torch
import pandas as pd
import numpy as np

from utils.models import MultiVAE


def write_dataset(file_path, dataset):
    with open(file_path, 'w') as json_file:
        for row in dataset:
            json_str = json.dumps(row)
            json_file.write(json_str + '\n')


def concat_embs(embeddings):
    new_embs = []
    for emb in embeddings.values.tolist():
        new_embs.append(emb[0])
    global embs_length
    embs_length.append(len(new_embs))
    return new_embs


def find_k_nearest_neigh(vect, vectors, k_neigh):
    dists = [np.sqrt(np.sum((curr_vect-vect)*(curr_vect-vect))) for curr_vect in vectors]
    best_dists = np.sort(dists)[:k_neigh]
    return [vectors[dists.index(i)] for i in best_dists]


def make_one_hot_mean_emb(vector, nearest_vectors):
    new_vect = vector
    for near_vect in nearest_vectors:
        new_vect = new_vect | near_vect
    return new_vect


def oversampler(emb_list, ready_list_length, k_neigh):
    list_length = len(emb_list)
    diff = ready_list_length - list_length
    if list_length < k_neigh:
        k_neigh = list_length
    new_embs = []
    while len(new_embs) < diff:
        vect_ind = random.randint(0, list_length - 1)
        vectors_for_finding = copy(emb_list)
        vector = vectors_for_finding.pop(vect_ind)
        nearest_vectors = find_k_nearest_neigh(vector, vectors_for_finding, k_neigh)
        new_embs.append(make_one_hot_mean_emb(vector, nearest_vectors))
        # for VAE version
        # new_embs.append(find_mean_emb(nearest_vectors))
    new_embs += emb_list
    return new_embs


def undersampler(emb_list, ready_list_length):
    list_length = len(emb_list)
    diff = list_length - ready_list_length
    new_embs = []
    while len(new_embs) < diff:
        vect_ind = random.randint(0, list_length - 1)
        new_embs.append(emb_list[vect_ind])
    return new_embs


def find_mean_emb(embeddings):
    emb_matrix = np.array(embeddings)
    emb_matrix = np.transpose(emb_matrix).tolist()
    mean_vect = [np.mean(lst) for lst in emb_matrix]
    return mean_vect


def balance_dataset(dataset, k_neigh=20):
    dataset = dataset.groupby('classes').apply(concat_embs)
    mean_len = int(np.mean(embs_length))
    balanced_dataset = []
    embeddings_matr = []
    for i, emb_list in tqdm(enumerate(dataset)):
        if len(emb_list) >= mean_len:
            new_list = undersampler(emb_list, mean_len)
            for lst in new_list:
                embeddings_matr.append(np.array(lst))
                balanced_dataset.append([list(lst), i])
        else:
            new_list = oversampler(emb_list, mean_len, k_neigh)
            for lst in new_list:
                embeddings_matr.append(np.array(lst))
                balanced_dataset.append([list(lst), i])
    return pd.DataFrame(data=balanced_dataset, columns=['embeddings', 'classes']), np.array(embeddings_matr)


def norm_matr(matrix):
    users_max_els = matrix.max(axis=1)
    for i, row in enumerate(matrix):
        matrix[i] = row / users_max_els[i]
    return matrix


def process_results(matrix, skills_list, key):
    skills_dataset = []
    row_skills = []
    for i, row in enumerate(matrix):
        row_skills.clear()
        for j, val in enumerate(row):
            if val > key:
                row_skills.append(skills_list[j])
        skills_dataset.append(copy(row_skills))
    return skills_dataset


def decode_data(model, matrix, skills_list, key=0.7):
    X = torch.FloatTensor(matrix).to(device)
    X_out, mu, logvar = model(X)
    sampled_z, mu, logvar = model.forward(X_out)
    sampled_z = sampled_z.cpu().detach().numpy()
    normed_matr = norm_matr(sampled_z)
    return process_results(normed_matr, skills_list, key)


def make_train_dataset(dataset):
    rows = []
    for ind, row in enumerate(dataset):
        rows.append({"id": ind, "skills_dist": row, "count": len(row)})
    return rows


def decode_one_hot_enc(array, all_skills):
    skills_rows = []
    curr_skills = []
    for i, row in enumerate(array):
        curr_skills.clear()
        for j, el in enumerate(row):
            if el == 1:
                curr_skills.append(all_skills[j])
        skills_rows.append(copy(curr_skills))
    return skills_rows


def main():
    data_dir = "data"
    data_model_dir = os.path.join(data_dir, "filtered_dataset_path")
    out_filepath = os.path.join(data_dir, "balanced_one_hot_dataset.json")
    meta_out_path = os.path.join(data_dir, "balanced_meta.tsv")
    dataset = pd.read_parquet(os.path.join(data_dir, "classification_dataset.parquet"))
    balanced_df, emb_matrix = balance_dataset(dataset)
    np.save(os.path.join(data_dir, "balanced_embeddings.npy"), emb_matrix)
    # emb_matrix = np.load(os.path.join(data_dir, "balanced_embeddings.npy"), allow_pickle=True)
    skills_list = np.load(os.path.join(data_dir, "skill2id.npy"), allow_pickle=True).tolist()

    # Version with VAE
    # model_name = 'vae_on_filtered_dataset'
    # log_dir = "results"
    # p_dims, q_dims, dropout_enc, dropout_dec = [200, 600, 1304], [1304, 600, 200], [0.5, 0.0], [0.0, 0.0]
    # model_weights = os.path.join(log_dir, 'weights')
    # model = MultiVAE(
    #     p_dims=p_dims,
    #     q_dims=q_dims,
    #     dropout_enc=dropout_enc,
    #     dropout_dec=dropout_dec,
    # )
    # model.to(device)
    # model.load_state_dict(torch.load(os.path.join(model_weights, model_name + ".pt")))
    # decoded_data = decode_data(model, emb_matrix, skills_list)
    decoded_data = decode_one_hot_enc(emb_matrix, skills_list)
    balanced_df.columns = ['skills', 'classes']
    balanced_df['skills'] = decoded_data
    balanced_df.to_csv(meta_out_path, sep='\t', index=False)
    train_dataset = make_train_dataset(decoded_data)
    write_dataset(out_filepath, train_dataset)


if __name__ == "__main__":
    embs_length = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    main()