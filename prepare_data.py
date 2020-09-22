import os
import sys
import math
import random
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


def split_users(
    users, test_users_size: Union[float, int]
) -> Tuple[pd.Index, pd.Index, pd.Index]:

    n_users = users.size

    if isinstance(test_users_size, int):
        n_heldout_users = test_users_size
    else:
        n_heldout_users = int(test_users_size * n_users)

    tr_users = users[: (n_users - n_heldout_users * 2)]
    vd_users = users[(n_users - n_heldout_users * 2) : (n_users - n_heldout_users)]
    te_users = users[(n_users - n_heldout_users) :]

    return tr_users, vd_users, te_users


def split_train_test(
    data: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_grouped_by_user = data.groupby("user_id")
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (nm, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(
                    n_items_u, size=int(test_size * n_items_u), replace=False
                ).astype("int64")
            ] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def make_table_user_item(df, unique_skills, sparsity, idf_dict):
    columns_names = ['user', 'skill', 'rate']
    pd.DataFrame(columns=columns_names)
    pares = []
    for ind, row in df.iterrows():
        pare = row.tolist()
        user = pare[0]
        skill_list = pare[1]
        zero_rows = make_skills_zero_rows(user, unique_skills, sparsity)
        pares += zero_rows
        for skill in skill_list:
            pares.append([user, skill, idf_dict[skill]])
    new_df = pd.DataFrame(pares, columns=columns_names)
    return new_df


def numerize(df: pd.DataFrame, skills2id, user2id) -> pd.DataFrame:
    rows = []
    for ind, row in df.iterrows():
        user_id = user2id[row[0]]
        skill_id = skills2id[row[1]]
        rate = row[2]
        rows.append([user_id, skill_id, rate])
    return pd.DataFrame(rows, columns=['user', 'skill', 'rate'])


def determine_unique_skills(df):
    new_list = []
    for skill_list in df['skill'].values:
        new_list += skill_list
    return set(new_list)


def make_skills_zero_rows(user, unique_skills, sparsity):
    number_of_ready_rows = 0
    rows = []
    unique_skills = list(unique_skills)
    skills_count = len(unique_skills)
    added_skills = []
    rows.clear()
    while number_of_ready_rows < sparsity:
        ind = random.randint(0, skills_count - 1)
        if unique_skills[ind] not in added_skills:
            added_skills.append(unique_skills[ind])
            number_of_ready_rows += 1
            rows.append([user, unique_skills[ind], 0])
    return rows


def split_data(data, train_size):
    train = data.iloc[0:int(len(data) * train_size)]
    test_te = data.iloc[int(len(data) * train_size):int(len(data) * (train_size + 0.05))]
    test_tr = data.iloc[int(len(data) * (train_size + 0.05)):int(len(data) * (train_size + 0.1))]
    val_te = data.iloc[int(len(data) * (train_size + 0.1)):int(len(data) * (train_size + 0.15))]
    val_tr = data.iloc[int(len(data) * (train_size + 0.15)):]
    return train, test_te, test_tr, val_te, val_tr


def compute_idf(word, corpus):
  if word in corpus:
    return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))
  else:
    return 0


def prepare_data(df, new_colnames):
    df.drop('count', axis=1)
    keep_cols = ["id", "skills_dist"]
    df = df[keep_cols]
    df.columns = new_colnames
    return df


def find_unique_skills(general_data, vacancies_data):
    unique_skills = determine_unique_skills(general_data)
    vac_unique_skills = determine_unique_skills(vacancies_data)
    all_unique_skills = set.union(vac_unique_skills, unique_skills)
    skill2id = dict((sid, i) for (i, sid) in enumerate(all_unique_skills))
    return all_unique_skills, skill2id


def make_processed_data(data, out_path, unique_skills, skill2id, idf_dict):
    sparsity = 3  # number 0 rows for each user
    data = make_table_user_item(data, unique_skills, sparsity, idf_dict)
    user2id = dict((sid, i) for (i, sid) in enumerate(set(data['user'].tolist())))
    np.save(os.path.join(out_path, 'user2id.npy'), list(set(user2id.keys())))
    data = numerize(data, skill2id, user2id)
    data = data.sample(frac=1)
    train_size = 0.8
    train, test_te, test_tr, val_te, val_tr = split_data(data, train_size)
    # write data
    data.to_csv(os.path.join(out_path, "full_enc_data.csv"), index=False)
    train.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val_tr.to_csv(os.path.join(out_path, "validation_tr.csv"), index=False)
    val_te.to_csv(os.path.join(out_path, "validation_te.csv"), index=False)
    test_tr.to_csv(os.path.join(out_path, "test_tr.csv"), index=False)
    test_te.to_csv(os.path.join(out_path, "test_te.csv"), index=False)


def make_idf_dict(dataset, skills_list):
    idf_dict = {}
    corpus = []
    for skill_list in dataset['skill'].values.tolist():
      corpus.extend(skill_list)
    for skill in skills_list:
      idf_dict.update({skill: compute_idf(skill, corpus)})
    return idf_dict


def main():
    DATA_DIR = Path("data")
    united_path = os.path.join(DATA_DIR, "bigger_balanced_filtered_anywhere")
    vac_out_path = os.path.join(united_path, 'vacancy_data')
    gen_out_path = os.path.join(united_path, 'employee_data')
    os.mkdir(united_path)
    os.mkdir(gen_out_path)
    os.mkdir(vac_out_path)
    new_colnames = ["user_id", "skill"]
    general_filename = "1000_balanced_anywhere_dataset.json"
    vacancy_dataset_filename = "1000_balanced_anywhere_dataset.json"
    is_skill2id = True
    skill2id_path = os.path.join(DATA_DIR, "filtered_anywhere")

    data = pd.read_json(DATA_DIR / general_filename, lines=True)
    vac_data = pd.read_json(DATA_DIR / vacancy_dataset_filename, lines=True)

    data = prepare_data(data, new_colnames)
    vacancies_data = prepare_data(vac_data, new_colnames)
    if is_skill2id:
        unique_skills = np.load(os.path.join(skill2id_path, "skill2id.npy"), allow_pickle=True).tolist()
        skill2id = dict((sid, i) for (i, sid) in enumerate(unique_skills))
    else:
        unique_skills, skill2id = find_unique_skills(data, vacancies_data)
    np.save(os.path.join(united_path, 'skill2id.npy'), list(skill2id.keys()))
    idf_dict = make_idf_dict(data, list(skill2id.keys()))
    make_processed_data(data, gen_out_path, unique_skills, skill2id, idf_dict)
    make_processed_data(vac_data, vac_out_path, unique_skills, skill2id, idf_dict)


if __name__ == "__main__":
    main()
