import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Union

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


def make_table_user_item(df, columns_names, unique_skills, sparsity):
    pd.DataFrame(columns=columns_names.append('rate'))
    pares = []
    for ind, row in df.iterrows():
        pare = row.tolist()
        user = pare[0]
        skill_list = pare[1]
        zero_rows = make_skills_zero_rows(user, unique_skills, sparsity)
        pares += zero_rows
        for skill in skill_list:
            pares.append([user, skill, 1])
    new_df = pd.DataFrame(pares, columns=columns_names)
    return new_df


def numerize(tp: pd.DataFrame, user2id: Dict, item2id: Dict) -> pd.DataFrame:
    user = [user2id[x] for x in tp["user_id"]]
    item = [item2id[x] for x in tp["skill"]]
    return pd.DataFrame(data={"user_id": user, "skill": item}, columns=["user_id", "skill"])


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


def main():
    DATA_DIR = Path("data")
    out_path = os.path.join(DATA_DIR, 'data_processed')
    new_colnames = ["user_id", "skill"]
    filename = "data.json"
    data = pd.read_json(DATA_DIR / filename, lines=True)
    data.drop('count', axis=1)
    keep_cols = ["id", "skills_dist"]
    data = data[keep_cols]
    data.columns = new_colnames
    unique_skills = determine_unique_skills(data)
    sparsity = 3  # number 0 rows for each user
    data = make_table_user_item(data, new_colnames, unique_skills, sparsity)

    test_size = 0.2

    tr_users, vd_users, te_users = split_users(
        data['user_id'], test_users_size=test_size
    )

    # Select the training observations raw data
    tr_obsrv = data.loc[data["user_id"].isin(tr_users)]
    tr_items = pd.unique(tr_obsrv["skill"])

    np.save(os.path.join(out_path, 'tr_items.npy'), tr_items)

    unique_uid = frozenset(data['user_id'].tolist())
    # Save index dictionaries to "numerate" later one
    item2id = dict((sid, i) for (i, sid) in enumerate(tr_items))
    user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    with open(os.path.join(DATA_DIR, 'dict_of_skills.json'), 'w') as f:
        json.dump(item2id, f)

    vd_obsrv = data[
        data["user_id"].isin(vd_users)
        & data["skill"].isin(tr_items)
    ]
    te_obsrv = data[
        data["user_id"].isin(te_users)
        & data["skill"].isin(tr_items)
        ]
    vd_items_tr, vd_items_te = split_train_test(vd_obsrv, test_size=test_size)
    te_items_tr, te_items_te = split_train_test(te_obsrv, test_size=test_size)

    tr_data = numerize(tr_obsrv, user2id, item2id)
    tr_data.to_csv(os.path.join(out_path, "train.csv"), index=False)

    vd_data_tr = numerize(vd_items_tr, user2id, item2id)
    vd_data_tr.to_csv(os.path.join(out_path, "validation_tr.csv"), index=False)

    vd_data_te = numerize(vd_items_te, user2id, item2id)
    vd_data_te.to_csv(os.path.join(out_path,"validation_te.csv"), index=False)

    te_data_tr = numerize(te_items_tr, user2id, item2id)
    te_data_tr.to_csv(os.path.join(out_path,"test_tr.csv"), index=False)

    te_data_te = numerize(te_items_te, user2id, item2id)
    te_data_te.to_csv(os.path.join(out_path,"test_te.csv"), index=False)


if __name__ == "__main__":
    main()
