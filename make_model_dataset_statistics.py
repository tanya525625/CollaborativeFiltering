import os
import json

import numpy as np
import pandas as pd


def write_dict(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def make_statistics(items):
    unique_items = set(items)
    items_count = [items.count(item) for item in unique_items]
    statistics = dict(zip(unique_items, items_count))
    sorted_statistics = dict(sorted(statistics.items(), reverse=True, key=lambda kv: kv[1]))
    sorted_statistics.update({"unique_items_count": len(unique_items)})
    sorted_statistics.update({"all_count": sum(items_count)})
    return sorted_statistics


def save_files(df_seria, filepath):
    res = []
    res.clear()
    for lst in df_seria:
        lst = lst.replace('[', '').replace(']', '').replace('\'', '').split(',')
        for i, el in enumerate(lst):
            if el:
                if el[0] == ' ':
                    lst[i] = el[1:]
        res.extend(lst)
    np.save(filepath, res)


def main():
    data_dir = "data"
    st_path = os.path.join(data_dir, "model_dataset_statistics")
    is_save = False
    input_path = os.path.join(data_dir, "filtered_dataset.csv")
    if is_save:
        df = pd.read_csv(input_path)
        save_files(df['Position'].tolist(), os.path.join(data_dir, "all_labels.npy"))
        save_files(df['Skills'].tolist(), os.path.join(data_dir, "all_items.npy"))
    else:
        labels = np.load(os.path.join(data_dir, "all_labels.npy"), allow_pickle=True).tolist()
        items = np.load(os.path.join(data_dir, "all_items.npy"), allow_pickle=True).tolist()
        if '' in labels:
            labels.remove('')
        items_st = make_statistics(items)
        write_dict(items_st, os.path.join(st_path, "skills_statistics.json"))
        labels_st = make_statistics(labels)
        write_dict(labels_st, os.path.join(st_path, "labels_statistics.json"))


if __name__ == "__main__":
    main()