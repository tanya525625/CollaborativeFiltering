import os
import pandas as pd
import re


def find_enties_in_row(df):
    items = []
    labels = []
    curr_items = []
    сurr_labels = []
    for row in data.index.values:
        curr_items.clear()
        сurr_labels.clear()
        for el in row:
            if isinstance(el, str):
                el = el.replace('"', "").replace('}', "").replace(']', '').replace('[', '')
                if re.search(r'entity*:', el):
                    curr_items.append(el.split(': ')[1])
                if re.search(r'label*:', el):
                    сurr_labels.append(el.split(': ')[1])
        if сurr_labels:
            entities, items_labels = make_unique_lists(сurr_labels, curr_items)
            items.append(entities)
            labels.append(items_labels)
        else:
            print(row)

    return pd.DataFrame(data={'labels': labels, 'entities': items})


def make_unique_lists(labels, entities):
    low_ents = [ent.lower() for ent in entities]
    unique_items = list(set(low_ents))
    unique_items_indexes = [unique_items.index(ent) for ent in unique_items]
    items_labels = [labels[ind] for ind in unique_items_indexes]
    return unique_items, items_labels


if __name__ == "__main__":
    file_path = os.path.join("data", "model_dataset.csv")
    out_file_path = os.path.join("data", "parsed_model_dataset.csv")
    data = pd.read_csv(file_path, quoting=3, error_bad_lines=False, sep=',')
    res_df = find_enties_in_row(data)
    res_df.to_csv(out_file_path, index=False)
    df = pd.read_csv(out_file_path)