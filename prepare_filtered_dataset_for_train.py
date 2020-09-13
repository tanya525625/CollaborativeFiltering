import os
import json
import pandas as pd


def write_dataset(file_path, dataset):
    with open(file_path, 'w') as json_file:
        for row in dataset:
            json_str = json.dumps(row)
            json_file.write(json_str + '\n')


def prepare_dataset(df):
    res_dicts = []
    for i, row in df.iterrows():
        skills = row[1]
        if skills:
            res_dicts.append({"id": i, "skills_dist": skills, "count": len(skills)})
    return res_dicts


def make_lists(data):
    return data.replace(']', '').replace('[', '').replace("'", '').split(', ')


def main():
    data_dir = "data"
    dataset_path = os.path.join(data_dir, "filtered_dataset.csv")
    output_path = os.path.join(data_dir, "filtered_dataset.json")

    dataset = pd.read_csv(dataset_path)
    dataset = dataset.drop('Unnamed: 0', axis=1)
    dataset['Skills'] = dataset['Skills'].apply(make_lists)
    res_dicts = prepare_dataset(dataset)
    write_dataset(output_path, res_dicts)


if __name__ == "__main__":
    main()