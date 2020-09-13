import os
import json
import pandas as pd


def write_dataset(file_path, dataset):
    with open(file_path, 'w') as json_file:
        for row in dataset:
            json_str = json.dumps(row)
            json_file.write(json_str + '\n')


def read_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]


def main():
    data_dir = "data"
    filepath = os.path.join(data_dir, "united_dataset.json")
    users_dataset_path = os.path.join(data_dir, "kmeans_data.json")
    vacancy_dataset_path = os.path.join(data_dir, "kmeans_vacancies.json")

    users_data = read_dataset(users_dataset_path)
    vacancies_data = read_dataset(vacancy_dataset_path)
    users_data.extend(vacancies_data)
    write_dataset(filepath, users_data)


if __name__=="__main__":
    main()