import os
import json


def write_dataset(dataset, file_path):
    with open(file_path, 'w') as json_file:
        for row in dataset:
            json_str = json.dumps(row)
            json_file.write(json_str + '\n')


def read_json(path):
    with open(path, encoding='utf-8') as fs:
        data = json.load(fs)
    return data


def prepare_dataset_with_labels(data):
    return [row['skills_all'] for row in data]


def prepare_for_train(data, labelled_data):
    new_dataset = []
    skillset = set()
    for data_row, lab_data_row in zip(data, labelled_data):
        print(lab_data_row)
        for skill in data_row['skills']:
            skillset.add(skill['title'])
        for lab_skill in lab_data_row:
            skillset.add(lab_skill['skill'])
        new_dataset.append({"id": data_row["uid"], "skills_dist": list(skillset), "count": len(list(skillset))})
    return new_dataset


def filter_dataset(data, labelled_data, goal_cols):
    new_dataset = []
    skillset = set()
    for data_row, lab_data_row in zip(data, labelled_data):
        skillset.clear()
        for lab_skill in lab_data_row:
            print(lab_skill)
            if lab_skill['label'] in goal_cols:
                skillset.add(lab_skill['skill'])
            for skill in data_row['skills']:
                skillset.add(skill['title'])
        new_dataset.append({"id": data_row["uid"], "skills_dist": list(skillset), "count": len(list(skillset))})
    return new_dataset


def main():
    data_dir = "data"
    anywhere_path = os.path.join(data_dir, "anywhere_with_skills.json")
    out_path = os.path.join(data_dir, "fixed_anywhere.json")

    goal_cols = ['Programming Language', 'Solution', 'Platform', 'Database',
                 'Competency', 'Standard/Technology', 'Framework/Library']

    data = read_json(anywhere_path)
    labelled_dataset = prepare_dataset_with_labels(data)
    filtered_data = filter_dataset(data, labelled_dataset, goal_cols)
    write_dataset(filtered_data, out_path)


if __name__ == "__main__":
    main()
