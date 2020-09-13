import os
import json
from copy import deepcopy
from collections import OrderedDict

import pandas as pd


def write_dict(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def sort_dict(data):
    stats = OrderedDict(sorted(data.items(), key=lambda x: x[1]['count'], reverse=True))
    for position in data.keys():
        stats[position]["skills_stats"] = OrderedDict(sorted(data[position]["skills_stats"].items(),
                                                             key=lambda x: x[1], reverse=True))
    return stats


def filter_df(df, goal_cols, skills_drop_list):
    positions = []
    skills = []
    indexes = []
    stats = {}
    pos_with_one_skills = 0
    template = {"count": 0, "skills_stats": {}}
    for ind, row in df.iterrows():
        labels = row[0].replace('[', '').replace(']', '').replace('\'', '').replace('`', '').split(', ')
        ents = row[1].replace('[', '').replace(']', '').replace('\'', '').replace('`', '').split(', ')
        row_dict = dict(zip(labels, ents))
        if 'Position' in labels:
            position = row_dict.pop('Position')
        else:
            position = 'unknown'
        filtered_dict = {k: v for k, v in row_dict.items() if k in goal_cols}
        curr_skills = list(filtered_dict.values())
        if filtered_dict.values():
            for drop_skill in skills_drop_list:
                if drop_skill in curr_skills:
                    curr_skills.remove(drop_skill)
            skills.append(curr_skills)
            positions.append(position)
            indexes.append(ind)
            if position not in stats.keys():
                stats.update({position: deepcopy(template)})
            for skill in curr_skills:
                if skill not in stats[position]["skills_stats"].keys():
                    stats[position]["skills_stats"].update({skill: 0})
                stats[position]["skills_stats"][skill] += 1
            stats[position]['count'] += 1
            if len(curr_skills) == 1:
                pos_with_one_skills += 1
    stats = sort_dict(stats)
    return pd.DataFrame({'Indexes': indexes, 'Skills': skills, 'Position': positions}), stats


def main():
    data_dir = "data"
    st_path = os.path.join(data_dir, "model_dataset_statistics")
    input_path = os.path.join(data_dir, "parsed_model_dataset.csv")
    goal_cols = ['Programming Language', 'Solution', 'Platform', 'Database',
                 'Competency', 'Standard/Technology', 'Framework/Library']
    skills_drop_list = ['programming languages']
    df = pd.read_csv(input_path)
    filtered_df, stats = filter_df(df, goal_cols, skills_drop_list)
    filtered_df.to_csv(os.path.join(data_dir, "filtered_dataset.csv"))
    write_dict(stats, os.path.join(st_path, "filtered_dataset_st.json"))


if __name__ == "__main__":
    main()