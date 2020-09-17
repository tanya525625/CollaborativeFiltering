import os
import json

import pandas as pd


def write_dict(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(path):
    with open(path, encoding='utf-8') as fs:
        data = json.load(fs)
    return data


def frequency_analysis_for_simple_entity(data, column_name):
    items = []
    for entity in data:
        if column_name in entity.keys():
            if entity[column_name]:
                items.append(entity[column_name])
            else:
                items.append('Unknown')
        else:
            items.append('No title')

    stats = []
    unique_items = set(items)
    for item in unique_items:
        stats.append([item, items.count(item)])
    stats = sorted(stats, key=lambda item: item[1], reverse=True)
    return pd.DataFrame(data=stats, columns=['title', 'count'])


def category_frequency_analysis(data, column_name):
    united_categories_list = []
    curr_categories = []
    values_count = []

    for entity in data:
        curr_categories.clear()
        for item_info in entity['category']:
            category = item_info[column_name]
            values = item_info['value']
            values_count.append(len(values))
            united_categories_list.append(category)
    categories_stats = {}
    for cat_name, cat_values in zip(united_categories_list, values_count):
        if cat_name not in categories_stats.keys():
            categories_stats.update({cat_name: {}})
        if cat_values not in categories_stats[cat_name].keys():
            categories_stats[cat_name].update({cat_values: 0})
        categories_stats[cat_name][cat_values] += 1

    df = pd.DataFrame(categories_stats).sort_index()
    categories = set(united_categories_list)
    freqs = [united_categories_list.count(category) for category in categories]

    new_row_ind = max(values_count)+1
    df.loc[new_row_ind] = freqs
    return df.rename(index={new_row_ind: 'frequency'})


def main():
    data_dir = "data"
    anywhere_path = os.path.join(data_dir, "anywhere_dataset_2.json")
    stats_path = os.path.join(data_dir, 'anywhere_statistics')
    out_path = os.path.join(stats_path, 'statistics.xlsx')

    data = read_json(anywhere_path)
    column_names = ['assignment_duration', 'city', 'country']
    # f = open(out_path, 'w')
    # f.close()
    with pd.ExcelWriter(out_path, mode='wa', engine='openpyxl') as writer:
        for column in column_names:
            stats = frequency_analysis_for_simple_entity(data, column)
            stats.to_excel(writer, sheet_name=column, engine='openpyxl')

    stats_cat_df = category_frequency_analysis(data, 'international_name')

    # print(stats)


if __name__ == "__main__":
    main()