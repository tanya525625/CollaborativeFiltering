import os
import json

import pandas as pd
from bs4 import BeautifulSoup


def write_dict(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(path):
    with open(path, encoding='utf-8') as fs:
        data = json.load(fs)
    return data


def parse_html(content):
    soup = BeautifulSoup(content, 'lxml')
    for child in soup.recursiveChildGenerator():
        if child.name:
            for tag in soup.find_all(child.name):
                html_text = tag.text
                if not html_text:
                    return 'empty'
                else:
                    return 'non empty'


def not_null_frequency(data, column_name, neg_word='null', pos_word='non empty'):
    items = []
    for entity in data:
        if column_name in entity.keys():
            item = entity[column_name]
            if item:
                if '>' in item:
                    html_res = parse_html(item)
                    items.append(html_res)
                else:
                    items.append(pos_word)
            else:
                items.append(neg_word)
        else:
            items.append('no title')
    return make_stats_df(items)


def make_stats_df(items):
    stats = []
    unique_items = set(items)
    for item in unique_items:
        stats.append([item, items.count(item)])
    stats = sorted(stats, key=lambda item: item[1], reverse=True)
    return pd.DataFrame(data=stats, columns=['title', 'count'])


def frequency_analysis_for_simple_entity(data, column_name, column_name_2=None):
    items = []
    content = []
    for entity in data:
        if column_name in entity.keys():
            if column_name_2:
                content.clear()
                for item in entity[column_name]:
                    content.append(item[column_name_2])
            else:
                content = entity[column_name]
            if content:
                if isinstance(content, list):
                    items.extend(content)
                else:
                    items.append(content)
            else:
                if isinstance(content, bool):
                    items.append(content)
                else:
                    items.append('null')
        else:
            items.append('no title')
    return make_stats_df(items)


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
    new_row_ind = max(values_count) + 1
    df.loc[new_row_ind] = freqs
    return df.rename(index={new_row_ind: 'frequency'})


def write_results(data, out_path):
    column_names = ('assignment_duration', 'city', 'country', 'job_type', 'name',
                    'offer', 'remote_office', 'specialization', 'tags', 'type', 'jobFunctions')
    columns_for_not_null = ('description', 'is_hot', 'is_hidden', 'about_project')
    columns_pares = (('job_specialization', '_in_progress'), ('job_specialization', 'title'), ('skills', 'title'))

    with pd.ExcelWriter(out_path, mode='wa', engine='openpyxl') as writer:
        stats_cat_df = category_frequency_analysis(data, 'international_name')
        stats_cat_df.to_excel(writer, sheet_name='international_name', engine='openpyxl')
        for column in column_names:
            stats = frequency_analysis_for_simple_entity(data, column)
            stats.to_excel(writer, sheet_name=column, engine='openpyxl')
        for column in columns_for_not_null:
            if 'is' in column:
                pos_word = True
                neg_word = False
            else:
                pos_word = 'non empty'
                neg_word = 'empty'
            stats = not_null_frequency(data, column, neg_word=neg_word, pos_word=pos_word)
            stats.to_excel(writer, sheet_name=column, engine='openpyxl')
        for pare in columns_pares:
            stats = frequency_analysis_for_simple_entity(data, pare[0], pare[1])
            stats.to_excel(writer, sheet_name=f'{pare[0]}.{pare[1]}', engine='openpyxl')


def main():
    data_dir = "data"
    anywhere_path = os.path.join(data_dir, "anywhere_dataset_2.json")
    stats_path = os.path.join(data_dir, 'anywhere_statistics')
    out_path = os.path.join(stats_path, 'statistics.xlsx')

    data = read_json(anywhere_path)
    write_results(data, out_path)


if __name__ == "__main__":
    main()