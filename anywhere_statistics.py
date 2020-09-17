import os
import json


def write_dict(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    data_dir = "data"
    anywhere_path = os.path.join(data_dir, "anywhere_dataset_2.json")
    titles_stats_path = os.path.join(data_dir, "titles_stats.json")

    with open(anywhere_path, 'r', encoding='utf-8') as fs:
        file_list = fs.readlines()

    keywords = []
    for row in file_list:
        if '"' in row and '":' in row:
            keywords.append(row.split('"')[1])
    unique_keywords = set(keywords)
    keywords_stats = {}
    for keyw in unique_keywords:
        keywords_stats.update({keyw: keywords.count(keyw)})
    keywords_stats = {k: v for k, v in sorted(keywords_stats.items(), key=lambda item: item[1], reverse=True)}
    write_dict(keywords_stats, titles_stats_path)

    vac_count = keywords_stats['country']
    print(vac_count)


if __name__ == "__main__":
    main()