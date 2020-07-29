import json


def make_full_list(users):
    full_list = []
    for user in users:
        new_list = []
        for skill in list(user["skills_dist"]):
            for char in ['"', "\\", "]", "["]:
                skill = skill.replace(char, '')
            new_list.append(skill)
        full_list += new_list
    return full_list


def make_statistics(full_list):
    unique_skills = frozenset(full_list)
    print(len(unique_skills))
    skills_dict = dict.fromkeys(unique_skills)
    for skill in skills_dict.keys():
        skills_dict[skill] = full_list.count(skill)

    return dict(sorted(skills_dict.items(), reverse=True, key=lambda kv: kv[1]))


if __name__ == "__main__":
    with open('data.json', 'r') as file:
        data = file.readlines()

    users = [json.loads(user) for user in data]
    full_list = make_full_list(users)
    skills_dict = make_statistics(full_list)

    with open('statistics.json', 'w') as f:
        json.dump(skills_dict, f)
