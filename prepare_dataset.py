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


if __name__ == "__main__":
    with open('data.json', 'r') as file:
        data = file.readlines()

    users = [json.loads(user) for user in data]
    full_list = make_full_list(users)
    unique_skills = set(full_list)
    new_set = set()
    final_set = set()

    for skill in unique_skills:
        new_set.add(skill.lower())

    for low_skill in new_set:
        for skill in unique_skills:
            if skill.lower() == low_skill:
                big_skill = skill
                break
        final_set.add(big_skill)

    new_user_list = []
    new_user_low_list = []
    for user in users:
        new_user_list.clear()
        new_user_low_list.clear()
        for skill in list(user["skills_dist"]):
            for fixed_skill in final_set:
                if skill.lower() == fixed_skill.lower():
                    if fixed_skill.lower() not in new_user_low_list:
                        new_user_list.append(fixed_skill)
                    new_user_low_list.append(fixed_skill.lower())

        new_user_list = list(set(new_user_list))
        user["skills_dist"] = new_user_list

    with open('fixed_data.json', 'w') as f:
        json.dump(users, f)

