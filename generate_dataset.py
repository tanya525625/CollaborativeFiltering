import os
import json
import random
from tqdm import tqdm


def determine_skills_count(skills_min_count, skills_max_count, quantity):
    if skills_max_count > quantity:
        skills_max_count = quantity
    return random.randint(skills_min_count, skills_max_count)


def write_dataset(file_path, dataset):
    with open(file_path, 'w') as json_file:
        for row in dataset:
            json_str = json.dumps(row)
            json_file.write(json_str + '\n')


def generate_dataset(users_count, skills_min_count, skills_max_count, skills_dict, common_skills):
    professions_list = list(skills_dict.keys())
    professions_count = len(professions_list)
    user_skills = []
    skills = []
    rows = []
    print("Added users")
    for i in tqdm(range(users_count)):
        profession = professions_list[random.randint(0, professions_count - 1)]
        skills.clear()
        skills = skills_dict[profession] + common_skills
        skills_count = determine_skills_count(skills_min_count, skills_max_count, len(skills))
        user_skills.clear()
        added_skills_count = 0
        while added_skills_count < skills_count:
            new_skill = skills[random.randint(0, skills_count - 1)]
            if new_skill not in user_skills:
                user_skills.append(new_skill)
                added_skills_count += 1
        rows.append({"id": i, "skills_dist": user_skills.copy(), "count": len(user_skills)})

    return rows


if __name__ == "__main__":
    skills_dict = {
        "python-dev": ["python", "pandas", "scipy", "Scikit-learn", "numpy", "torch", "flair", "TensorFlow", "Keras"],
        "python-tester": ["Selenium", "Python", "Pytest", "TeamCity", "Testrail"],
        "kotlin-tester": ["Kotlin", "Kotest", "Selenium", "TeamCity", "Testrail"],
        "web": ["HTML", "CSS", "JavaScript", "Python", "PHP", "Bootstrap", "AJAX", "jQuery", "Django", "Flask"],
        "java-dev": ["Spring ", "Blade", "Java", "Vaadin", "Dropwizard", "Grails", "MyBatis", "JHipster", "JSF", "Google Web Toolkit"],
        "ruby": ["Ruby", "Sinatra", "Ruby on Rails", "Merb", "Hanami"],
        "C#": ["C#", ".NET", "ASP.NET"]
     }
    out_path = "data"
    users_count = 20000
    skills_min_count = 5
    skills_max_count = 20
    common_skills = ["agile", "git", "scrum", "sql", "mysql", "linux", "windows", "docker", "jira", "gitlab", "PostgreSQL"]

    dataset = generate_dataset(users_count, skills_min_count, skills_max_count, skills_dict, common_skills)
    write_dataset(os.path.join(out_path, "generated_data.json"), dataset)

