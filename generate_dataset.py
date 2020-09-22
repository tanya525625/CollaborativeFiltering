import os
import json
import random
from tqdm import tqdm


def determine_skills_count(skills_min_count, skills_max_count, quantity):
    if skills_max_count > quantity:
        skills_max_count = quantity
    if skills_min_count > skills_max_count:
        skills_min_count = skills_max_count
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
    is_added = False
    for i in tqdm(range(users_count)):
        # if i > users_count - users_count // 80 and not is_added:
        #     skills_dict['python-dev'].append('knime')
        #     is_added = True
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
        "python-dev": ["python3", "python", "pandas", "scipy", "Scikit-learn", "numpy", "torch", "flair", "TensorFlow", "Keras"],
        "python-tester": ["python3", "Selenium", "python", "Pytest", "Testrail",
                          "unittest", "coverage", "DocTest", "Testify", "Robot"],
        "web": ["HTML", "CSS", "JavaScript", "python3", "python", "PHP", "Bootstrap", "AJAX", "jQuery", "Django", "Flask"],
        "java-dev": ["Spring", "Blade", "Java", "Vaadin", "Dropwizard",
                     "Grails", "MyBatis", "JHipster", "JSF", "Google Web Toolkit"],
        "ruby": ["Ruby", "Sinatra", "Ruby on Rails", "Merb", "Hanami",
                 "Padrino", "NYNY", "Scorched", "Crepe", "Nancy"],
        "C#": ["C#", ".NET", "ASP.NET", ".NET Core", ".NET Framework",
               "Microsoft.CodeAnalysis.CSharp"],
        "DevOps": ["Ansible", "Terraform", "AWS", "Jenkins", "TeamCity", "Linux", "bash", "ssh"]
     }

    vacancies = {
        "python-dev": ["python2", "python", "pandas", "Scikit-learn", "torch", "TensorFlow", "Keras"],
        "python-tester": ["Selenium", "python", "Pytest", "Testrail",
                          "unittest", "coverage", "DocTest", "Testify", "Robot"],
        "web": ["HTML", "CSS", "JavaScript", "python", "PHP", "Bootstrap", "AJAX", "jQuery", "Django", "Flask"],
        "java-dev": ["Spring", "Blade", "Java", "Vaadin", "Dropwizard", "Kafka",
                     "Grails", "MyBatis", "JHipster", "JSF", "Google Web Toolkit"],
        "ruby": ["Ruby", "Sinatra", "Ruby on Rails", "Merb",
                 "Padrino", "NYNY", "Scorched", "Cuba"],
        "C#": ["C#", ".NET", "ASP.NET", ".NET Core", ".NET Framework",
               "Unity", "Newtonsoft.Json"],
        "DevOps": ["Ansible", "Terraform", "Jenkins", "TeamCity", "Linux"]
    }
    data_dir = "data"
    # with open(os.path.join(data_dir, "KMEANS_cluster_dict_for_gen.json")) as json_data:
    #     skills_dict = json.load(json_data)
    # vacancies = skills_dict
    out_path = os.path.join(data_dir)
    users_count = 20000
    vacancies_count = 10000
    skills_min_count = 8
    skills_max_count = 15
    skills_vacancies_min = 10
    skills_vacancies_max = 15

    common_skills = ["agile", "git", "RabbitMQ", "scrum", "sql",
                     "mysql", "windows", "docker", "jira", "gitlab"]

    common_vacancies_skills = ["agile", "git", "scrum", "sql",
                               "mysql", "windows", "docker", "jira", "gitlab"]

    # common_skills, common_vacancies_skills = [], []

    print("Added items")
    dataset = generate_dataset(users_count, skills_min_count, skills_max_count, skills_dict, common_skills)
    vacancies_dataset = generate_dataset(vacancies_count, skills_vacancies_min, skills_vacancies_max,
                                         vacancies, common_vacancies_skills)
    write_dataset(os.path.join(data_dir, "generated_employees.json"), dataset)
    write_dataset(os.path.join(data_dir, "generated_vacancies.json"), vacancies_dataset)

