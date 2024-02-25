import pandas as pd

import requests

def load_json_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

data = load_json_from_url('https://raw.githubusercontent.com/qiangning/TORQUE-dataset/main/data/train.json')

triplets_with_category = []

for item in data:
    for passage in item["passages"]:
        context = passage["passage"]
        for qa_pair in passage["question_answer_pairs"]:
            question = qa_pair["question"]
            answer = ", ".join(qa_pair["answer"]["spans"])  # Joining answer spans if there are multiple
            question_id = qa_pair["question_id"]
            category = int(question_id.split('-')[0])  # Extracting category from the 'question_id'
            triplets_with_category.append((context, question, answer, category))

# Convert the list of triplets into a DataFrame
df_with_category = pd.DataFrame(triplets_with_category, columns=["Context", "Question", "Answer", "Category"])

print(len(df_with_category))

selected_rows = pd.concat([
    df_with_category[df_with_category['Category'] == 0].sample(n=min(1000, len(df_with_category[df_with_category['Category'] == 0])), random_state=1),
    df_with_category[df_with_category['Category'] == 1].sample(n=min(1000, len(df_with_category[df_with_category['Category'] == 1])), random_state=1),
    df_with_category[df_with_category['Category'] == 2].sample(n=min(1000, len(df_with_category[df_with_category['Category'] == 2])), random_state=1)
])

selected_rows.reset_index(drop=True, inplace=True)

print(len(selected_rows))

csv_file_path = 'C:/Users/shell/Downloads/temporal_qa_data.csv'

selected_rows.to_csv(csv_file_path, index=False)