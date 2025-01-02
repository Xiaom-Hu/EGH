import torch
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc


def json_loader(file_name, task_type):
    f = open(file_name)
    lines = f.readlines()
    data_list = []
    for line in lines:
        data = json.loads(line)
        if task_type == "qa":
            question = data['knowledge'] + ' ' + data['question']
            # right_answer = data['right_answer']
            right_answer = {
                'question': question,
                'answer': data['right_answer'], 
                'label': 1
            }
            hallucinate_answer = {
                'question': question,
                'answer': data['hallucinated_answer'],
                'label': 0,
            }
            data_list.append(right_answer)
            data_list.append(hallucinate_answer)
        if task_type == "summary":
            question = data["document"]
            right_answer = {
                'question': question,
                'answer': data['right_summary'],
                'label': 1
            }
            hallucinate_answer = {
                'question': question,
                'answer': data['hallucinated_summary'],
                'label': 0
            }
            data_list.append(right_answer)
            data_list.append(hallucinate_answer)
        if task_type == "dialogue":
            question = data["knowledge"] + ' ' + data["dialogue_history"]
            right_answer = {
                'question': question,
                'answer': data['right_response'],
                'label': 1
            }
            hallucinate_answer = {
                'question': question,
                'answer': data['hallucinated_response'],
                'label': 0
            }
            data_list.append(right_answer)
            data_list.append(hallucinate_answer)
        if task_type == "general":
            question = data['user_query']
            label_map = {
                "no" : 0,
                "yes": 1
            }
            answer = {
                'question': question,
                'answer': data["chatgpt_response"],
                'label': label_map[data['hallucination']]
            }
            data_list.append(answer)
    return data_list

