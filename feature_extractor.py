import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import gc

def feature_extractor(question, answer, model, tokenizer, device):

    '''
        Input:
            question: question sentence. Shape: str
            answer: answer sentence. Shape: str
            model: LLM like Llama-2-7b-hf.
            tokenizer: tokenizer of the model.
            device: model device.
        Output:
            embedding: difference embedding. Shape: [answer_length, hidden_size]
            gradient: gradient of the embedding layer. Shape: [answer_length, hidden_size] 
    '''

    q = question + answer
    a = answer
    q_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(a, return_tensors="pt").input_ids.to(device)
    length_q = len(q_ids[0])
    length_a = len(a_ids[0])
    q_output = model(q_ids, output_hidden_states=True)
    a_output = model(a_ids, output_hidden_states=True)
    q_prob = q_output[0].squeeze(0)[length_q-length_a+1:, :]
    a_prob = a_output[0].squeeze(0)[1:, :]
    q_vector = q_output[2][-1]
    a_vector = a_output[2][-1]
    # kl_divergence = F.kl_div(q_prob.softmax(dim=-1).log(), a_prob.softmax(dim=-1), reduction='sum')
    # kl_loss = nn.KLDivLoss(reduction="sum")
    # kl_divergence = kl_loss(q_prob.softmax(dim=-1).log(), a_prob.softmax(dim=-1))
    kl_divergence = torch.sum(a_prob.softmax(dim=-1) * (a_prob.softmax(dim=-1).log() - torch.log_softmax(q_prob, dim=-1)))
    # The gradient computation step can only be performed on a single GPU; using multiple GPUs will result in an error.
    gradient = torch.autograd.grad(outputs=kl_divergence, inputs=a_vector, create_graph=False)[0].squeeze(0)
    gradient = gradient[1:, :]
    embedding_q = q_vector.squeeze(0)[length_q-length_a+1:, :]
    embedding_a = a_vector.squeeze(0)[1:, :]
    embedding = embedding_q - embedding_a
    return embedding.detach().to("cpu"), gradient.detach().to("cpu")

def batch_extract(data_list, model, tokenizer, device):
    embedding_list = []
    gradient_list = []
    
    for data in tqdm(data_list):
        embedding, gradient = feature_extractor(data['question'], data['answer'],model, tokenizer, device)
        embedding_list.append(embedding)
        gradient_list.append(gradient)
    return embedding_list, gradient_list
 


if __name__ == "__main__":
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
    question = "How are you ?"
    answer = "Fine, thank you."
    embedding, gradient = feature_extractor(question, answer, model, tokenizer, device)
    print(embedding.shape, gradient.shape)
