from utils import json_loader
from HalluDataset import HallDataset, hallu_collate_fn
from HalluDetector import HallModel
from feature_extractor import batch_extract
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
# import torch_npu
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def train(model, criterion, optimizer, dataloader):
    total_loss = 0
    for _, batch in enumerate(dataloader):
        optimizer.zero_grad()
        embedding, gradient, label = batch
        label = label.float()
        out = model(embedding, gradient).squeeze()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss
def eval(model, dataloader):
    total_label = []
    total_pred = []
    total_out = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            embedding, gradient, label = batch
            out = model(embedding, gradient).squeeze()
            total_out += out.tolist()
            total_label += label.tolist()
            pred = list(map(lambda x: round(x), out.tolist()))
            total_pred += pred
        f1 = f1_score(total_label, total_pred)
        acc = accuracy_score(total_label, total_pred)
        precision, recall, _ = precision_recall_curve(total_label, total_pred)
        pr_auc = auc(recall, precision)
    return acc, f1, pr_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--task_type", type=str, default="qa")
    parser.add_argument("--model_name", type=str, default="meta/Llama-2-7b-hf")
    parser.add_argument("--train_ratio", type=float, default=0.1)
    parser.add_argument("--lambda", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    data_list = json_loader(args.data_dir, args.task_type)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    embedding_list, gradient_list = batch_extract(data_list, model, tokenizer, device)
    label = [data['label'] for data in data_list]

    dataset = HallDataset(embedding_list, gradient_list, label)
    train_size = int(args.train_ratio* len(dataset))
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=hallu_collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=hallu_collate_fn, shuffle=True)
    
    hallu_detector = HallModel(4096, 2048, 1)
    max_iter = 20
    criterion = nn.BCELoss()
    optim = torch.optim.Adam(hallu_detector.parameters(), lr=1e-3)
    best_acc = 0
    best_f1 = 0
    for epoch in range(max_iter):
        total_loss = train(hallu_detector, criterion, optim, train_dataloader)
        print(f'Epoch [{epoch+1}/{max_iter}], Loss: {total_loss/2000:.4f}')
        acc, f1, pr_auc = eval(hallu_detector, test_dataloader)
        print(f'Epoch [{epoch+1}/{max_iter}], ACC: {acc:.4f}, F1: {f1:.4f}, PR-AUC:{pr_auc:.4f}')
        if acc > best_acc:
            best_acc = acc
        if f1 > best_f1:
            best_f1 = f1
        if total_loss < 1:
            break
    print(f'eval accuracy: {best_acc:.4f}. F1: {best_f1:.4f}')
    print("done!")