import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import pdb

def sample_gumbel(shape, eps=1e-20):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    U = torch.empty(shape, device='cpu').uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
import pandas as pd
from pathlib import Path
# import csv
# import json
# # 1. 读取 CSV 文件
# csv_data = []
# with open('./data/infant/meta_withbirth.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         csv_data.append(row)


# # 2. 读取 JSON 文件（假设是 JSON 数组）
# with open('./data/infant/infant_metadata.json', mode='r', encoding='utf-8') as json_file:
#     json_list = json.load(json_file)  # 如果是单个 JSON 对象，改成 [json.load(json_file)]

# # 3. 构建 {run_accession: sample_alias} 字典
# json_dict = {item["run_accession"]: item["sample_alias"] for item in json_list}

# # 4. 替换 Env 列
# for row in csv_data:
#     sample_id = row["SampleID"]
#     if sample_id in json_dict:
#         # pdb.set_trace()
#         row["Env"] = f"{json_dict[sample_id]}({row['Env'].split('(')[1]}"

# # 5. 保存到新的 CSV 文件
# with open('output.csv', mode='w', encoding='utf-8', newline='') as csv_file:
#     fieldnames = ['SampleID', 'Env']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(csv_data)

# print("✅ 替换完成，结果已保存到 output.csv")
