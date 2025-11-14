import re
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_losses_and_accuracy(log_file_path):
    losses_pattern = re.compile(
        r'loss: (-?\d+\.\d+), intra1 loss: (-?\d+\.\d+), intra2 loss: (-?\d+\.\d+), intra3 loss: (-?\d+\.\d+), coarse loss: (-?\d+\.\d+), fine_loss: (-?\d+\.\d+)'
    )
    accuracy_pattern = re.compile(r'Linear40 Accuracy : (\d+\.\d+)')

    length = []
    loss = []
    intra1_loss = []
    intra2_loss = []
    intra3_loss = []
    coarse_loss = []
    fine_loss = []
    linear40_accuracies = []

    i = 0
    with open(log_file_path, 'r') as file:
        for line in file:
            # 提取损失值
            losses_match = losses_pattern.search(line)
            if losses_match:
                losses_data = list(map(float, losses_match.groups()))
                i += 1
                length.append(i)
                loss.append(losses_data[0]),
                intra1_loss.append(losses_data[1])
                intra2_loss.append(losses_data[2])
                intra3_loss.append(losses_data[3])
                coarse_loss.append(losses_data[4])
                fine_loss.append(losses_data[5])

            # 提取Linear40 Accuracy
            accuracy_match = accuracy_pattern.search(line)
            if accuracy_match:
                linear40_accuracies.append(float(accuracy_match.group(1)))

    return length, loss, intra1_loss, intra2_loss, intra3_loss, coarse_loss, fine_loss, linear40_accuracies


# 假设日志文件路径为 'path_to_your_log_file.log'
log_file_path = './output/pretrain/dgcnn_cls/run.log'

length, loss_list, intra1_loss_list, intra2_loss_list, intra3_loss_list, coarse_loss_list, fine_loss_list, linear40_accuracies_list \
    = extract_losses_and_accuracy(log_file_path)

data = {
    'length': length,
    'loss': loss_list,
    'intra1_loss': intra1_loss_list,
    'intra2_loss': intra2_loss_list,
    'intra3_loss': intra3_loss_list,
    'coarse_loss': coarse_loss_list,
    'fine_loss': fine_loss_list,
    'linear40_accuracy': linear40_accuracies_list
}

df = pd.DataFrame(data)

excel_file_path = './output/visualization/loss/results.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"data saved to {excel_file_path}")
