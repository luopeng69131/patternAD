# -*- coding: utf-8 -*-
import os 

import torch
from torch.utils.data import TensorDataset, DataLoader
'''
short_seq: pad -1 in tail 
output requirements:
name: dataset/%s/filter_torch_%s_map_test_abnormal_v2.pth
format:{
    'test': {'X'=, 'y'=, 'y_event':},
    'abnormal': ...,
    'mapping': ...
    }
X, y, event: torch.tensor


'''


# ---------------------------------------------------------------
# HDFS
basic_path = 'D:\\work\\code\\GPT-security\\data\\dir\\anomaly-detection-log-datasets\\hdfs_logdeep'
# dataset_path = '/home/luopeng/code/deeplog/data/hdfs_train.txt'
# normal_dataset_file = 'hdfs_test_normal'
normal_dataset_file = 'hdfs_test_normal'
abnormal_dataset_file = 'hdfs_test_abnormal'
dataset_name = 'hdfs'

# ---------------------------------------------------
window_size = 10
# ---------------------------------------------------
def generate(file_path):
    inputs = []
    outputs = []
    
    num_sessions = 0
    short_seq = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            if len(line) + 1 >= window_size: #at least 11 if win=10 due to next for
                for i in range(len(line) - window_size):
                    inputs.append(line[i:i + window_size])
                    outputs.append(line[i + window_size])
            else:
                line = line + [-1] * (window_size + 1 - len(line))
                inputs.append(line[: window_size])
                outputs.append(line[window_size])
                
                short_seq += 1
    # undo: 
    print('data %s:\n num line %s, short seq %s' % (file_path, num_sessions, short_seq))
    return torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs, dtype=torch.int64)

# ----------------------------------------------------------------
normal_file_path = os.path.join(basic_path, normal_dataset_file)
abnormal_file_path = os.path.join(basic_path, abnormal_dataset_file)

# --------------------------------------------------

normal_data_X, normal_data_event = generate(normal_file_path)
abnormal_data_X, abnormal_data_event = generate(abnormal_file_path)

normal_y_label = torch.zeros(normal_data_event.shape[0], dtype=torch.int64)
abnormal_y_label = torch.ones(abnormal_data_event.shape[0], dtype=torch.int64)

# mapping = None

dataset_dict_save = {'test': {'X': normal_data_X, 'y': normal_y_label, 'y_event': normal_data_event},
                   'abnormal': {'X': abnormal_data_X, 'y': abnormal_y_label, 'y_event': abnormal_data_event},
                   'mapping': None}


dataset_path_new = 'dataset/%s/filter_torch_%s_map_test_abnormal_v2.pth' %(dataset_name, dataset_name)



torch.save(dataset_dict_save, dataset_path_new)
new_dataset_number = (normal_y_label.shape[0], abnormal_y_label.shape[0])
print('New dataset: normal %s, abnormal %s' % new_dataset_number)



# ----------------------
data_X = torch.cat((normal_data_X, normal_data_event.unsqueeze(1)), dim = 1)
data_Y = torch.cat((abnormal_data_X, abnormal_data_event.unsqueeze(1)), dim = 1)
data_cat = torch.cat((data_X, data_Y))
z = torch.unique(data_cat)
print(z.shape)
print(z)