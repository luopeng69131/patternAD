import os
import argparse
import torch

import numpy as np

import json
from datetime import datetime
import pandas as pd
def get_args():
    parser = argparse.ArgumentParser(description='deeplog')
    parser.add_argument(
        '--clf',
        default='',
        help='set model name:ref var clf_list'
    )
    parser.add_argument(
        '--dataset',
        default='',
        help='set dataset'
    )
    parser.add_argument(
        '--test-num',
        default='',
        help='set dataset'
    )
    args = parser.parse_args()
    
    args.clf = args.clf.split(',')
    return args

def validate_clf(clf_list, input_clfs):
    if len(input_clfs):
        for clf in input_clfs:
            if clf not in clf_list:
                raise Exception('input clf error: %s' % clf)
    else:
        raise Exception('empty clfs: %s' % input_clfs)

def shuffle_data(X, y):
    random_indices = np.random.permutation(X.shape[0])
    new_X, new_Y = X[random_indices, :], y[random_indices]
    return new_X, new_Y
            
def get_temp_file():
    # 获取当前日期和时间
    current_time = datetime.now()
    
    # 使用strftime()方法格式化时间
    formatted_time = current_time.strftime("%m-%d_%H_%M_%S")
    
    rela_path = 'data/temp/'
    
    # file = '%s.txt' % formatted_time
    if not os.path.exists(rela_path):
        os.makedirs(rela_path)
        
    # file_abs_path = os.path.join(rela_path, file)
    return formatted_time, rela_path


def map2event(y, map_relation):
    
    mapping_func_to_clf = lambda label: map_relation[label]
    y = y.apply_(mapping_func_to_clf)
    
    return y

def get_dataset(path):
    # return : tensor, array
    dataset = torch.load(path)
    # load real event label
    X = dataset['X']
    y = dataset['y'].numpy()
    unique_values, unique_counts = np.unique(y, return_counts=True)
    print('dataset path %s: %s' % (path, X.shape))
    print('class number %s' % (unique_values))
    print('counts %s' % unique_counts)
    print('---------------------------------')
    return X, y 

def save_dict_data(dir_path, create_time, dataset_name, *json_data, **json_name_data):
    file_name = '%s_' % dataset_name + '%s_' % create_time + '.txt'
    file_abs_path = os.path.join(dir_path, file_name)
    
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将NumPy数组转换为列表
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # 将NumPy数组转换为列表
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    # 将字符串写入同一个文本文件中
    with open(file_abs_path, 'w') as file:
        for d in json_data:
            file.write("\n\nDictionary :\n")
            file.write(json.dumps(d, indent=4, default=default))
            
    with open(file_abs_path, 'w') as file:
        for k, v in json_name_data.items():
            file.write("\n\n Name: %s\n" % k)
            file.write(json.dumps(v, indent=4, default=default))

def save_excel_reports(report_list, dir_path, create_time, test_index, dataset_name):
    file_excel_testnum = create_time + '_%s_' % dataset_name + '_num-%s.xlsx' % test_index
    
    file_abs_path = os.path.join(dir_path, file_excel_testnum)
    with pd.ExcelWriter(file_abs_path, engine='xlsxwriter') as writer:
        for name, report_str in report_list.items():
            print('clf save:', name)
            report_lines = report_str.strip().split('\n')
            data = [line.split() for line in report_lines[2:]]
    
            df = pd.DataFrame(data, columns=[' ', 'class', 'precision', 'recall', 'f1-score', 'support'])
            
            # 将DataFrame保存为Excel文件
            df.to_excel(writer, sheet_name=name, index=False)
            
def save_excel_reports_testnum(report_list_testnum, dir_path, create_time, test_num, dataset_name):
    file_excel_testnum = create_time + '_%s_' % dataset_name + '_num-%s.xlsx' % test_num
    
    file_abs_path = os.path.join(dir_path, file_excel_testnum)
    
    start_rows_clf = {}
    with pd.ExcelWriter(file_abs_path, engine='xlsxwriter') as writer:
        for test_index, report_list in report_list_testnum.items():
            for name, report_str in report_list.items():
                print('clf save:', name)
                report_lines = report_str.strip().split('\n')
                data = [line.split() for line in report_lines[2:]]
        
                df = pd.DataFrame(data, columns=[' ', 'class', 'precision', 'recall', 'f1-score', 'support'])
                
                if name not in start_rows_clf:
                    start_rows_clf[name] = 0
                
                start_rows = start_rows_clf[name] 
                # 将DataFrame保存为Excel文件
                df.to_excel(writer, sheet_name=name, index=False, startrow=start_rows)
                # +1 for title, +3 for space
                start_rows_clf[name] = start_rows + len(data) + 1 + 3 
                
        
    