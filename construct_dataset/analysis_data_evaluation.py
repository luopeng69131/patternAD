import torch
import numpy as np

from train.train_utils import unbalance_data_sample_num

# window_size = 10
dataset_name = 'openstack'
normal_sample_num = None 
# bgl hdfs: 1_000_000

def read_data():

    dataset_path = 'dataset/%s/filter_torch_%s_map_test_abnormal_v2.pth' %(dataset_name, dataset_name)
    
    dataset = torch.load(dataset_path)
    normal_data = dataset['test']
    abnormal_data = dataset['abnormal']
    return normal_data, abnormal_data

def remove_short(dataset):
    X, y, y_event = dataset['X'], dataset['y'], dataset['y_event']
    long_indices = X[:, -1] != -1
    
    X_new = X[long_indices]
    y_new = y[long_indices]
    y_event_new = y_event[long_indices]
    
    dataset_new = {
        'X': X_new,
        'y': y_new,
        'y_event': y_event_new
    }
    print('Short seq num: ', X.shape[0] - long_indices.sum())
    return dataset_new

def count_repeated_samples(dataset):
    # remove repeated samples
    X, y, y_event = dataset['X'], dataset['y'], dataset['y_event']
    
    # 将X和y结合成一个新的tensor，其中y被视为最后一列
    combined = torch.cat((X, y_event.unsqueeze(1)), dim=1)
    
    # 使用unique函数去除重复的行，并返回不重复的行和它们原来的索引
    unique_combined, inverse_indices = combined.unique(dim=0, return_inverse=True)
    
    # 通过索引，我们可以恢复出去重后的X和y
    unique_X = unique_combined[:, :-1]
    unique_y_event = unique_combined[:, -1]
    
    # 如果需要，可以将y转换回原来的数据类型（例如，如果y原来是整型）
    unique_y_event = unique_y_event.long()
    print('After unique data:', unique_y_event.shape)

    return unique_X, unique_y_event


def sample_data(dataset, num):
    X, y, y_event = dataset['X'], dataset['y'], dataset['y_event']
    if num != None and num > 0:
        X_sample, y_event_sample = unbalance_data_sample_num(X, y_event, num)
    else:
        X_sample, y_event_sample = X, y_event
    return X_sample, y_event_sample

# def remove_confilct_data(X_normal, y_normal, y_event_normal, X_abnormal, y_abnormal, y_event_abnormal):
#     # remove normal and abnormal labels conflicts samples 
#     X_normal_cat_event = torch.cat((X_normal, y_event_normal.unsqueeze(1)), dim=1)
#     X_abnormal_cat_event = torch.cat((X_abnormal, y_event_abnormal.unsqueeze(1)), dim=1)
    
#     X = torch.cat((X_normal_cat_event, X_abnormal_cat_event), dim = 0)
#     y = torch.cat((y_normal, y_abnormal), dim = 0)
    
#     def _remove_conflict(X, y):
#         combined = torch.cat((X, y.unsqueeze(1)), dim=1)

#         # 使用字典来存储每个特征对应的标签
#         feature_dict = {}
        
#         for row in combined:
#             features = tuple(row[:-1].tolist())  # 将特征转换成元组
#             label = row[-1].item()
        
#             if features in feature_dict:
#                 feature_dict[features].add(label)
#             else:
#                 feature_dict[features] = {label}
        
#         # 过滤掉冲突样本
#         filtered_data = [row for row in combined if len(feature_dict[tuple(row[:-1].tolist())]) == 1]
        
#         # 转换回 X 和 y
#         filtered_data = torch.stack(filtered_data)
#         X_filtered = filtered_data[:, :-1]
#         y_filtered = filtered_data[:, -1]
#         return X_filtered, y_filtered
    
#     X, y = _remove_conflict(X, y)
    
#     X_normal_cat_new, y_normal_new = X[y==0], y[y==0]
#     X_abnormal_cat_new, y_abnormal_new = X[y==1], y[y==1]
    
#     print('remove conflicts: normal %s, abnormal %s' % (y_normal_new.shape, y_abnormal_new.shape))
#     return X_normal_cat_new, y_normal_new,\
#         X_abnormal_cat_new, y_abnormal_new 
        
def remove_confilct_data_count(X_normal, y_normal, y_event_normal, X_abnormal, y_abnormal, y_event_abnormal):
    # remove normal and abnormal labels conflicts samples 
    X_normal_cat_event = torch.cat((X_normal, y_event_normal.unsqueeze(1)), dim=1)
    X_abnormal_cat_event = torch.cat((X_abnormal, y_event_abnormal.unsqueeze(1)), dim=1)
    
    X = torch.cat((X_normal_cat_event, X_abnormal_cat_event), dim = 0)
    y = torch.cat((y_normal, y_abnormal), dim = 0)
    
    def _remove_conflict(X, y):
        combined = torch.cat((X, y.unsqueeze(1)), dim=1)

        # 使用字典来存储每个特征对应的标签
        feature_dict = {}
        
        for row in combined:
            features = tuple(row[:-1].tolist())  # 将特征转换成元组
            label = row[-1].item()
        
            if features in feature_dict:
                feature_dict[features].add(label)
            else:
                feature_dict[features] = {label}
        
        # 过滤掉冲突样本
        # filtered_data = [row for row in combined if len(feature_dict[tuple(row[:-1].tolist())]) == 1]
        filtered_data = []
        num_normal_conflict, num_abnormal_conflict = 0, 0
        for i, row in enumerate(combined):
            row_list = row.tolist()
            if len(feature_dict[tuple(row_list[:-1])]) == 1:
                filtered_data.append(row)
            else:
                # in conflicts
                if row_list[-1] == 0:
                    # normal: keep
                    filtered_data.append(row)
                    
                    num_normal_conflict += 1
                else:
                    num_abnormal_conflict += 1
        
        print('remove conflicts: num_normal_conflict %s, num_abnormal_conflict %s' % \
              (num_normal_conflict, num_abnormal_conflict))
        # 转换回 X 和 y
        filtered_data = torch.stack(filtered_data)
        X_filtered = filtered_data[:, :-1]
        y_filtered = filtered_data[:, -1]
        return X_filtered, y_filtered
    
    X, y = _remove_conflict(X, y)
    
    X_normal_cat_new, y_normal_new = X[y==0], y[y==0]
    X_abnormal_cat_new, y_abnormal_new = X[y==1], y[y==1]
    
    print('remove conflicts: normal %s, abnormal %s' % (y_normal_new.shape, y_abnormal_new.shape))
    return X_normal_cat_new, y_normal_new,\
        X_abnormal_cat_new, y_abnormal_new 

# --------------------------------------------------------------------------------------
normal_dataset, abnormal_dataset = read_data()
print('Normal samples: %s, abnormal samples: %s' % (normal_dataset['y'].shape[0], abnormal_dataset['y'].shape[0]))

normal_dataset = remove_short(normal_dataset)
abnormal_dataset = remove_short(abnormal_dataset)
print('Remove short | Normal samples: %s, abnormal samples: %s' % (normal_dataset['y'].shape[0], abnormal_dataset['y'].shape[0]))


X_normal, y_event_normal = sample_data(normal_dataset, normal_sample_num)
X_abnormal, y_event_abnormal = abnormal_dataset['X'], abnormal_dataset['y_event']
print('Sample dara | Normal samples: %s, abnormal samples: %s' % (X_normal.shape, X_abnormal.shape))


y_normal = torch.zeros(y_event_normal.shape[0], dtype=torch.int64)
y_abnormal = torch.ones(y_event_abnormal.shape[0], dtype=torch.int64)

X_event_normal, y_normal, X_event_abnormal, y_abnormal = remove_confilct_data_count(X_normal, y_normal, y_event_normal, X_abnormal, y_abnormal, y_event_abnormal)

new_dataset = {
    'normal':{'X': X_event_normal, 'y': y_normal},
    'abnormal':{'X': X_event_abnormal, 'y': y_abnormal},
    }

torch.save(new_dataset, 'dataset/test/eval_benchmark_%s.pth' % dataset_name)


# test in supervised training----------------------------------------------------------------------------------------------------------------
from sklearn.utils import shuffle
# 假设 X_event_normal, y_normal, X_event_abnormal, y_abnormal 已经定义且是NumPy数组
train_samples = 1_000


# 打乱数据
X_event_normal, y_normal = shuffle(X_event_normal, y_normal, random_state=0)
X_event_abnormal, y_abnormal = shuffle(X_event_abnormal, y_abnormal, random_state=0)

# 选择每个类别的前1000个样本
X_train_normal = X_event_normal[:train_samples]
y_train_normal = y_normal[:train_samples]

X_train_abnormal = X_event_abnormal[:train_samples]
y_train_abnormal = y_abnormal[:train_samples]

# 合并正常和异常样本以形成训练集
X_train = np.concatenate((X_train_normal, X_train_abnormal), axis=0)
y_train = np.concatenate((y_train_normal, y_train_abnormal), axis=0)

X_test = np.concatenate((X_event_normal, X_event_abnormal), axis=0)
y_test = np.concatenate((y_normal, y_abnormal), axis=0)

# 再次打乱合并后的训练集
X_train, y_train = shuffle(X_train, y_train, random_state=0)

# ----------------------------------------------------------------------------
from sklearn.metrics import classification_report
from sklearn.tree          import DecisionTreeClassifier
import lightgbm as lgb
from sklearn import ensemble

# tree_clf = DecisionTreeClassifier()
# tree_clf = ensemble.RandomForestClassifier()
tree_clf = lgb.LGBMClassifier(objective='binary')
# [:, :-1]
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

print(classification_report(
    y_true = y_test,
    y_pred = y_pred,
    digits = 4,
    zero_division = 0,
))
    



    
