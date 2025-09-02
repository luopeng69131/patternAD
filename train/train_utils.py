# -*- coding: utf-8 -*-
import torch
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report as cr_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# def get_balance_data(X, y, sample_num_every_class, sample_class):
#     # sample_class = [0, 1, 2, 5, 6, 10, 12, 14]
#     sample_class = sorted(sample_class)

#     balanced_X = []
#     balanced_y = []
#     unique_values, unique_counts = np.unique(y, return_counts=True)
#     for i in range(len(sample_class)):
#         class_label =  sample_class[i]
        
#         assert class_label in unique_values and \
#         unique_counts[class_label] >= sample_num_every_class
        
#         class_indices = np.where(y == class_label)[0]
#         selected_indices = np.random.choice(class_indices, \
#                                             sample_num_every_class, replace=False)
#         balanced_X.extend(X[selected_indices])
#         balanced_y.extend(y[selected_indices])
#         print(X[selected_indices].shape)
#         print(y[selected_indices].shape)
        
    
#     balanced_X, balanced_y = np.array(balanced_X, dtype=float), \
#         np.array(balanced_y, dtype=float)
#     balanced_X = torch.tensor(balanced_X)
#     balanced_y = torch.tensor(balanced_y)
#     return balanced_X, balanced_y

def _report_metrics(y_true, y_score):
    """calculate evaluation metrics"""

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    # print(ratio, thresh)
    
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    print('y_pred shape: ', y_pred.shape)
    # print('y_true shape: ', y_true.shape)
    # print(np.unique(y_pred), np.unique(y_true))
#     p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return classification_report(
        y_true = y_true,
        y_pred = y_pred,
        digits = 4,
        zero_division = 0,
    )


def classification_report(y_true, y_pred, digits, zero_division):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    report = cr_report(        
        y_true = y_true,
        y_pred = y_pred,
        digits = digits,
        zero_division = zero_division
    )
    
    class_0_metirc = [precision[0], recall[0], f1_score[0]]
    class_1_metirc = [precision[1], recall[1], f1_score[1]]
    return report, accuracy, class_0_metirc, class_1_metirc

def construct_dataset(X_normal, y_normal, X_abnormal, y_abnormal):
    X_data = torch.cat((X_normal, X_abnormal), dim=0)
    
    
    y_truth_0 = torch.zeros(y_normal.shape[0], dtype=torch.int64)
    y_truth_1 = torch.ones(y_abnormal.shape[0], dtype=torch.int64)
    y_truth_anomal = torch.cat((y_truth_0, y_truth_1), dim=0)
    
    y_truth_event = torch.cat((y_normal, y_abnormal), dim=0)
    return X_data, y_truth_anomal, y_truth_event

def get_balance_data(X, y, sample_num_every_class, sample_class):
    # sample_class = [0, 1, 2, 5, 6, 10, 12, 14]
    sample_class = sorted(sample_class)

    balanced_X = []
    balanced_y = []
    unique_values, unique_counts = np.unique(y, return_counts=True)
    for i in range(len(sample_class)):
        class_label =  sample_class[i]
        
        assert class_label in unique_values and \
        unique_counts[class_label] >= sample_num_every_class
        
        class_indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(class_indices, \
                                            sample_num_every_class, replace=False)
        # using 'extend' may raise error: sequence error
        balanced_X.append(X[selected_indices])
        balanced_y.extend(y[selected_indices])
        # print(X[selected_indices].shape)
        # print(y[selected_indices].shape)
        
    
    balanced_X = np.concatenate(balanced_X)
    balanced_y = np.array(balanced_y)
    
    balanced_X = torch.tensor(balanced_X)
    balanced_y = torch.tensor(balanced_y)
    return balanced_X, balanced_y

def unbalance_data(X, y, sample_num_every_class, sample_class):
    if isinstance(sample_class, list):
        sample_num = len(sample_class) * sample_num_every_class
    else:# int
        sample_num = sample_class * sample_num_every_class
    train_raito = sample_num /X.shape[0]
    assert train_raito < 1, 'sample %s, X %s' % (sample_num, X.shape[0])
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_raito)
    return torch.tensor(X_train), torch.tensor(y_train)

def unbalance_data_sample_num(X, y, sample_num, is_fix = False):
    # sample_num
    train_raito = sample_num /X.shape[0]
    assert train_raito <= 1
    if is_fix:
        random_state = 42
    else:
        random_state = None
    if train_raito < 1:
        X_train, _, y_train, _ = train_test_split(X, y, \
                                                  train_size=train_raito, \
                                                      random_state=random_state)
    else:
        X_train, y_train = X, y
    return torch.tensor(X_train), torch.tensor(y_train)

    
def get_topk_result(y_pred, y_test):
    # Set prediction to "most likely" prediction
    prediction = y_pred[:, 0]
    # In case correct prediction was in top k, set prediction to correct prediction
    for column in range(1, y_pred.shape[1]):
        # Get mask where prediction in given column is correct
        mask = y_pred[:, column] == y_test
        # Adjust prediction
        prediction[mask] = y_test[mask]
    return prediction

# ========================================================================================
# features
def _normalize(scaler, data, feature_method):
    assert feature_method != '11_label', 'normal of 11 label will generate 0 error'
    data = scaler.transform(data)
    return torch.tensor(data)
    

def feature_engineer(X_10, y_11, method, label=None):
    print('feature: ', method)
    if method == '10_feature':
        X_new = X_10
    elif method == '11_feature':
        X_new = test_11_event_feature(X_10, y_11)
    elif method == '11_label':
        assert label in ['normal', 'abnormal']
        X_new = test_11_label_feature(X_10, y_11, label)
    
    else:
        raise Exception('No feature method exists!')
    print('new feature engineering shape: ', X_new.shape)
    return X_new
    

def test_11_event_feature(X_10, y_11):
    X_11 = torch.cat((X_10, y_11.view(-1,1)), dim=1)
    return X_11

def test_11_label_feature(X_10, y_11, label):
    lable_vec = torch.ones((y_11.shape[0], 1))
    if label == 'normal':
        lable_vec = lable_vec * 0
    X_11 = torch.cat((X_10, lable_vec), dim=1)
    return X_11

def fuze_predict_clf_feature(predict_clf, data_test_X, y_test_event, data_test_Y=None, label='train'):
    data_test_Y_pred_event_11 = torch.tensor(predict_clf.predict(data_test_X[:, :10]))
    data_test_Y_pred_event_12 = torch.tensor(predict_clf.predict(data_test_X[:, 1: 11]))
    predict_11_label = torch.tensor(data_test_Y_pred_event_11) == torch.tensor(y_test_event)
    
    
    acc = (100 * predict_11_label.sum()/predict_11_label.shape[0])
    
    print('Features event %s Acc = %.2f' % (label, acc))
        
    if data_test_Y != None:
    # normal label is '0'
        normal_index = data_test_Y == 0
        predict_11_label_normal = torch.tensor(data_test_Y_pred_event_11)[normal_index] == torch.tensor(y_test_event)[normal_index]
        normal_acc = (100 * predict_11_label_normal.sum()/predict_11_label_normal.shape[0])
        print('Features %s event: normal acc = %.2f' % (label,normal_acc))
        abnormal_index = data_test_Y == 1
        predict_11_label_abnormal = torch.tensor(data_test_Y_pred_event_11)[abnormal_index] == torch.tensor(y_test_event)[abnormal_index]
        abnormal_acc = (predict_11_label_abnormal.sum()/predict_11_label_abnormal.shape[0]) *100
        print('Features %s event: abnormal acc = %.2f' % (label, abnormal_acc))
    
    data_test_X = torch.cat([data_test_X, \
                             data_test_Y_pred_event_12.view(-1,1), \
                             data_test_Y_pred_event_11.view(-1,1), \
                                 predict_11_label.to(torch.float).view(-1,1)], \
                            dim=1)
    return data_test_X