import numpy as np
import torch
from sklearn.tree          import DecisionTreeClassifier
import lightgbm as lgb

from utils import (\
    get_args,
    save_dict_data,
    get_temp_file
)
    
from train.train_utils import (
    classification_report
    )
    
window_size = 11
# num_noise_element = 2
# -------------------------------------------
def shuffle_data(X, y):
    random_indices = np.random.permutation(X.shape[0])
    new_X, new_Y = X[random_indices, :], y[random_indices]
    return new_X, new_Y

def process_matrix_rows(matrix):
    X, y, sample_index = [], [], []
    for row_index, row in enumerate(matrix):
        # print(' ', row)
        for i, element in enumerate(row):
            # print(row)
            x = np.copy(row)
            x[i] = -1
            
            X.append(x)
            y.append(element)
            sample_index.append(row_index)
    
    # return X, y, sample_index
    return np.array(X), np.array(y), np.array(sample_index)

def generate_pattern_feature(clf, X_data, y_data):
    cols = X_data.shape[-1]
    y_pred = clf.predict(X_data)
    
    pattern_mask = (y_pred == y_data).reshape(-1, cols)
    return pattern_mask

def create_pattern_feature(dataset, pattern_clf):
    X_event_normal, y_normal_label_0 = dataset['normal']['X'], dataset['normal']['y']
    X_event_abnormal, y_abnormal_label_1 = dataset['abnormal']['X'], dataset['abnormal']['y']

    X, y, _ = process_matrix_rows(X_event_normal)
    X_abnormal, y_abnormal, _ = process_matrix_rows(X_event_abnormal)
    
    classifier = pattern_clf

    X_normal_pattern = generate_pattern_feature(classifier, X, y)
    X_abnormal_pattern = generate_pattern_feature(classifier, X_abnormal, y_abnormal)

    X_normal_cat_pattern_feature = np.concatenate((X_event_normal, X_normal_pattern), axis=1)
    X_abnormal_cat_pattern_feature = np.concatenate((X_event_abnormal, X_abnormal_pattern), axis=1)
    return X_normal_cat_pattern_feature, y_normal_label_0, \
        X_abnormal_cat_pattern_feature, y_abnormal_label_1

def create_pattern_feature_X(X_data, pattern_clf):
    # X_data,  = dataset['normal']['X'], dataset['normal']['y']
    # X_event_abnormal, y_abnormal_label_1 = dataset['abnormal']['X'], dataset['abnormal']['y']

    X, y, _ = process_matrix_rows(X_data)
    # X_abnormal, y_abnormal, _ = process_matrix_rows(X_event_abnormal)
    
    classifier = pattern_clf

    X_pattern = generate_pattern_feature(classifier, X, y)
    # X_abnormal_pattern = generate_pattern_feature(classifier, X_abnormal, y_abnormal)

    X_data_cat_pattern_feature = np.concatenate((X_data, X_pattern), axis=1)
    # X_abnormal_cat_pattern_feature = np.concatenate((X_event_abnormal, X_abnormal_pattern), axis=1)
    return X_data_cat_pattern_feature

# --------------------------------------------

def count_repeated_samples(combined):
    # remove repeated samples
    # X, y, y_event = dataset['X'], dataset['y'], dataset['y_event']
    
    # # 将X和y结合成一个新的tensor，其中y被视为最后一列
    # combined = torch.cat((X, y_event.unsqueeze(1)), dim=1)
    
    # 使用unique函数去除重复的行，并返回不重复的行和它们原来的索引
    unique_combined, inverse_indices = combined.unique(dim=0, return_inverse=True)
    
    # 通过索引，我们可以恢复出去重后的X和y
    # unique_X = unique_combined[:, :-1]
    # unique_y_event = unique_combined[:, -1]
    
    # 如果需要，可以将y转换回原来的数据类型（例如，如果y原来是整型）
    # unique_y_event = unique_y_event.long()
    print('After unique data:', unique_combined.shape[0])

    return unique_combined

def load_clean_data(normal_sample_size, dataset_path):
    dir_path = dataset_path
    # dir_path = 'dataset/%s/eval_benchmark_%s.pth' % (dataset_name, dataset_name)
    dataset = torch.load(dir_path)
    print('Clean loaded | Normal %s, abnormal %s' % (dataset['normal']['X'].shape[0], dataset['abnormal']['X'].shape[0]))  
    # dataset['normal']['X'] =  dataset['normal']['X'][:, :10]
    # dataset['abnormal']['X'] = dataset['abnormal']['X'][:, :10]
    
    normal_num = dataset['normal']['X'].shape[0]
    if normal_sample_size != None and  normal_sample_size < normal_num:
        normal_sample_indices = np.random.choice(normal_num, normal_sample_size, replace=False)
        dataset['normal']['X'] = dataset['normal']['X'][normal_sample_indices, :]
        dataset['normal']['y'] = dataset['normal']['y'][normal_sample_indices]
    print('Clean sample | Normal %s' % (dataset['normal']['X'].shape[0])) 
    dataset['sample_normal'] = {}
    dataset['sample_normal']['X'] = dataset['normal']['X']
    dataset['sample_normal']['y'] = dataset['normal']['y']
    
    # func count_repeated_samples will copy a new var; 
    # so 'sample_normal' will be different with 'normal'
    dataset['normal']['X'] = count_repeated_samples(dataset['normal']['X'])
    dataset['abnormal']['X'] = count_repeated_samples(dataset['abnormal']['X'])
    
    dataset['normal']['y'] = dataset['normal']['y'][:dataset['normal']['X'].shape[0]]
    dataset['abnormal']['y'] = dataset['abnormal']['y'][:dataset['abnormal']['X'].shape[0]]
    
    # shuffle
    dataset['normal']['X'], dataset['normal']['y'] = \
        shuffle_data(dataset['normal']['X'], dataset['normal']['y'])
    dataset['abnormal']['X'], dataset['abnormal']['y'] = \
        shuffle_data(dataset['abnormal']['X'], dataset['abnormal']['y'])
    
    print('Remove repeated| Normal %s, abnormal %s' % (dataset['normal']['X'].shape[0], dataset['abnormal']['X'].shape[0]))
    print('pattern sample normal| Normal %s' % (dataset['sample_normal']['X'].shape[0]))
    return dataset

def load_pattern_model(dataset):
    X_event_normal, y_normal_label_0 = dataset['normal']['X'], dataset['normal']['y']

    random_indices = np.random.permutation(X_event_normal.shape[0])
    X_event_normal, y_normal_label_0 = X_event_normal[random_indices, :], y_normal_label_0[random_indices]

    X, y, sample_index = process_matrix_rows(X_event_normal)
    print('Train pattern dataset X: ', X.shape)

    cart = DecisionTreeClassifier()
    classifier = cart
    # classifier = lgb.LGBMClassifier(objective='multiclass')

    sample_num = X_event_normal.shape[0]
    cols = X.shape[-1]
    X_small, y_small = X[:sample_num*cols, :], y[:sample_num*cols]
    # X_small, y_small = 

    classifier.fit(X_small, y_small)
    acc = classifier.score(X, y)
    print('pattern dataset train data: ', acc)
    return classifier


def replace_random_elements(matrix, n):
    down_bound, upper_bound = int(matrix.min()), int(matrix.max())
    
    rows, cols = matrix.shape
    assert n <= cols
    for i in range(rows):
        # 生成n个随机位置
        random_positions = np.random.choice(cols, n, replace=False)
        # 在这些位置上生成随机数并替换原来的值
        matrix[i, random_positions] = np.random.randint(down_bound, upper_bound, size=n)
    return matrix

def create_detect_dataset_noise(pattern_clf, clean_dataset, num_noise_element):
    X_normal_cat_pattern_feature, y_normal_label_0, \
        X_abnormal_cat_pattern_feature, y_abnormal_label_1 =\
            create_pattern_feature(clean_dataset, pattern_clf)
        
    print('Cat pattern feature shape: %s, %s' % (X_normal_cat_pattern_feature.shape, \
                                                  X_abnormal_cat_pattern_feature.shape))
    train_samples = 1_000
    # sample data from origin dataset and use it as normal train data
    X_normal_origin_sample, y_normal_origin_sample = clean_dataset['sample_normal']['X'][:train_samples],\
        clean_dataset['sample_normal']['y'][:train_samples]
    X_normal_cat_pattern_feature_origin = create_pattern_feature_X(X_normal_origin_sample, pattern_clf)
    X_train_normal = X_normal_cat_pattern_feature_origin
    y_train_normal = y_normal_origin_sample   
    
    
    X_noise = replace_random_elements(X_normal_origin_sample.clone().numpy(), num_noise_element)
    # X_abnormal_origin_sample, y_abnormal_origin_sample = orginal_dataset['abnormal']['X'][:train_samples],\
    #     orginal_dataset['abnormal']['y'][:train_samples]
    X_noise_cat_pattern_feature_origin = create_pattern_feature_X(X_noise, pattern_clf)
    X_train_abnormal = X_noise_cat_pattern_feature_origin
    y_train_abnormal = np.ones(X_noise.shape[0], dtype=int)
    
    X_train = np.concatenate((X_train_normal, X_train_abnormal), axis=0)
    y_train = np.concatenate((y_train_normal, y_train_abnormal), axis=0)
    
    X_test = np.concatenate((X_normal_cat_pattern_feature, X_abnormal_cat_pattern_feature), axis=0)
    y_test = np.concatenate((y_normal_label_0, y_abnormal_label_1), axis=0)
    return (X_train, y_train), (X_test, y_test)

def train_detector(detect_dataset):
    X_train, y_train, X_test, y_test = detect_dataset
    
    tree_clf = lgb.LGBMClassifier(objective='binary')
    tree_clf.fit(X_train, y_train)
    print('train acc: %.2f' %  tree_clf.score(X_train, y_train))
    print('clean dataset report: ')
    y_pred = tree_clf.predict(X_test)
    print(classification_report(
        y_true = y_test,
        y_pred = y_pred,
        digits = 4,
        zero_division = 0,
    )[0])
    print('----------------------------')
    return tree_clf
    
def read_original_data(dataset_path):
    # dataset_path = 'dataset/%s/eval_benchmark_%s.pth' % (dataset_name, dataset_name)
    # dataset_path = dataset_path
    dataset = torch.load(dataset_path)
    
    # dataset['normal'] = dataset.pop('test')
    
    dataset['normal']['X'], dataset['normal']['y'] = \
        shuffle_data(dataset['normal']['X'], dataset['normal']['y'])
    dataset['abnormal']['X'], dataset['abnormal']['y'] = \
        shuffle_data(dataset['abnormal']['X'], dataset['abnormal']['y'])
    return dataset

def sample_data(dataset, normal_sample_size=None, abnormal_sample_size=None):
    normal_num = dataset['normal']['X'].shape[0]
    abnormal_num = dataset['abnormal']['X'].shape[0]
    print('Normal num %s | abnormal num %s' % (normal_num, abnormal_num))
    
    # normal_sample_size, abnormal_sample_size = 100_000, 50_000
    
    new_dataset = {'normal': dict(), 'abnormal':dict()}
    
    new_dataset['normal']['X'], new_dataset['normal']['y']  = dataset['normal']['X'], dataset['normal']['y']
    if normal_sample_size != None and  normal_sample_size < normal_num:
        normal_sample_indices = np.random.choice(normal_num, normal_sample_size, replace=False)
        new_dataset['normal']['X'] = dataset['normal']['X'][normal_sample_indices, :]
        new_dataset['normal']['y'] = dataset['normal']['y'][normal_sample_indices]
        
    new_dataset['abnormal']['X'], new_dataset['abnormal']['y']  = dataset['abnormal']['X'], dataset['abnormal']['y']
    if abnormal_sample_size != None and abnormal_sample_size < abnormal_num:
        abnormal_sample_indices = np.random.choice(abnormal_num, abnormal_sample_size, replace=False)
        new_dataset['abnormal']['X'] = dataset['abnormal']['X'][abnormal_sample_indices, :]
        new_dataset['abnormal']['y'] = dataset['abnormal']['y'][abnormal_sample_indices]
    return new_dataset
    

def pattern_feature_model_fit(pattern_clf, original_dataset):
    # dataset = original_dataset
    # X_normal_cat_pattern_feature, y_normal_label_0 = dataset['normal']['X'], dataset['normal']['y']
    # X_abnormal_cat_pattern_feature, y_abnormal_label_1 = dataset['abnormal']['X'], dataset['abnormal']['y']
    
    X_normal_cat_pattern_feature, y_normal_label_0, \
        X_abnormal_cat_pattern_feature, y_abnormal_label_1 =\
            create_pattern_feature(original_dataset, pattern_clf)

    X_test = np.concatenate((X_normal_cat_pattern_feature, X_abnormal_cat_pattern_feature), axis=0)
    y_test = np.concatenate((y_normal_label_0, y_abnormal_label_1), axis=0)
    
    # X_test = X_test[:, :window_size]
    # pattern feaure
    # X_test = X_test[:, -window_size:]
    
    print('Test dataset shape', X_test.shape)
    return X_test, y_test


if __name__ == '__main__':
    args = get_args() 
    print('dataset: ', args.dataset)
    test_num = 1
    if args.test_num != '':
        test_num = int(args.test_num)
    print('test: ', test_num)
    
    dataset_config = {
        'hdfs': {'train_sample': 8_000},
        'bgl': {'train_sample': 32_000},
        'hadoop': {'train_sample': 8_000},
        }
    assert args.dataset in dataset_config
    dataset_name = args.dataset
    
    # dataset_name = 'hdfs'
    normal_sample_size = dataset_config[dataset_name]['train_sample']
    print('dataset %s, clearn normal sample size %s' % (dataset_name, normal_sample_size))
    
    dataset_path = 'dataset/%s/eval_benchmark_%s.pth' % (dataset_name, dataset_name)
    clean_dataset = load_clean_data(normal_sample_size, dataset_path)
    original_dataset = read_original_data(dataset_path)
    
    pattern_feature_model = load_pattern_model(clean_dataset)
    # get test dataset: sample_original_dataset for quick test
    sample_original_dataset = sample_data(original_dataset)
    fuze_feature, label = pattern_feature_model_fit\
        (pattern_feature_model, sample_original_dataset)
    print('fuze X shape: ', fuze_feature.shape)
    
    print('prepare data ok!')
    
    # -----------------------
    method_results = {}
    ave_res_data = {}
    for num_noise in range(1,6):
        method_results[num_noise] = {}
        ave_res_data[num_noise] = {}
        for mode in ['seq', 'pattern', 'fuze']:
            method_results[num_noise][mode] = {
                    'class 0': [],
                    'class 1': [],
                    'acc': []
                }
            ave_res_data[num_noise][mode] = {
                    'class 0': [],
                    'class 1': [],
                    'acc': []
                }
    for z in range(test_num):
        print('&&&&&&&&&&&&  Test num: %s  &&&&&&&&&&&&' % z)
        for num_noise in range(1,6):
            print("********* noise num: %s ********" % num_noise)
            (X_train, y_train), (X_test, y_test) = create_detect_dataset_noise\
                (pattern_feature_model, clean_dataset, num_noise)
    
            # -------------------------------------------------------
            for mode in ['seq', 'pattern', 'fuze']:
                print('=========== mode %s ============' % mode)
                if mode == 'seq':
                    X_train_new = X_train[:, :window_size]
                    X_test_new = X_test[:, :window_size]
                    fuze_feature_new = fuze_feature[:, :window_size]
                elif mode == 'pattern':
                    X_train_new = X_train[:, window_size:]
                    X_test_new = X_test[:, window_size:]
                    fuze_feature_new = fuze_feature[:, window_size:]
                else:
                    X_train_new = X_train
                    X_test_new = X_test
                    fuze_feature_new = fuze_feature
                    
                clf_model = train_detector([X_train_new, y_train, X_test_new, y_test])
                y_pred = clf_model.predict(fuze_feature_new)
                report, accuracy, class_0_metirc, class_1_metirc = classification_report(
                    y_true = label,
                    y_pred = y_pred,
                    digits = 4,
                    zero_division = 0,
                )
                
                method_results[num_noise][mode]['class 0'].append(class_0_metirc)
                method_results[num_noise][mode]['class 1'].append(class_1_metirc)
                method_results[num_noise][mode]['acc'].append(accuracy)
                
                print('Res | test %s, noise %s, mode %s ' % (z, num_noise, mode))
                print('Acc: ', accuracy)
                print('Class 0:', class_0_metirc)
                print('Class 1:', class_1_metirc)
                print('Report: ')
                print(report)
                print('-----------------')
            
    print('=========  Average results =========================')

    for num_noise in range(1,6):
        for mode in ['seq', 'pattern', 'fuze']:
            class_0_res = np.array(method_results[num_noise][mode]['class 0'])
            class_1_res = np.array(method_results[num_noise][mode]['class 1'])
            acc_res = np.array(method_results[num_noise][mode]['acc'])
            
            ave_class_0_res = class_0_res.sum(0) / test_num
            ave_class_1_res = class_1_res.sum(0) / test_num
            ave_acc_res = acc_res.sum(0) / test_num
            
            ave_res_data[num_noise][mode] = {
                    'class 0': ave_class_0_res,
                    'class 1': ave_class_1_res,
                    'acc': ave_acc_res
                }
            
            print('Aver| noise %s, mode %s, result %s' % (num_noise, mode, ave_res_data[num_noise][mode]))
    create_time, dir_path = get_temp_file()
    save_dict_data(dir_path, create_time, dataset_name, average=ave_res_data, original=method_results) 
                
                # print(classification_report(
                #     y_true = label,
                #     y_pred = y_pred,
                #     digits = 4,
                #     zero_division = 0,
                # ))




