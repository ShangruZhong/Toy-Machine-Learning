# -*- coding: utf-8 -*-
"""
    Decision Tree with Entropy Gain (ID3)
    feature slection criteron: information gain
    @author: Shangru
"""
import pandas as pd
import numpy as np

def entropy(labels):
    """
    @params: labels, [1, 0, 1, 0, ...]
    @return: entropy of labels
    """
    n = len(labels)
    if not n:
        return 0
    label_counts = {}
    for cur_label in labels:
        if cur_label not in label_counts:
            label_counts[cur_label] = 1
        else:
            label_counts[cur_label] += 1
    entropy = 0.0
    for key in label_counts:
        p = float(label_counts[key]) / n
        entropy += -p * np.log2(p)
    return entropy

def best_feature(train):
    """
    @params: training dataset
    @return: infomation gain, best_feature
    """
    y = train['label']
    num_sample = len(y)
    entro = entropy(y)

    features = [col for col in train.columns if col != 'label']
    min_condi_entro = 100
    best_feat = str()

    for feat in features:
        feat_vals = list(train[feat].value_counts().index) # vals of current feature, [1, 2, 3]
        subset = {}

        for feat_val in feat_vals:
            subset[feat_val] = list(train[train[feat] == feat_val]['label'])

        condi_entro = 0.0
        for feat_val in feat_vals:
            sub_labels = subset[feat_val] # sub_labels: [1, 0, 1, 1,...]
            condi_entro += 1.0 * len(sub_labels) / num_sample * entropy(sub_labels)
        if condi_entro < min_condi_entro:
            min_condi_entro = condi_entro
            best_feat = feat

    return entro-min_condi_entro, best_feat

def split_dataset(train, feature, feature_value):
    return train[train[feature] == feature_value].drop(feature, axis=1, inplace=False)

def create_tree(train):
    """
    @params: training dataset
    @return: root node of tree
    """
    if len(train['label'].value_counts().index) == 1: # only have 1 label, return it
        return train['label'].iloc[0]
    if len(train.columns) == 1: # no features, return most frequent label
        return train['label'].value_counts().index[0]
    info_gain, best_feat = best_feature(train)
    print best_feat
    if info_gain < 0.001:
        return train['label'].value_counts().index[0]
    feat_vals = list(train[best_feat].value_counts().index)  # vals of current feature, [1, 2, 3]
    tree = {best_feat:{}}
    for feat_val in feat_vals:
        tree[best_feat][feat_val] = create_tree(split_dataset(train, best_feat, feat_val))
    return tree

if __name__ == "__main__":
    # test case in Hang Li's book (pp 62)
    data = np.array([[1, 0, 0, 1, 0],
                     [1, 0, 0, 2, 0],
                     [1, 1, 0, 2, 1],
                     [1, 1, 1, 1, 1],
                     [1, 0, 0, 1, 0],
                     [2, 0, 0, 1, 0],
                     [2, 0, 0, 2, 0],
                     [2, 1, 1, 2, 1],
                     [2, 0, 1, 3, 1],
                     [2, 0, 1, 3, 1],
                     [3, 0, 1, 3, 1],
                     [3, 0, 1, 2, 1],
                     [3, 1, 0, 2, 1],
                     [3, 1, 0, 3, 1],
                     [3, 0, 0, 1, 0],
                     ])
    train = pd.DataFrame(data=data, columns=['age_level', 'has_job', 'has_house', 'load_level', 'label'])
    print create_tree(train)
