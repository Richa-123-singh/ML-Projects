import csv
import math
import copy
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np


class DecisionTree:
    def count_each_label(self,labels_list):
        counts = {item: labels_list.count(item) for item in labels_list}
        return counts

    def labels(self,dataset):
        return [data[-1] for data in dataset]

    def entropy(self,dataset):
        labels_list = self.labels(dataset)        
        counts = self.count_each_label(labels_list)
        total_possibilities = len(labels_list)
        entropy = 0
        for key in counts:
            p = counts[key]/total_possibilities
            entropy -= (p*(math.log2(p)))
        return entropy

    def entropy_gini(self,dataset):
        labels_list = self.labels(dataset)
        counts = self.count_each_label(labels_list)
        total_possibilities = len(labels_list)
        entropy = 1
        for key in counts:
            p = counts[key]/total_possibilities
            entropy -= (p**2)
        return entropy

    def information_gain(self,parent_dataset, index_to_split):
        split_ds = self.split_the_dataset(parent_dataset, index_to_split)
        len_of_dataset = len(parent_dataset)
        parent_entropy = self.entropy(parent_dataset)
        weighted_sum_entropy = 0
        weighted_sum_entropy = sum((len(split_ds[key]) / len_of_dataset) * self.entropy(split_ds[key]) for key in split_ds)
        return parent_entropy - weighted_sum_entropy
        
    def gini(self,parent_dataset, index_to_split):
        split_ds = self.split_the_dataset(dataset=parent_dataset, label_index_to_split_with=index_to_split)
        len_of_dataset = len(parent_dataset)
        parent_entropy = self.entropy_gini(parent_dataset)
        weighted_sum_entropy = 0
        weighted_sum_entropy = sum((len(split_ds[key]) / len_of_dataset) * self.entropy_gini(split_ds[key]) for key in split_ds)
        return parent_entropy - weighted_sum_entropy

    def high_information_gain_feature(self,dataset):
        try:
            highest_index = None
            highest_info_gain = 0
            for i in range(len(dataset[0])-1):
                i_g = self.information_gain(parent_dataset=dataset, index_to_split=i)
                if i_g >= highest_info_gain:
                    highest_index = i
                    highest_info_gain = i_g
            return highest_index
        except:
            return None
        
    def high_gini_gain_feature(self,dataset):
        try:
            highest_index = None
            highest_info_gain = 0
            for i in range(len(dataset[0])-1):
                i_g = self.gini(parent_dataset=dataset, index_to_split=i)
                if i_g >= highest_info_gain:
                    highest_index = i
                    highest_info_gain = i_g
            return(highest_index)
        except:
            return None
    
    def split_the_dataset(self,dataset, label_index_to_split_with):
        split = {}
        for data in dataset:
            key = data[label_index_to_split_with]
            if key not in split:
                split[key] = []
            split[key].append(data)
        return split

    def check_85p(self,count_dict):
        tot = count_dict['0'] + count_dict['1']
        _85p = (85 / 100) * tot
        return count_dict['0'] > _85p or count_dict['1'] > _85p

    def build_tree(self,parent_node, max_label=None, method = 'entropy', earlyStop = False):
        ##### STOPPING CRITERIA #####
        if not parent_node.dataset:
            parent_node.label = max_label
            return
        else:
            labels_list = self.labels(dataset=parent_node.dataset)
            _counts_dict = self.count_each_label(labels_list=labels_list)
            if len(_counts_dict)<=1: # dataset is homogenous
                    parent_node.label = next(iter(_counts_dict.keys()))
                    return
            elif earlyStop and self.check_85p(_counts_dict):
                parent_node.label = max(_counts_dict, key=_counts_dict.get)
            else:
                    if method == 'entropy':
                        _high_info_feature_index = self.high_information_gain_feature(dataset=parent_node.dataset)
                    elif method == 'gini':
                        _high_info_feature_index = self.high_gini_gain_feature(dataset=parent_node.dataset) #with gini
                    else:
                        print("Incorrect Method Selected")
                        return None
                    _feature = headers[_high_info_feature_index]
                    parent_node.feature = _feature


                    if not parent_node.attributes_set:
                            labels_list = self.labels(dataset=parent_node.dataset)
                            _counts_dict = self.count_each_label(labels_list=labels_list)
                            parent_node.label = max(_counts_dict, key=_counts_dict.get)
                            return
                    
                    child_left_node = Node(dataset=[], label=None, feature=None, left=[], middle=[], right=[], attributes_set={})
                    child_middle_node = Node(dataset=[], label=None, feature=None, left=[], middle=[], right=[], attributes_set={})
                    child_right_node = Node(dataset=[], label=None, feature=None, left=[], middle=[], right=[], attributes_set={})               
                    child_attributes_set = set()

                    for k in parent_node.attributes_set:
                        if k != _feature:
                            child_attributes_set.add(k)

                    d = self.split_the_dataset(dataset=parent_node.dataset, label_index_to_split_with=_high_info_feature_index)
                    for key in d:
                        if key == '0':
                            child_left_node.dataset = d[key]
                            child_left_node.attributes_set = child_attributes_set
                        elif key == '1':
                            child_middle_node.dataset = d[key]
                            child_middle_node.attributes_set = child_attributes_set
                        elif key == '2':
                            child_right_node.dataset = d[key]
                            child_right_node.attributes_set = child_attributes_set

                    parent_node.left = child_left_node
                    parent_node.middle = child_middle_node
                    parent_node.right = child_right_node
                    self.build_tree(parent_node=child_left_node, max_label=max_label,earlyStop=earlyStop)
                    self.build_tree(parent_node=child_middle_node, max_label=max_label,earlyStop=earlyStop)
                    self.build_tree(parent_node=child_right_node, max_label=max_label,earlyStop=earlyStop)
        
    def predict(self, root_node, X, actual_headers):
        _feature_of_node = root_node.feature

        if _feature_of_node:
            _index = actual_headers.index(_feature_of_node)
            if X[_index] == str(0):
                return self.predict(root_node.left, X, actual_headers)
            elif X[_index] == str(1):
                return self.predict(root_node.middle, X, actual_headers)
            elif X[_index] == str(2):
                return self.predict(root_node.right, X, actual_headers)
        else:
             return root_node.label 

        
    def prediction_for_dataset(self,root_node, dataset,actual_headers):
        t_t_prediction = []
        for t_ins in dataset:
            p = self.predict(root_node, t_ins, actual_headers)
            t_t_prediction.append(1 if t_ins[-1] == p else 0)
        return np.mean(t_t_prediction)

class Node:
    def __init__(self, dataset=[], label=None, feature=None, left=[], middle=[], right=[], attributes_set={}):
        self.dataset = dataset
        self.label = label
        self.feature = feature
        self.left = left
        self.middle = middle
        self.right = right
        self.attributes_set = attributes_set

    def get_label(self):
        return (self.label)

    def __repr__(self) -> str:
        return(f"{self.feature}-{self.label}")


if __name__ == "__main__":
    data = [] 
    filename='house_votes_84.csv'
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    headers = data[0]
    headers = [x.replace("\ufeff","") for x in headers]
    dataset = data[1:]
    train_accuracies = []
    test_accuracies = []
    for _ in range(100):
        # (a) Shuffling the dataset
        dTree = DecisionTree()
        rng = np.random.default_rng()
        rng.shuffle(dataset)
        per_80 = int(len(dataset)* 0.8)

        # (b) partitioning the dataset 
        training_dataset = dataset[:per_80]
        testing_dataset = dataset[per_80:]
        
        headers_set = set(copy.deepcopy(headers))
        p_n = Node(dataset=training_dataset, attributes_set=headers_set)
        # p_n = Node(dataset=dataset, attributes_set=headers_set) # Use this to run for entire dataset
        
        # max_label calculation for entire dataset
        labels_list = dTree.labels(dataset=training_dataset)
        _counts_dict = dTree.count_each_label(labels_list=labels_list)
        max_label = -1
        max_val = 0
        for key in _counts_dict:
            if _counts_dict[key] >= max_val:
                max_val = _counts_dict[key]
                max_label = key
        
        #dTree.build_tree(parent_node=p_n, max_label=max_label)
        #dTree.build_tree(parent_node=p_n, max_label=max_label, method='gini') # for gini
        dTree.build_tree(parent_node=p_n, max_label=max_label,earlyStop=True) # for EarlyStop
    
        # dTree.print_tree(node=p_n, level=0)
        
        # please toggle the below two lines to run for testing and training data respectively
        train_accuracies.append(dTree.prediction_for_dataset(root_node=p_n, dataset=training_dataset, actual_headers=headers))
        test_accuracies.append(dTree.prediction_for_dataset(root_node=p_n, dataset=testing_dataset, actual_headers=headers))
        
    mean = np.mean(train_accuracies)
    std = np.std(train_accuracies)
    print("Train Mean", mean)
    print("Train SD", std)
    plt.hist(train_accuracies)
    plt.xlabel('accuracy')
    plt.ylabel('Frequency')
    #plt.show()
    plt.savefig("Train_Accuracies.png")
    plt.close()

    mean = np.mean(test_accuracies)
    std = np.std(test_accuracies)
    print("Test Mean", mean)
    print("Test SD", std)
    plt.hist(test_accuracies)
    plt.xlabel('accuracy')
    plt.ylabel('Frequency')
    #plt.show()
    plt.savefig("Test_Accuracies.png")
    plt.close()
    