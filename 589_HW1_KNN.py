import csv
import math
import copy

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

def load_dataset(filename):
    dataset = [] 
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                dataset.append(row)
        return dataset
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []


def calculate_euclidean_distance(instance1, instance2):
    try:
        if len(instance1) != len(instance2):
            raise ValueError("Unequal length of instances")
        
        distance_sum = sum((float(instance1[i]) - float(instance2[i]))**2 for i in range(len(instance1)))
        return math.sqrt(distance_sum)
    
    except ValueError as e:
        print(f"Error: {e}")
        return -1

def extract_column_values(column_no, dataset):
    column_values = []
    try:
        for row in dataset:
            column_values.append(float(row[column_no]))
        return column_values
    
    except ValueError:
        print(f"Error: Failed to convert values in column {column_no} to float.")
        return []


def feature_normalisation(input_data):
    normalized_data = copy.deepcopy(input_data)
    for feature_index in range(len(input_data[0]) - 1):
        column_values = extract_column_values(feature_index, input_data)
        max_value = max(column_values)
        min_value = min(column_values)
        
        for instance_index in range(len(normalized_data)):
            normalized_data[instance_index][feature_index] = (float(normalized_data[instance_index][feature_index]) - min_value) / (max_value - min_value)
    
    return normalized_data


def count_each_label(labels_list):
    type_labels = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
    type_counts = {}
    for i in type_labels:
        type_counts[i] = labels_list.count(i)
    return(type_counts)


def KNN(k, dataset, X):
    distances_list = []
    for i in range(len(dataset)):
        distances_list.append(calculate_euclidean_distance(dataset[i][:-1], X))
    sorted_list = sorted(enumerate(distances_list), key=lambda x: x[1])
    nearest_labels = []
    for i in sorted_list[:k]:
        nearest_labels.append(dataset[i[0]][-1])
    return count_each_label(nearest_labels)


def accuracy(counts_dic, actual_label):
    most_predicted = max(counts_dic.values())
    for item, count in counts_dic.items():
        if count == most_predicted and item == actual_label:
            return 1
    else:
        return 0
    

def predict_on_training_dataset(given_k, training_dataset):
    total_predictions = []
    for instance_index in range(len(training_dataset)):
        current_instance = training_dataset[instance_index][:-1]
        knn_counts = KNN(given_k, training_dataset, current_instance)
        prediction_accuracy = accuracy(knn_counts, actual_label=training_dataset[instance_index][-1])
        total_predictions.append(prediction_accuracy)
    
    total_accuracy = sum(total_predictions)
    accuracy_percentage = total_accuracy / len(training_dataset) * 100
    return accuracy_percentage



def predict_on_testing_dataset(given_k, training_dataset, testing_dataset): 
    total_predictions = []
    for instance_index in range(len(testing_dataset)):
        current_instance = testing_dataset[instance_index][:-1]
        knn_counts = KNN(given_k, training_dataset, current_instance)
        prediction_accuracy = accuracy(knn_counts, actual_label=testing_dataset[instance_index][-1])
        total_predictions.append(prediction_accuracy)
    
    total_accuracy = sum(total_predictions)
    accuracy_percentage = total_accuracy / len(testing_dataset) * 100
    return accuracy_percentage

def calculate_prediction_matrix(cleaned_data):
    prediction_matrix = []

    for itr in range(20):
        sum_pred_accuracy = []
        normalised_dataset = feature_normalisation(cleaned_data)

        for k in range(1, 53, 2):
            shuffled_normalised_dataset = shuffle(normalised_dataset)
            part_data = int(len(shuffled_normalised_dataset) * 0.2)
            training_dataset = shuffled_normalised_dataset[part_data:]
            testing_dataset = shuffled_normalised_dataset[:part_data]
            sum_pred_accuracy.append(predict_on_training_dataset(k, training_dataset))

        prediction_matrix.append(sum_pred_accuracy)
    return prediction_matrix


def plot_and_save_results(prediction_matrix):
    np_prediction_matrix = np.array(prediction_matrix)
    transpose_matrix = np_prediction_matrix.transpose()
    
    x_axis = []
    y_axis = []
    count = 1
    
    for row in transpose_matrix:
        average_prediction_accuracy = sum(row) / 20
        #print(average_prediction_accuracy)
        y_axis.append(average_prediction_accuracy)
        x_axis.append(count)
        count += 2

    y_error = [np.std(row) for row in transpose_matrix]

    #print(y_error)
    plt.plot(x_axis, y_axis)
    plt.xlabel('Values of k')
    plt.ylabel('Accuracy over training data')
    plt.errorbar(x_axis, y_axis, yerr=y_error)
    # plt.show()
    plt.savefig("Plot_q1.1.png")
    plt.close()

def calculate_prediction_matrix_testing(cleaned_data):
    prediction_matrix = []

    for itr in range(20):
        sum_pred_accuracy = []
        normalised_dataset = feature_normalisation(cleaned_data)

        for k in range(1, 53, 2):
            # Data shuffling
            shuffled_normalised_dataset = shuffle(normalised_dataset)
            part_data = int(len(shuffled_normalised_dataset) * 0.2)

            # Data partionining
            training_dataset = shuffled_normalised_dataset[part_data:]
            testing_dataset = shuffled_normalised_dataset[:part_data]

            # Using KNN to find accuracy
            sum_pred_accuracy.append(predict_on_testing_dataset(k, training_dataset, testing_dataset))

        prediction_matrix.append(sum_pred_accuracy)

    return prediction_matrix


def plot_and_save_results_testing(prediction_matrix):
    np_prediction_matrix = np.array(prediction_matrix)
    transpose_matrix = np_prediction_matrix.transpose()

    x_axis = []
    y_axis = []
    count = 1

    for row in transpose_matrix:
        average_pred_accuracy = sum(row) / 20
        #print(average_pred_accuracy)
        y_axis.append(average_pred_accuracy)
        x_axis.append(count)
        count += 2

    y_error = [np.std(row) for row in transpose_matrix]

    #print(y_err)
    plt.plot(x_axis, y_axis)
    plt.xlabel('Values of k')
    plt.ylabel('Accuracy over testing data')
    plt.errorbar(x_axis, y_axis, yerr=y_error)
    plt.savefig("Plot_q1.2.png")
    plt.close()

def calculate_prediction_matrix_testing_unnormalized(cleaned_data):
    prediction_matrix = []

    for _ in range(20):
        sum_pred_accuracy = []

        for k in range(1, 53, 2):
            shuffled_data = shuffle(cleaned_data)
            part_data = int(len(shuffled_data) * 0.2)
            training_dataset = shuffled_data[part_data:]
            testing_dataset = shuffled_data[:part_data]
            sum_pred_accuracy.append(predict_on_testing_dataset(k, training_dataset, testing_dataset))

        prediction_matrix.append(sum_pred_accuracy)

    return prediction_matrix

def plot_and_save_results_testing_non_normalized(prediction_matrix):
    np_prediction_matrix = np.array(prediction_matrix)
    transpose_matrix = np_prediction_matrix.transpose()

    x_axis = []
    y_axis = []
    count = 1

    for row in transpose_matrix:
        average_pred_accuracy = sum(row) / 20
        print(average_pred_accuracy)
        y_axis.append(average_pred_accuracy)
        x_axis.append(count)
        count += 2

    y_error = [np.std(row) for row in transpose_matrix]

    #print(y_error)
    plt.plot(x_axis, y_axis)
    plt.xlabel('Values of k')
    plt.ylabel('Accuracy over non normalized testing data')
    plt.errorbar(x_axis, y_axis, yerr=y_error)
    plt.savefig("Plot_q1.6.png")
    plt.close()

if __name__ == "__main__":

    data = load_dataset(filename='iris.csv')
    cleaned_data = [x for x in data if x != []]
    normalised_dataset = feature_normalisation(cleaned_data)

# Plotting accuracy for training data with standard deviation
    prediction_matrix = calculate_prediction_matrix(cleaned_data)
    plot_and_save_results(prediction_matrix)

# Plotting accuracy for testing data with standard deviation
    prediction_matrix_testing = calculate_prediction_matrix_testing(cleaned_data)
    plot_and_save_results_testing(prediction_matrix_testing)

# Plotting accuracy for non-normalized testing data with standard deviation
    prediction_matrix_testing_unnormalized = calculate_prediction_matrix_testing_unnormalized(cleaned_data)
    plot_and_save_results_testing_non_normalized(prediction_matrix_testing_unnormalized)