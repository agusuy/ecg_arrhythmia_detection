import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from project_constants import DATASET_FOLDER, RESULTS_FOLDER
from imblearn.under_sampling import RandomUnderSampler

TRAIN_DATA_FILEPATH = os.path.join(DATASET_FOLDER, "mitbih_train.csv")
TEST_DATA_FILEPATH = os.path.join(DATASET_FOLDER, "mitbih_test.csv")


'''
This file contains functions to load the dataset
'''


def load_ecg_data(balance_data=False):
    '''
    Load and pre process the ECG database from CSV files
    '''

    train_dataset = pd.read_csv(TRAIN_DATA_FILEPATH, header=None)
    test_dataset = pd.read_csv(TEST_DATA_FILEPATH, header=None)

    # print(train_dataset.shape)
    # print(test_dataset.shape)

    x_train = train_dataset.iloc[:, 0:-1]
    y_train = train_dataset.iloc[:, -1]
    x_test = test_dataset.iloc[:, 0:-1]
    y_test = test_dataset.iloc[:, -1]
    
    if balance_data:
        x_train, y_train = data_balancing(x_train, y_train)
        x_test, y_test = data_balancing(x_test, y_test)

    # Label encoding and meaning
    # 0) N: Normal beat
    # 1) S: Supraventricular premature beat
    # 2) V: Premature ventricular contraction
    # 3) F: Fusion of ventricular and normal beat
    # 4) Q: Unclassifiable beat
    labels = ['N', 'S', 'V', 'F', 'Q']

    # change label from float to int
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # encode labels
    y_train_encoded = pd.get_dummies(y_train)
    y_test_encoded = pd.get_dummies(y_test)

    return labels, x_train, y_train, y_train_encoded, x_test, y_test, y_test_encoded

def data_balancing(x, y): 
    '''
    Balance dataset classes
    '''   
    sampler = RandomUnderSampler(random_state=42)
    X_smote, y_smote = sampler.fit_resample(x, y)
    return X_smote, y_smote

def plot_data_distribution(data, labels, save_path=None):
    '''
    Plot the ammpunt of data for each class
    '''
    _, counts = np.unique(data, return_counts=True)
    print(counts)
    plt.bar(labels, counts)
    plt.xticks(labels)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Labels in dataset")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    
    balance = False
    file_name = "data_distribution_balance.png" if balance else "data_distribution_imbalance.png"

    data = load_ecg_data(balance)
    labels, x_train, y_train, y_train_encoded, x_test, y_test, y_test_encoded = data

    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)
    print("y_train_encoded:", y_train_encoded.shape)
    print("y_test_encoded:", y_test_encoded.shape)

    distribution_filepath = os.path.join(RESULTS_FOLDER, file_name)
    plot_data_distribution(y_train, labels, save_path=distribution_filepath)
