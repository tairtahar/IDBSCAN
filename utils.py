from scipy.io import arff
import numpy as np
# from keras.utils import np_utils
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_svmlight_file


def load_data_arff(file_name: str):
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    return df


def load_preprocess_mushrooms(flag=False):
    if flag:  # manual data
        df = load_data_arff('datasets/mushroom_arff.arff')
        df = df.drop(df.columns[12], axis=1)  # discarding column with missing values
        true_class = df.iloc[:, -1]
        df = df.drop(df.columns[-1], axis=1)
        df = categorial_handle(df, 1)
        encoder = LabelEncoder()
        true_class = encoder.fit_transform(true_class)
        # df.info()
    else:  # svm data
        data, true_class = load_svmlight_file("datasets/mushrooms.txt")
        df = pd.DataFrame(data.todense())
        true_class = np.asarray(true_class)
    return df, true_class


def load_preprocess_letters(flag=False):
    if flag:
        df = load_csv_data("datasets/letter.csv")
        true_class = df.iloc[:, -1]
        df = df.drop(df.columns[-1], axis=1)
        # df = categorial_handle(df, 2)  # ordinal
        encoder = LabelEncoder()
        true_class = encoder.fit_transform(true_class)
        scaler = StandardScaler()  # StandardScaler()  #
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df))  # normalized data
        print("normalization done")
    else:
        data, true_class = load_svmlight_file("datasets/letter.scale.txt")
        df = pd.DataFrame(data.todense())
        true_class = np.asarray(true_class)
    return df, true_class


def load_preprocess_catadata():
    data, true_class = load_svmlight_file("datasets/cadata.txt")
    df = pd.DataFrame(data.todense())
    true_class = np.asarray(true_class)
    return df, true_class


def load_preprocess_pendigit():
    data, true_class = load_svmlight_file("datasets/pendigits.txt")
    df = pd.DataFrame(data.todense())
    true_class = np.asarray(true_class)
    return df, true_class


def load_preprocess_abalone():
    data, true_class = load_svmlight_file("datasets/abalone_scale.txt")
    df = pd.DataFrame(data.todense())
    true_class = np.asarray(true_class)
    return df, true_class


def load_preprocess_sensorless():
    data, true_class = load_svmlight_file("datasets/Sensorless.txt")
    df = pd.DataFrame(data.todense())
    true_class = np.asarray(true_class)
    return df, true_class


def load_csv_data(file_name: str):
    data = pd.read_csv(file_name, delimiter=",")
    return data


def categorial_handle(data: np.ndarray, encode_option: int):
    if encode_option == 1:  # hot hot encoding
        encoder = OneHotEncoder(sparse=False)
        data = encoder.fit_transform(data)
        data = pd.DataFrame(data)
        print("one hot encoder done")
    else:
        encoder = LabelEncoder()
        for column in data.columns:
            data[column] = encoder.fit_transform(data[column])
        data = pd.DataFrame(data)
        print("Categorial to ordinal Done")
    return data  # returns dataFrame


def l2norm(v1, v2):
    # return np.sqrt(np.sum(np.power(v1,2) - np.power(v2,2)))
    # return ((v1 - v2) ** 2).sum()
    return np.linalg.norm(v1-v2, ord=2)

# for DEBUG only
# load_data('mushroom_arff.arff')
