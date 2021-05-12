from scipy.io import arff
import pandas as pd
import pandas as pd
import numpy as np
from keras.utils import np_utils
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data_arff(file_name: str):
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    # df_numeric = df.copy()
    # df_numeric._get_numeric_data()
    #
    # scaler = MinMaxScaler()
    # scaler.fit(df_numeric)
    # df_numeric = scaler.transform(df_numeric)



    # df.info()
    return df

def load_csv_data(file_name: str):
    data = pd.read_csv(file_name, delimiter=",")
    return data


def categorial_handle(data: np.ndarray, encode_option: int):
    if encode_option == 1:  #hot hot encoding
        encoder = OneHotEncoder(sparse=False)
        data = encoder.fit_transform(data)
        print("one hot encoder done")
    else:
        encoder = LabelEncoder()
        for column in data.columns:
            data[column] = encoder.fit_transform(data[column])
        data = pd.DataFrame(data)
        print("Categorial to ordinal Done")
    return data  #returns dataFrame


def l2norm(v1, v2):
    # return np.sqrt(np.sum(np.power(v1,2) - np.power(v2,2)))
    return ((v1 - v2) ** 2).sum()

# for DEBUG only
# load_data('mushroom_arff.arff')
