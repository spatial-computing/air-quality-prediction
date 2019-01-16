import pandas as pd
import numpy as np
import os

"""
    A seperate program that measures the performance
"""


def get_mae(df, ground_truth, prediction_result):
    df['inter_result'] = abs(df[ground_truth] - df[prediction_result])
    mae = df['inter_result'].sum()
    return mae / df.shape[0]


def get_rmse(df, ground_truth, prediction_result):
    df['inter_result'] = (df[ground_truth] - df[prediction_result]) * (df[ground_truth] - df[prediction_result])
    rmse = df['inter_result'].sum()
    return np.sqrt(rmse / df.shape[0])


def get_mape(df, ground_truth, prediction_result):
    df['inter_result'] = abs((df[ground_truth] - df[prediction_result]) / df[ground_truth])
    mape = df['inter_result'].sum()
    return np.sqrt(mape / df.shape[0])


# file_path = '../data/result/cv_pm25_0.005_1536049127/'
file_path = 'data/result/utah_180501_180701_validation_pm25_0.005/'
files = os.listdir(file_path)

df_list = []
for each_file in files:
    if each_file[-3:] == 'csv':
        df = pd.read_csv(file_path + each_file, header=0, sep=',')
        df.dropna(inplace=True)
        cols = df.columns
        print(each_file, len(df), get_mae(df, cols[2], cols[3]))
        print(each_file, len(df), get_rmse(df, cols[2], cols[3]))
        print(each_file, len(df), get_mape(df, cols[2], cols[3]))


print()
#
# # file_path = '../data/result/IDW/O3/'
# file_path = '../data/result/idw_pm25_1536114072/'
# files = os.listdir(file_path)
#
# df_list = []
# for each_file in files:
#     if each_file[-3:] == 'csv':
#         df = pd.DataFrame.from_csv(file_path + each_file, header=0, sep=',')
#         cols = df.columns
#         print(each_file, len(df), get_rmse(df, cols))
