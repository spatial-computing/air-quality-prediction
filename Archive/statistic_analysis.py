import pandas as pd
import numpy as np
import os
import math
import statistics


def distinct_rows(df, col):
    location = df[col]
    return len(set(list(location)))


def time_range(df):
    start_time = min(df.index)
    end_time = max(df.index)
    return start_time, end_time, len(set(list(df.index)))


def cal_stats(df, col):
    a_list = list(df[col])
    return sum(a_list) / len(a_list), statistics.stdev(a_list)


def get_grouped_mae(df, ground_truth, prediction_res):
    df['diff'] = abs(df[ground_truth] - df[prediction_res])
    diff = df[['location','diff']]
    grouped_mae = diff.groupby(diff['location']).mean()
    mae_mean = grouped_mae.mean()
    mae_std = grouped_mae.std()
    return mae_mean, mae_std


def get_grouped_rmse(df, ground_truth, prediction_res):
    df['diff'] = (df[ground_truth] - df[prediction_res]) * (df[ground_truth] - df[prediction_res])
    diff = df[['location','diff']]
    grouped_rmse = diff.groupby(diff['location']).mean().apply(np.sqrt)
    rmse_mean = grouped_rmse.mean()
    rmse_std = grouped_rmse.std()
    return rmse_mean, rmse_std


def select_sentence_in_qgis(df, timestamp_list):
    df['diff'] = abs(df['ground_truth'] - df['prediction_value'])
    df = df.loc[df['timestamp'].isin(timestamp_list)][['location', 'diff']]
    grouped_df = df.groupby(df['location']).mean()
    grouped_df_list = list(grouped_df.loc[grouped_df['diff'] > 80].index)
    # all_locations = ['\"id\"=' + str(x) for x in all_locations]
    grouped_df_list = ['\"id\"=' + str(x) for x in grouped_df_list]
    # print(' OR '.join(all_locations))
    print(' OR '.join(grouped_df_list))
    print()


file_path = 'data/result/validation_pm25_cleaned_0.005_1537766229/'
files = os.listdir(file_path)

df_list = []
for each_file in files:
    if each_file[-3:] == 'csv':
        df = pd.read_csv(file_path + each_file, header=0, sep=',')
        df.dropna(inplace=True)
        cols = df.columns
        print(each_file, len(df), distinct_rows(df, 'location'))
        print(time_range(df))

        print(cal_stats(df, 'ground_truth'))
        print(cal_stats(df, 'prediction_value'))
        print(get_grouped_mae(df, 'ground_truth', 'prediction_value'))
        print(get_grouped_rmse(df, 'ground_truth', 'prediction_value'))
        select_sentence_in_qgis(df, ['2018-08-24 03:00:00-07:00',
                                     '2018-08-24 04:00:00-07:00',
                                     '2018-08-24 05:00:00-07:00',
                                     '2018-08-24 06:00:00-07:00',
                                     '2018-08-24 07:00:00-07:00'])

print()
