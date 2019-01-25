from scipy.spatial.distance import euclidean

import pandas as pd


def paa_similarity(df, seg_type, key_column='location', time_column='timestamp', value_column='value'):

    df = timestamp_segmentation(df, time_column)
    selected_columns = [key_column, seg_type, value_column]
    df_selected = df[selected_columns]

    time_series = time_series_construction_avg(df_selected, index=seg_type)
    time_series.dropna(inplace=True)

    sim_mat = pd.DataFrame(index=time_series.columns, columns=time_series.columns)

    for each_col_a in time_series.columns:
        for each_col_b in time_series.columns:
            sim = euclidean_sim(time_series[each_col_a], time_series[each_col_b])
            sim_mat[each_col_a][each_col_b] = sim

    return sim_mat


def timestamp_segmentation(df, time_column_name):
    """
    Get day, week, and month respectively from timestamp

    """
    df['hour'] = df[time_column_name].apply(lambda x: x.strftime('%Y-%m-%d %H'))
    df['day'] = df[time_column_name].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['week'] = df[time_column_name].apply(lambda x: x.strftime('%Y %W'))
    df['month'] = df[time_column_name].apply(lambda x: x.strftime('%Y-%m'))
    return df


def euclidean_sim(a, b):
    dis = euclidean(a, b)
    return 1.0 / (1.0 + dis / len(a))