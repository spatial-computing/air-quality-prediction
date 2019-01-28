import pandas as pd

KEY_COL = 'station_id'
TIME_COL = 'date_observed'


def time_series_construction_archive_1(df, value_col='value'):
    """
    Construct time series vector, indexed by distinct timestamp, columned by locations
    Using merge function to merge locations one by one
    If there is not timestamp at the location, filled with NaN as default

    :param df: input dataFrame
    :param value_col: column name of value
    :return: time series format data
    """
    time_idx = sorted(list(set(df[TIME_COL])))
    time_series = pd.DataFrame(time_idx, columns=[TIME_COL])
    time_series.set_index(TIME_COL, inplace=True)

    df_grouped = df.groupby(KEY_COL)
    for each_group in df_grouped.groups:
        group = df_grouped.get_group(each_group)
        sorted_group = group.sort_values(by=TIME_COL)[[TIME_COL, value_col]]
        sorted_group.set_index(TIME_COL, inplace=True)
        time_series = time_series.join(sorted_group, how='left')
        time_series.rename(columns={value_col: each_group}, inplace=True)
    return time_series


def time_series_construction(df, value_col='value'):
    """
    An updated version from the first construction method
    Should be faster than the previous method

    :param df: input dataFrame
    :param value_col: column name of value
    :return: time series format data
    """
    min_time = min(df[TIME_COL])
    max_time = max(df[TIME_COL])
    times = pd.date_range(start=min_time, end=max_time, freq='1H')
    times_pd = pd.DataFrame(index=times)

    time_series_list = []

    df_grouped = df.groupby(KEY_COL)
    for each_group in df_grouped.groups:
        group = df_grouped.get_group(each_group)
        group.set_index(TIME_COL, inplace=True)
        time_series = times_pd.join(group[value_col], how='left')
        time_series.rename(columns={value_col: each_group}, inplace=True)
        time_series_list.append(time_series)
    time_series_result = pd.concat(time_series_list, axis=1)
    return time_series_result


def time_series_construction_archive_2(df, value_col='value'):
    """
    Join operation takes too long if the column number is above 20.
    This will also takes sometime but short than join when column number is large

    :param df: input dataFrame
    :param value_col: column name of value
    :return: time series format data
    """
    min_time = min(df[TIME_COL])
    max_time = max(df[TIME_COL])
    times = pd.date_range(start=min_time, end=max_time, freq='1H')
    keys = list(set(df[KEY_COL]))

    result_time_series = pd.DataFrame(index=times, columns=keys)

    for index, row in df.iterrows():
        station_id = row[KEY_COL]
        time = row[TIME_COL]
        result_time_series.loc[time, station_id] = row[value_col]
    return result_time_series


def time_series_smoothing(time_seires, window_size, method='mean'):
    """
        Smooth each time series (each column is a time series)

    """
    min_time = min(time_seires.index)
    max_time = max(time_seires.index)
    assert min_time < max_time

    times = pd.date_range(start=min_time, end=max_time, freq='1H')
    time_series_smooth = pd.DataFrame(index=times)

    if method == 'mean':
        for idx, key in enumerate(time_seires.columns):
            ts = time_seires[key]
            rolmean = ts.rolling(window=window_size, min_periods=1, center=True).mean()
            time_series_smooth[key] = rolmean

    if method == 'median':
        for idx, key in enumerate(time_seires.columns):
            ts = time_seires[key]
            rolmed = ts.rolling(window=window_size, min_periods=1, center=True).median()
            time_series_smooth[key] = rolmed

    return time_series_smooth


def check_max_min_correlation(df, stations):
    """
        Check the correlation of the duplicates at the same location & time

    """

    df_min = df.groupby([KEY_COL, TIME_COL]).min()
    df_max = df.groupby([KEY_COL, TIME_COL]).max()
    joint_min_max = df_min.join(df_max, lsuffix='_min', rsuffix='_max')
    max_min_corr = {}
    for station in stations:
        corr = joint_min_max.loc[station].corr(method='pearson')
        max_min_corr[station] = corr.iloc[0, 1]
    return max_min_corr

