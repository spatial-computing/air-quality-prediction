from lib.helpers import geo_distance
from lib.helpers import write_csv

import pandas as pd
import numpy as np


def prediction(air_quality_model, config, conn):

    print('Start prediction using IDW.')

    air_quality_locations = air_quality_model.air_quality_locations
    air_quality_time_series = air_quality_model.air_quality_time_series

    training_location_table_name = config['training_location_table_name']
    training_location_column_set = config['training_location_column_set']

    sql = 'select {columns} from {table_name};'\
        .format(columns=','.join(training_location_column_set), table_name=training_location_table_name)
    training_location = conn.execute_wi_return(sql)
    conn.close_conn()

    training_location_df = pd.DataFrame(training_location, columns=['id', 'x', 'y'])
    training_location_df = training_location_df[training_location_df['id'].isin(air_quality_locations)]
    training_location_df.drop_duplicates(inplace=True)

    if config['testing_method'] == 'cv_idw':
        weight_mat = get_weight_mat(training_location_df, training_location_df)

        for each_location in air_quality_locations:
            training_air_quality_time_series = air_quality_time_series.drop([each_location], axis=1)
            testing_air_quality_time_series = air_quality_time_series[each_location].dropna()
            weight = weight_mat[weight_mat['id_a'] == each_location]
            weight = weight.set_index('id_b')
            result = idw(training_air_quality_time_series, testing_air_quality_time_series, weight)
            file_path = config['output_path'] + '/' + each_location + '.csv'
            write_csv(result, file_path)
            print('location' + each_location + 'is finished')

    if config['testing_method'] == 'validation_idw':

        validating_air_quality_model = los_angeles_purple_air_preprocess(config, conn)
        validating_air_quality_location = validating_air_quality_model.air_quality_location

        validating_location_table_name = config['validating_location_table_name']
        validating_location_column_set = config['validating_location_column_set']
        validating_location = conn.read(validating_location_table_name, validating_location_column_set)
        validating_location_df = pd.DataFrame(validating_location,
                                              columns=['id', 'sensor_id', 'channel', 'x', 'y'])[['id', 'x', 'y']]
        validating_location_df = validating_location_df[validating_location_df['id'].isin(validating_air_quality_location)]
        validating_location_df.drop_duplicates(inplace=True)

        # validating_location_df.to_csv('./data/validating_locations.csv', header=True, index=True, sep=',', mode='w')

        weight_mat = get_weight_mat(training_location_df, validating_location_df)
        weight = weight_mat.set_index('id_b')
        result = idw(air_quality_time_series, validating_air_quality_model.air_quality_time_series, weight)
        file_path = config['output_path'] + '/prediction.csv'
        write_csv(result, file_path)
        print('Prediction is finished')


def get_weight_mat(df1, df2):
    df1['key'] = 1
    df2['key'] = 1
    df = df1.merge(df2, on='key', suffixes=['_a', '_b']).drop('key', axis=1)
    df['distance'] = df.apply(lambda x: geo_distance(x['x_a'], x['y_a'], x['x_b'], x['y_b']), axis=1)
    df['weight'] = df.apply(lambda x: np.nan if x['distance'] == 0.0 else 1 / x['distance'], axis=1)
    weight_mat = df[['id_a', 'id_b', 'weight']]
    return weight_mat


def idw(training_air_quality_time_series, testing_air_quality_time_series, weight):

    testing_time = list(testing_air_quality_time_series.index)

    # Create a new dataframe to store the result
    result = pd.DataFrame(testing_air_quality_time_series)
    result['predict_result'] = np.nan

    for each_time in testing_time:
        y = pd.DataFrame(training_air_quality_time_series.loc[each_time])
        y.dropna(inplace=True)
        loc = list(y.index)

        # If there are very few number of other stations, continue
        if len(loc) >= int(len(training_air_quality_time_series.columns) * 0.8):
            y = y.join(weight)
            y['prediction'] = y[each_time] * y['weight']
            weight_sum = y['weight'].sum()
            prediction_result = y['prediction'].sum()
            result.loc[each_time, 'predict_result'] = prediction_result / weight_sum
    return result


def idw1(training_air_quality_time_series, testing_air_quality_time_series, weight, config):

    testing_time = list(testing_air_quality_time_series.index)

    """
        Create a new DataFrame to store the result
    """
    result_list = []

    for each_time in testing_time:

        y = training_air_quality_time_series.loc[each_time]
        y.dropna(inplace=True)
        loc = list(y.index)

        '''
            If there are very few number of other stations, continue
        '''
        if len(loc) >= int(len(training_air_quality_time_series.columns) * 0.8):
            y = y.join(weight)
            y['prediction'] = y[each_time] * y['weight']
            weight_sum = y['weight'].sum()
            y['prediction_result'] = y['prediction'] / weight_sum

            ground_truth = list(testing_air_quality_time_series.loc[each_time])
            time_list = [each_time for x in range(len(ground_truth))]
            predict_result = list(y['prediction_result'])

            result_df = pd.DataFrame(list(zip(list(testing_air_quality_time_series.columns), time_list, ground_truth, predict_result)),
                                     columns=['location', 'timestamp', 'ground_truth', 'prediction_value'])
            result_list.append(result_df)

    result = pd.concat(result_list)
    result = result.set_index('timestamp')
    Writer(result, "prediction", config, file_type='.csv')
    print("prediction", 'finished.')

