from modeling.gen_prediction import get_prediction_wi_ground_truth
from utils.libs import *
from modeling.gen_feature_importance import *
from utils.helpers import write_csv

import pandas as pd
import numpy as np
import os


def prediction(air_model, geo_model, config):

    # NOTE: Get training air quality data and geographic data
    time_series = air_model.time_series
    geo_feature_vector = geo_model.scaled_geo_feature_vector
    geo_feature_name = geo_model.geo_feature_name

    station_locations = air_model.get_locations()

    for station in station_locations:

        other_stations = [i for i in station_locations if i != station]
        training_time_series = time_series[other_stations]
        testing_time_series = time_series[station].dropna()
        training_geo_feature_vector = geo_feature_vector[other_stations]
        testing_geo_feature_vector = geo_feature_vector[station]

        # NOTE: Scale the air quality data according to the mean and standard deviation
        air_scaler = StandardScaler2(
            mean=training_time_series.values[~np.isnan(training_time_series.values)].mean(),
            std=training_time_series.values[~np.isnan(training_time_series.values)].std())
        training_time_series = air_scaler.transform(training_time_series)

        important_feature_list, _ = get_feature_importance(training_time_series,
                                                           training_geo_feature_vector, geo_feature_name, config)

        x_train = training_geo_feature_vector.loc[important_feature_list]
        x_test = testing_geo_feature_vector.loc[important_feature_list]

        # Just for debugging and visualization
        x_view = geo_feature_vector.loc[important_feature_list]
        x_view_testing = geo_feature_vector.loc[important_feature_list]
        # End

        result_list = []
        testing_time = list(training_time_series.index)

        for each_time in testing_time:
            result_df = get_prediction_wi_ground_truth(training_time_series, testing_time_series, each_time,
                                                       x_train, x_test, air_scaler, config)
            result_list.append(result_df)

        result = pd.concat(result_list)
        result = result.set_index('timestamp')
        file_path = config['output_path'] + '/' + station + '.csv'
        write_csv(result, file_path)
        print('location {} is finished'.format(station))


