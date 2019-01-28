from modeling.gen_feature_importance import get_feature_importance
from modeling.gen_prediction import get_prediction_wi_ground_truth
from utils.learning_lib import StandardScaler2
from utils.other_utils import write_csv

import pandas as pd
import numpy as np


def prediction(training_air_model, training_geo_model, testing_air_model, testing_geo_model, config):

    # NOTE: Get training air quality data and geographic data
    training_time_series = training_air_model.time_series
    training_geo_feature_vector = training_geo_model.scaled_geo_feature_vector
    geo_feature_name = training_geo_model.geo_feature_name

    # NOTE: Scale the air quality data according to the mean and standard deviation
    air_scaler = StandardScaler2(
        mean=training_time_series.values[~np.isnan(training_time_series.values)].mean(),
        std=training_time_series.values[~np.isnan(training_time_series.values)].std())
    training_time_series = air_scaler.transform(training_time_series)

    important_feature_list, _ = get_feature_importance(training_time_series,
                                                       training_geo_feature_vector, geo_feature_name, config)

    # NOTE: Get testing air quality data and geographic data
    testing_time_series = testing_air_model.time_series
    testing_geo_feature_vector = testing_geo_model.scaled_geo_feature_vector
    important_feature_list = list(set(important_feature_list).intersection(set(testing_geo_model.geo_feature_name)))

    x_train = training_geo_feature_vector.loc[important_feature_list]
    x_test = testing_geo_feature_vector.loc[important_feature_list]

    # NOTE: Create a new DataFrame to store the result
    result_list = []
    testing_time = sorted(list(set(training_time_series.index).intersection(set(testing_time_series.index))))

    for each_time in testing_time:

        result_df = get_prediction_wi_ground_truth(training_time_series, testing_time_series, each_time,
                                                   x_train, x_test, air_scaler, config)
        result_list.append(result_df)

    result = pd.concat(result_list)
    file_path = config['output_path'] + '/prediction.csv'
    write_csv(result, file_path)
    print('Prediction is finished')
