from lib.libs import random_forest_regressor

import pandas as pd


def get_prediction_wi_ground_truth(train_time_series, testing_time_series, time, x_train, x_test, air_scaler, config):

    rf_regression_tree_num = config['rf_regression_tree_num']
    rf_regression_tree_depth = config['rf_regression_tree_depth']

    ground_truth_list = list(testing_time_series.loc[time])

    y_train = train_time_series.loc[time]
    y_train.dropna(inplace=True)
    loc = list(y_train.index)

    # If there are very few number of other stations, continue
    if len(loc) >= int(len(train_time_series.shape[1]) * 0.8):
        predict_result = random_forest_regressor(x_train[loc].T, y_train, x_test.T,
                                                 rf_regression_tree_num, rf_regression_tree_depth)
        predict_result = air_scaler.inverse_transform(predict_result)

        time_list = [time for i in range(len(predict_result))]
        result_df = pd.DataFrame(list(zip(list(x_test.columns), time_list, ground_truth_list, predict_result)),
                                 columns=['location', 'timestamp', 'ground_truth', 'prediction_value'])

        print('{} finished.'.format(time))
        return result_df
    else:
        return None


def get_prediction_wo_ground_truth(train_time_series, time, x_train, x_test, air_scaler, config):

    rf_regression_tree_num = config['rf_regression_tree_num']
    rf_regression_tree_depth = config['rf_regression_tree_depth']

    y_train = train_time_series.loc[time]
    y_train.dropna(inplace=True)
    loc = list(y_train.index)

    # If there are very few number of other stations, continue
    if len(loc) >= int(len(train_time_series.shape[1]) * 0.8):
        predict_result = random_forest_regressor(x_train[loc].T, y_train, x_test.T,
                                                 rf_regression_tree_num, rf_regression_tree_depth)
        predict_result = air_scaler.inverse_transform(predict_result)

        time_list = [time for i in range(len(predict_result))]
        result_df = pd.DataFrame(list(zip(list(x_test.columns), time_list, predict_result)),
                                 columns=['location', 'timestamp', 'prediction_value'])

        print('{} finished.'.format(time))
        return result_df
    else:
        return None

