from data_preprocessing import load_data
from utils.other_utils import load_json

import logging
import time
import os


def prepare_data():
    """
    Load config information
    Construct training config and testing config
    Load training and testing data

    :return: training_air_model, training_geo_model, testing_air_model, testing_geo_model, config
    """

    config = load_config('data/model/model_config.json')

    feature_set = config['feature_set']
    training_config = load_training_config(config)
    testing_config = load_testing_config(config)

    training_air_model, training_geo_model = load_data(feature_set, training_config)
    testing_air_model, testing_geo_model = load_data(feature_set, testing_config)

    logging.info("Training data source: {}".format(training_config["data_source"].upper()))
    logging.info("Training stations: {}".format(list(training_air_model.time_series.columns)))

    logging.info("Testing data source: {}".format(testing_config["data_source"].upper()))
    logging.info("Testing stations: {}".format(list(testing_air_model.time_series.columns)))

    return training_air_model, training_geo_model, testing_air_model, testing_geo_model, config


def load_training_config(config):
    training_config = config['training']
    training_config['study_area'] = config['study_area']
    if config.get("no_training_stations"):
        training_config["remover"] += config["no_training_stations"]
    return training_config


def load_testing_config(config):
    testing_config = config['testing']
    training_config = config['training']
    testing_config['study_area'] = config['study_area']
    if config.get("no_training_stations") and training_config['data_source'] == testing_config['data_source']:
        testing_config["testing_stations"] = config["no_training_stations"]
    return testing_config


def load_config(file_path):
    config = load_json(file_path)
    data_file_path = config['data_config']
    data_config = load_json(data_file_path)
    config = dict(config, **data_config)
    config['time'] = int(time.time())  # NOTE: Add the current time in the config
    config['output_path'] = get_output_file_path(config)  # NOTE: Add output path in the config
    conf_logging(config)
    return config


def conf_logging(config):
    logging.basicConfig(filename=config['output_path'] + '/information.log', level=logging.INFO)


def get_output_file_path(config):
    testing_method = config['study_method']
    testing_content = config['study_content']
    output_path = config['result_path']
    study_area = config['study_area']
    study_time = config['study_time']
    parameter_name = config['parameter_name']
    n_clusters = str(config['n_clusters'])
    geo_feature_percent = str(config['geo_feature_percent'])
    output_path = output_path + '_'.join([study_area, study_time, testing_method,
                                          testing_content, parameter_name, n_clusters, geo_feature_percent])
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path
