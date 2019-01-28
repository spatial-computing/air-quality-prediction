from data_preprocessing.services.data_loader import *
from data_preprocessing.services.postgres_connection import Connection
from data_preprocessing.services.interpolation import get_interpolated_values
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

    # TODO: Modify connection method, switch to MVC structure
    conn = Connection(host='jonsnow.usc.edu', database='air_quality_dev')

    config = load_config('data/model/model_config.json')

    feature_set = config['feature_set']

    training_config = load_training_config(config)
    testing_config = load_testing_config(config)

    print('Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    training_data_source = training_config['data_source']
    print('Loading {} {} air quality data.'.format(config['study_area'], training_data_source))
    training_air_model = load_air_quality_data(training_config, conn)
    training_locations = training_air_model.get_locations()

    print('Loading {} {} geographic data.'.format(config['study_area'], training_data_source))
    training_geo_model = load_geo_data(training_locations, feature_set, training_config, conn)

    print('Loading {} {} meterological data.'.format(config['study_area'], training_data_source))
    training_met_model = load_meo_data(training_locations, training_config, conn)

    print('Loading {} {} location coordinates.'.format(config['study_area'], training_data_source))
    training_location_coord = load_location_coord(training_config, conn)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Training ')
    print()
    print('Testing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    testing_data_source = testing_config['data_source']
    print('Loading {} {} air quality data.'.format(config['study_area'], testing_data_source))
    testing_air_model = load_air_quality_data(testing_config, conn)
    testing_locations = testing_air_model.get_locations()

    print('Loading {} {} geographic data.'.format(config['study_area'], testing_data_source))
    testing_geo_model = load_geo_data(testing_locations, feature_set, testing_config, conn)

    print('Loading {} {} meterological data.'.format(config['study_area'], testing_data_source))
    testing_met_model = load_meo_data(testing_locations, testing_config, conn)

    print('Loading {} {} location coordinates.'.format(config['study_area'], testing_data_source))
    testing_location_coord = load_location_coord(testing_config, conn)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Testing ')

    logging.info('Training data source: {}'.format(training_data_source.upper()))
    logging.info('Training stations: {}'.format(list(training_air_model.time_series.columns)))

    logging.info('Testing data source: {}'.format(testing_data_source.upper()))
    logging.info('Testing stations: {}'.format(list(testing_air_model.time_series.columns)))

    if training_met_model is None and testing_met_model is None:
        logging.info('No meterological data used.')

    elif training_met_model is None:
        met_df, locations = get_interpolated_values(testing_met_model, testing_location_coord, training_location_coord)
        training_met_model = MetModel(locations, testing_met_model.config, conn, met_df=met_df)
    elif testing_met_model is None:
        met_df, locations = get_interpolated_values(training_met_model, training_location_coord, testing_location_coord)
        testing_met_model = MetModel(locations, training_met_model.config, conn, met_df=met_df)

    return training_air_model, training_geo_model, training_met_model, \
           testing_air_model, testing_geo_model, testing_met_model, config


def load_training_config(config):
    training_config = config['training']
    training_config['study_area'] = config['study_area']
    if config.get("not_training_stations"):
        training_config["remover"] += config["no_training_stations"]
    return training_config


def load_testing_config(config):
    testing_config = config['testing']
    training_config = config['training']
    testing_config['study_area'] = config['study_area']
    if config.get("not_training_stations") and training_config['data_source'] == testing_config['data_source']:
        testing_config["testing_stations"] = config["not_training_stations"]
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
