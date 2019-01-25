from data_preprocessing import load_data
from utils.utils import load_config, write_csv
from modeling import gen_context_similarity
from modeling import gen_cluster
from demo import validating_prediction
import torch
from torch import nn


import numpy as np
import pandas as pd
import logging


def feature_selection():

    pass

def placeholder(vec1, vec2):
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)


    print(vec1.shape)
    print(vec2.shape)
    return vec1_np


def main():

    """
        Program settings
    """
    pd.set_option('precision', 15)
    config = load_config('/Users/yijunlin/PycharmProjects/air-quality-prediction/data/model/model_config.json')

    feature_set = config['feature_set']
    training_config = config['training']
    testing_config = config['testing']

    # NOTE: if "is_split" is on, means we need to split the training data
    if config["is_split"] and config["no_training_stations"]:
        training_config["remover"] += config["no_training_stations"]
        if training_config['data_source'] == testing_config['data_source']:
            testing_config["testing_stations"] = config["no_training_stations"]
        else:
            testing_config["testing_stations"] = []

    # NOTE: get the training and testing data
    training_air_model, training_geo_model = load_data(feature_set, training_config)
    testing_air_model, testing_geo_model = load_data(feature_set, testing_config)

    logging.info("Training data source: {}".format(training_config["data_source"].upper()))
    logging.info("Training stations: {}".format(list(training_air_model.time_series.columns)))

    logging.info("Testing data source: {}".format(testing_config["data_source"].upper()))
    logging.info("Testing stations: {}".format(list(testing_air_model.time_series.columns)))

    # gen_cluster.cluster_main(training_air_model)
    logging.info("Training stations have {} clusters".format(config['n_clusters']))

    return training_air_model, training_geo_model


if __name__ == '__main__':
    training_air_model, training_geo_model = main()
    placeholder(training_air_model.time_series, training_geo_model.scaled_geo_feature_vector)