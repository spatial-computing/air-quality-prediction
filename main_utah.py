from demo import validating_prediction
from data_preprocessing.preprocess.utah_purple_air import utah_purple_air_preprocess
from data_preprocessing.preprocess.utah_epa import utah_epa_preprocess
from lib.utils import load_config
from modeling.gen_context_similarity import get_context_similarity

import pandas as pd
import time
import os


def main():

    """
        Program settings
    """
    pd.set_option('precision', 15)
    config = get_config('data/model/utah_model_config.json')

    feature_set = config['feature_set']
    training_config = config['training']
    testing_config = config['training']

    training_air_model, training_geo_model = None, None
    if training_config['air_quality']['data_source'] == 'purple_air':
        training_air_model, training_geo_model = utah_purple_air_preprocess(feature_set, training_config)
    elif training_config['air_quality']['data_source'] == 'epa':
        training_air_model, training_geo_model = utah_epa_preprocess(feature_set, training_config)

    testing_air_model, testing_geo_model = None, None
    if testing_config['air_quality']['data_source'] == 'purple_air':
        testing_air_model, testing_geo_model = utah_purple_air_preprocess(feature_set, testing_config)
    elif testing_config['air_quality']['data_source'] == 'epa':
        testing_air_model, testing_geo_model = utah_epa_preprocess(feature_set, testing_config)

    # get_cluster.cluster_main(training_air_model)
    get_context_similarity(training_air_model, training_geo_model, config)

    # validating_prediction.prediction(training_air_model, training_geo_model,
    #                                  testing_air_model, testing_geo_model, config)


if __name__ == "__main__":
    main()
