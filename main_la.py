from data_preprocessing import load_data
from data_preprocessing.preprocess.utah_epa import load_utah_epa
from utils.utils import load_config, write_csv, load_training_testing_config
from modeling import gen_context_similarity
from modeling import gen_cluster
from demo import validating_prediction

import pandas as pd
import logging


def main():

    """
        Program settings
    """
    pd.set_option('precision', 15)
    config = load_config('data/model/model_config.json')

    feature_set = config['feature_set']
    training_config, testing_config = load_training_testing_config(config)

    training_air_model, training_geo_model = None, None
    if training_config['air_quality']['data_source'] == 'purple_air':
        training_air_model, training_geo_model = load_los_angeles_ppa_preprocess(feature_set, training_config)
    elif training_config['air_quality']['data_source'] == 'epa':
        training_air_model, training_geo_model = los_angeles_epa_preprocess(feature_set, training_config)

    testing_air_model, testing_geo_model = None, None
    if testing_config['air_quality']['data_source'] == 'purple_air':
        testing_air_model, testing_geo_model = los_angeles_ppa_preprocess(feature_set, testing_config)
    elif testing_config['air_quality']['data_source'] == 'epa':
        testing_air_model, testing_geo_model = los_angeles_epa_preprocess(feature_set, testing_config)

    # if config['testing_method'] == 'cv':
    #     cross_validation.prediction(air_quality_model, geo_feature_model, config)
    #
    # elif config['testing_method'] == 'cv_idw':
    #     interpolation.prediction(air_quality_model, config, conn)
    #
    # elif config['testing_method'] == 'validation':
    #     validating_prediction.prediction(air_quality_model, geo_feature_model, config, conn)
    #
    # elif config['testing_method'] == 'validation_idw':
    #     interpolation.prediction(air_quality_model, config, conn)
    #
    # elif config['testing_method'] == 'fishnet':
    #     fine_scale_prediction.prediction(air_quality_model, geo_feature_model,
    #                                      la_fishnet_geofeature_tablename, config, conn)


if __name__ == "__main__":
    main()
