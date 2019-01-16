from demo import cross_validation, validating_prediction, fine_scale_prediction, interpolation
from data_preprocessing.preprocess.los_angeles_purple_air import los_angeles_ppa_preprocess
from data_preprocessing.preprocess.los_angeles_epa import los_angeles_epa_preprocess
from lib.utils import load_config

import pandas as pd


def main():

    """
        Program settings
    """
    pd.set_option('precision', 15)
    config = load_config('data/model/utah_model_config.json')

    feature_set = config['feature_set']
    training_config = config['training']
    testing_config = config['training']

    training_air_model, training_geo_model = None, None
    if training_config['air_quality']['data_source'] == 'purple_air':
        training_air_model, training_geo_model = los_angeles_ppa_preprocess(feature_set, training_config)
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
