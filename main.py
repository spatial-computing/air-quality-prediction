from utils.data_preparation import prepare_data
from demo import validating_prediction

import pandas as pd
import logging


def main():

    """
        Program settings
    """
    pd.set_option('precision', 15)

    training_air_model, training_geo_model, testing_air_model, testing_geo_model, config = prepare_data()

    # gen_cluster.cluster_main(training_air_model)
    logging.info("Training stations have {} clusters".format(config['n_clusters']))

    # NOTE: This is for the deep learning model
    # gen_context_similarity.get_context_similarity(training_air_model, training_geo_model, config)

    validating_prediction.prediction(training_air_model, training_geo_model,
                                     testing_air_model, testing_geo_model, config)

    # output_data = training_air_model.time_series
    # output_data['date_observed'] = output_data.index
    # output_data.to_csv('data/data/utah_purplar_air_pm25', header=True, index=False, sep=',', mode='w')


if __name__ == "__main__":
    main()
