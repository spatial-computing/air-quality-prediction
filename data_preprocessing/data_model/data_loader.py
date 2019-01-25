from .air_model import AIRModel
from .geo_model import GEOModel
from data_preprocessing.services import *


def load_data(feature_set, config):

    # TODO: Modify connection method, switch to MVC structure
    conn = Connection(host='jonsnow.usc.edu', database='air_quality_dev')

    air_quality_model = AIRModel(config, conn=conn)

    if config["tag"] == "testing" and config.get("testing_stations"):
        air_quality_model.select_stations(config["testing_stations"])

    assert air_quality_model.air_quality_df is not None

    switch_case(air_quality_model, config)

    time_series = time_series_construction(air_quality_model.air_quality_df)
    air_quality_model.time_series = time_series
    assert air_quality_model.time_series is not None

    # NOTE: Smooth the time series only if there is a window tag
    if config['air_quality'].get('smooth_window_size'):
        air_quality_model.time_series = time_series_smoothing(time_series,
                                                              window_size=config['air_quality']['smooth_window_size'])

    # NOTE: Get the geographic features
    geo_feature_model = GEOModel(air_quality_model.get_locations(), feature_set, config, conn=conn)

    print('Loading {} {} air quality data and geographic data loading.'
          .format(config['study_area'], config['data_source']))

    conn.close_conn()
    return air_quality_model, geo_feature_model


def switch_case(air_quality_model, config):
    area = config['study_area']
    if area == 'utah':
        load_utah_air_quality(air_quality_model, config)
    if area == 'los_angeles':
        load_la_air_quality(air_quality_model, config)


def load_la_air_quality(air_quality_model, config):

    remover = config['remover']
    air_quality_model.remove_stations(remover)

    return air_quality_model


def load_utah_air_quality(air_quality_model, config):

    remover = config['remover']

    # NOTE: Utah data have duplicates for each pair [Station, Date_observed]
    # NOTE: 1. check the duplicates if the duplicates would effect (too different btw duplicates)
    max_min_corr_dic = check_max_min_correlation(air_quality_model.air_quality_df, air_quality_model.get_locations(),
                                                 key_col='station_id', time_col='date_observed')

    #       2. remove the stations that correlation of the values is low
    for k in max_min_corr_dic:
        if max_min_corr_dic[k] < 0.8 and k not in remover:
            remover.append(k)
    air_quality_model.remove_stations(remover)

    #       3. take the mean of the duplicates for the rest of the stations
    air_quality_model.air_quality_df = air_quality_model.air_quality_df.groupby(['station_id', 'date_observed']).mean()
    air_quality_model.air_quality_df.reset_index(inplace=True)

    return air_quality_model
