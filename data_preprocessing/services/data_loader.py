from data_preprocessing.data_model.air_model import AirModel
from data_preprocessing.data_model.geo_model import GeoModel
from data_preprocessing.data_model.met_model import MetModel
from data_preprocessing.data_model.location_model import LocModel
from data_preprocessing.services.timeseries_utils import *


KEY_COL = 'station_id'
TIME_COL = 'date_observed'


def load_location_coord(config, conn):
    return LocModel(config['location_coord'], conn=conn)


def load_meo_data(locations, config, conn):
    if config.get('meterology'):
        return MetModel(locations, config['meterology'], conn=conn)
    else:
        return None


def load_geo_data(locations, feature_set, config, conn):
    return GeoModel(locations, feature_set, config['geo_feature'], conn=conn)


def load_air_quality_data(config, conn):

    air_quality_config = config['air_quality']
    air_quality_model = AirModel(air_quality_config, conn=conn)

    if config["tag"] == "testing" and config.get("testing_stations"):
        air_quality_model.select_stations(config["testing_stations"])

    assert air_quality_model.air_quality_df is not None

    if config['study_area'] == 'utah':
        air_quality_model = load_utah_air_quality(air_quality_model, air_quality_config)
    if config['study_area'] == 'los_angeles':
        air_quality_model = load_la_air_quality(air_quality_model, air_quality_config)

    time_series = time_series_construction(air_quality_model.air_quality_df)
    air_quality_model.time_series = time_series
    assert air_quality_model.time_series is not None

    # NOTE: Smooth the time series only if there is a window tag
    if air_quality_config.get('ts_smooth_window_size'):
        air_quality_model.time_series = time_series_smoothing(time_series, air_quality_config['ts_smooth_window_size'])
    return air_quality_model


def load_la_air_quality(air_quality_model, config):

    remover = config['remover']
    air_quality_model.remove_stations(remover)
    return air_quality_model


def load_utah_air_quality(air_quality_model, config):

    remover = config['remover']

    # NOTE: Utah data have duplicates for each pair [Station, Date_observed]
    # NOTE: 1. check the duplicates if the duplicates would effect (too different btw duplicates)
    max_min_corr_dic = check_max_min_correlation(air_quality_model.air_quality_df, air_quality_model.get_locations())

    #       2. remove the stations that correlation of the values is low
    for k in max_min_corr_dic:
        if max_min_corr_dic[k] < 0.8 and k not in remover:
            remover.append(k)

    air_quality_model.remove_stations(remover)

    #       3. take the mean of the duplicates for the rest of the stations
    air_quality_model.air_quality_df = air_quality_model.air_quality_df.groupby([KEY_COL, TIME_COL]).mean()
    air_quality_model.air_quality_df.reset_index(inplace=True)

    return air_quality_model
