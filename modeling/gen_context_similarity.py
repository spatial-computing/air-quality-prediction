from utils import StandardScaler2, standard_scaler
from modeling.gen_feature_importance import get_feature_importance

import numpy as np
import pickle
from scipy import spatial


def get_context_similarity(training_air_model, training_geo_model, config):

    # NOTE: Get training air quality data and geographic data
    training_time_series = training_air_model.time_series
    training_geo_feature_vector = training_geo_model.scaled_geo_feature_vector
    geo_feature_name = training_geo_model.geo_feature_name

    # NOTE: Scale the air quality data according to the mean and standard deviation
    air_scaler = StandardScaler2(
        mean=training_time_series.values[~np.isnan(training_time_series.values)].mean(),
        std=training_time_series.values[~np.isnan(training_time_series.values)].std())
    training_time_series = air_scaler.transform(training_time_series)

    important_feature_list, _ = get_feature_importance(training_time_series,
                                                       training_geo_feature_vector, geo_feature_name, config)

    geo_context = training_geo_feature_vector.loc[important_feature_list]
    geo_context = standard_scaler(geo_context)

    stations = list(training_geo_feature_vector.columns)

    # NOTE: Build sensor id to index map.
    station_id_to_ind = get_index(stations)

    # NOTE: Build distance matrix
    dist_mx = get_dist_matrix(stations, geo_context, station_id_to_ind)

    # NOTE: Normalize distance matrix and get the adjacency matrix
    distances = dist_mx[~np.equal(dist_mx, np.inf)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std) / 2)
    adj_mx[adj_mx < 0.1] = 0

    print(adj_mx)

    # NOTE: Write to a pkl file
    with open('/Users/yijunlin/PycharmProjects/air-quality-prediction/data/data/adj_mat.pkl', 'wb') as f:
        pickle.dump([stations, station_id_to_ind, adj_mx], f, protocol=2)


def get_index(stations):
    station_id_to_ind = {}
    for i, station_id in enumerate(stations):
        station_id_to_ind[station_id] = i
    return station_id_to_ind


def get_dist_matrix(stations, geo_context, station_id_to_ind):

    # NOTE: Initial a distance matrix
    dist_mx = np.zeros((len(stations), len(stations)), dtype=np.float32)
    dist_mx[:] = np.inf

    # NOTE: Compute each element in the matrix
    for stations_i in stations:
        for stations_j in stations:
            i = station_id_to_ind[stations_i]
            j = station_id_to_ind[stations_j]
            if stations_i == stations_j:
                dist_mx[i][j] = 0.0
            else:
                dis = spatial.distance.euclidean(geo_context[i], geo_context[j])
                dist_mx[i][j] = dis

    dist_mx = np.array(dist_mx, dtype=np.float32)
    return dist_mx