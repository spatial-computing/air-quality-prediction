from lib.libs import *

import numpy as np
import matplotlib.pyplot as plt


def cluster_main(air_quality_model):

    time_series = air_quality_model.time_series

    # NOTE: Scale the air quality data according to the mean and standard deviation
    air_scaler = StandardScaler2(
        mean=time_series.values[~np.isnan(time_series.values)].mean(),
        std=time_series.values[~np.isnan(time_series.values)].std())
    training_time_series = air_scaler.transform(time_series)
    training_time_series_dropna = training_time_series.dropna()
    get_best_k(training_time_series_dropna)
    return


def get_best_k(time_series):

    distortions = []
    K = range(1, len(time_series.columns) + 1, 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, max_iter=300).fit(time_series)

        # Sum of squared distances of samples to their closest cluster center
        err = kmeans.inertia_
        distortions.append(err)

    # Plot the elbow
    plt.figure(figsize=(15, 20))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(' The Elbow Method showing the optimal k')
    plt.show()


def get_best_k_cv(air_quality_model):
    """
    Specifically for leave one out CV algorithm

    :param air_quality_model:
    :return: None
    """

    locations = air_quality_model.air_quality_locations
    time_series = air_quality_model.air_quality_time_series

    for each_location in locations:

        other_locations = [i for i in locations if i != each_location]
        training_time_series = time_series[other_locations]
        scaled_training_time_series = air_quality_model.scaler.transform(training_time_series)
        training_time_series_dropna = scaled_training_time_series.dropna().T

        # k means determine k
        distortions = []
        K = range(1, len(other_locations) + 1, 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, max_iter=300).fit(training_time_series_dropna)
            # err = sum(np.min(cdist(training_time_series_dropna, kmeans.cluster_centers_, 'euclidean'), axis=1)) \
            #       / training_time_series_dropna.shape[0]

            # Sum of squared distances of samples to their closest cluster center
            err = kmeans.inertia_
            distortions.append(err)
            print(k, dict(zip(other_locations, kmeans.labels_)))
            print(each_location, k, 'err=', err)

        # Plot the elbow
        plt.figure(figsize=(15, 20))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title(str(each_location) + ' The Elbow Method showing the optimal k')
        plt.show()

