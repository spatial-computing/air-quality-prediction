from lib.libs import random_forest_classifier, get_k_means_label


def get_feature_importance(time_series, geo_feature_vector, geo_feature_name, config):
    """
    Cluster the time series to get their label
    Classify the label with the geographic features
    Generate important features list and sorted by their importance

    :param time_series: Input air quality time series
    :param geo_feature_vector: geographic feature vector
    :param geo_feature_name: geographic feature model
    :param config:
    :return:
    """
    rf_classifier_tree_num = config['rf_classifier_tree_num']
    rf_classifier_tree_depth = config['rf_classifier_tree_depth']
    geo_feature_percent = config['geo_feature_percent']
    n_cluster = config['n_cluster']

    # NOTE: Drop NaNs for clustering
    time_series_dropna = time_series.dropna()
    label = get_k_means_label(time_series_dropna.T, n_cluster)

    # NOTE: Compute feature importance and important features
    feature_importance = random_forest_classifier(geo_feature_vector.T, label,
                                                  rf_classifier_tree_num, rf_classifier_tree_depth)

    important_feature_list, sorted_important_feature = \
        get_important_feature_name_with_percent(geo_feature_name, feature_importance, percent=geo_feature_percent)

    return important_feature_list, sorted_important_feature


def get_important_feature_name(feature_name, importance):
    """
    Get feature names whose importance != 0.0

    :param feature_name: a list of feature names
    :param importance: a list of importance according to the feature names
    :return: a list of important features, sorted important features with its importance as a dict
    """
    feature_dic = dict(zip(feature_name, importance))
    important_feature_dic = {k: v for k, v in feature_dic.items() if v != 0.0}
    sorted_important_feature = [(k, important_feature_dic[k])
                                for k in sorted(important_feature_dic, key=important_feature_dic.get, reverse=True)]
    important_feature_list = list(important_feature_dic.keys())
    return important_feature_list, sorted_important_feature


def get_important_feature_name_with_percent(feature_name, importance, percent=0.1):
    """
    Get top "percent" of feature names based on importance

    :param feature_name: a list of feature names
    :param importance: a list of importance according to the feature names
    :param percent: percentage of the features
    :return: a list of important features, sorted important features with its importance as a dict
    """
    feature_dic = dict(zip(feature_name, importance))
    sorted_feature = [(k, feature_dic[k])
                      for k in sorted(feature_dic, key=feature_dic.get, reverse=True)]
    num_important_feature = int(len(sorted_feature) * percent)
    sorted_important_feature = sorted_feature[:num_important_feature]
    important_feature_list = [k for (k, v) in sorted_important_feature]
    return important_feature_list, sorted_important_feature

