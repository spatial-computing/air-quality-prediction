from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


class StandardScaler2:
    """
        Standard the whole input data
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def standard_scaler(df):
    scl = preprocessing.StandardScaler().fit(df)
    return scl.transform(df)


def get_k_means_label(df, k, max_iter=300):
    kmeans = KMeans(n_clusters=k, max_iter=max_iter).fit(df)
    labels = kmeans.labels_
    return labels


def random_forest_classifier(x, y, n_estimators=100, max_depth=10):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(x, y)
    return clf.feature_importances_


def random_forest_regressor(x, y, x_testing, n_estimators=100, max_depth=10):
    regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regr.fit(x, y)
    return regr.predict(x_testing)
