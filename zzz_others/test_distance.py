from scipy.spatial.distance import euclidean


def compute_time_series_distance(air_quality_time_series, air_quality_location):
    air_quality_time_series = air_quality_time_series.fillna(0.0)
    for each_location_a in air_quality_location:
        for each_location_b in air_quality_location:
            dis = euclidean(air_quality_time_series[each_location_a],
                            air_quality_time_series[each_location_b])
            print(each_location_a, each_location_b, dis)