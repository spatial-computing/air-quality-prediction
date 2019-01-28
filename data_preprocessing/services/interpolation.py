from data_preprocessing.services.postgres_connection import Connection

import math
import pandas as pd


R = 6373.0
KEY_COL = 'station_id'
TIME_COL = 'date_observed'
LON_COL = 'lon'
LAT_COL = 'lat'


def get_interpolated_values(met_model, loc_coord, i_loc_coord):
    loc_coord_df = loc_coord.locations
    i_loc_coord_df = i_loc_coord.locations
    i_locations = list(i_loc_coord_df[KEY_COL])

    weights, sum_weights = get_weights(loc_coord_df, i_loc_coord_df)

    met_df = met_model.met_df
    VALUE_COLS = met_model.VALUE_COLS
    i_df = invere_distance_weighted(met_df, i_locations, weights, sum_weights, VALUE_COLS)


    return i_df, i_locations


def invere_distance_weighted(df, i_locations, weights, sum_weights, VALUE_COLS):

    output_list = []

    for loc in i_locations:
        loc_weight = weights[loc]
        loc_weight_df = df.join(loc_weight, on=KEY_COL)
        loc_weight_df = loc_weight_df[VALUE_COLS].mul(loc_weight_df[loc], axis=0)
        loc_weight_df[TIME_COL] = df[TIME_COL]
        i_loc_df = loc_weight_df[VALUE_COLS + [TIME_COL]].groupby([TIME_COL]).sum()
        i_loc_df = i_loc_df / sum_weights[loc]
        i_loc_df[KEY_COL] = loc
        i_loc_df[TIME_COL] = i_loc_df.index
        i_loc_df.reset_index(drop=True, inplace=True)
        output_list.append(i_loc_df)

    i_df = pd.concat(output_list)
    return i_df


def geo_distance(lon1, lat1, lon2, lat2):

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_weights(loc_coord_df, i_loc_coord_df):

    locations = list(loc_coord_df[KEY_COL])
    i_locations = list(i_loc_coord_df[KEY_COL])

    weights = pd.DataFrame(index=locations, columns=i_locations)
    for a in locations:
        for b in i_locations:
            a_lon = loc_coord_df[loc_coord_df[KEY_COL] == a][LON_COL].values[0]
            a_lat = loc_coord_df[loc_coord_df[KEY_COL] == a][LAT_COL].values[0]
            b_lon = i_loc_coord_df[i_loc_coord_df[KEY_COL] == b][LON_COL].values[0]
            b_lat = i_loc_coord_df[i_loc_coord_df[KEY_COL] == b][LAT_COL].values[0]
            dis = geo_distance(a_lon, a_lat, b_lon, b_lat)
            weight = 1 / dis if dis != 0.0 else 1.0
            weights.loc[a, b] = weight
    sum_weights = weights.sum(axis=0)
    # weights[KEY_COL] = weights.index
    return weights, sum_weights


