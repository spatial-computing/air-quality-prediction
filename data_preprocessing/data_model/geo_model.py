from utils.learning_lib import norm_to_zero_one

import pandas as pd
import numpy as np


GID_COL = 'gid'
GEO_FEATURE_COL = 'geo_feature'
FEATURE_TYPE_COL = 'feature_type'
BUFFER_SIZE_COL = 'buffer_size'
VALUE_COL = 'value'
COLUMN_SET = [GID_COL, GEO_FEATURE_COL, FEATURE_TYPE_COL, BUFFER_SIZE_COL, VALUE_COL]


class GeoModel:

    def __init__(self, locations, feature_set, config, conn):
        self._config = config
        self._geo_feature_set = feature_set

        geo_feature_table_name_dic = self._get_geo_feature_table_name(self._config['table_name_pr'])
        self._geo_feature_df = self._get_geo_feature(locations, geo_feature_table_name_dic, conn)
        self.geo_feature_vector_df = self._constructing_feature_vector(locations)
        self.geo_feature_name = self._get_feature_name()
        self.scaled_geo_feature_vector_df = self._scaling_feature_vector(locations)

    def _get_geo_feature_table_name(self, geo_feature_table_name_pr):
        geo_feature_table_name = {}
        for geo_feature in self._geo_feature_set:
            geo_feature_table_name[geo_feature] = geo_feature_table_name_pr + '_' + geo_feature
        return geo_feature_table_name

    def _get_feature_name(self):
        """
        Get geographic feature types from the index of geo_feature_vector

        :return: A list of geographic feature type names
        """
        geo_feature_name = self.geo_feature_vector_df.index
        return list(geo_feature_name)

    def _scaling_feature_vector(self, locations):
        """
        Scale geographic feature vector to (0,1)

        """
        scaled_geo_feature_vector = norm_to_zero_one(self.geo_feature_vector_df.T.values)
        scaled_geo_feature_vector_df = pd.DataFrame(scaled_geo_feature_vector.T,
                                                 index=self.geo_feature_name, columns=locations)
        return scaled_geo_feature_vector_df

    def _constructing_feature_vector(self, locations):
        """
        Construct the feature vector, indexed by distinct geo features, columned by locations

        :param locations: selected location
        :return: a DataFrame columned (available) station name, indexed by geographic feature types
        """

        df_grouped = self._geo_feature_df.groupby(GID_COL)

        self._geo_feature_df['feature_name'] = self._geo_feature_df[GEO_FEATURE_COL] + '_' \
                                               + self._geo_feature_df[FEATURE_TYPE_COL] + '_' \
                                               + self._geo_feature_df[BUFFER_SIZE_COL].map(str)
        feature_vector_list = []

        for each_group in df_grouped.groups:
            group = df_grouped.get_group(each_group)
            series = pd.Series(data=group[VALUE_COL].values, index=group['feature_name'], name=each_group)
            feature_vector_list.append(series)

        feature_vector = pd.concat(feature_vector_list, axis=1)

        # NOTE: In case there is no geographic features around certain locations, set value = 0.0
        for each_loc in locations:
            if each_loc not in feature_vector.columns:
                feature_vector[each_loc] = np.nan

        feature_vector = feature_vector.fillna(0.0)
        feature_vector = feature_vector[locations]
        return feature_vector

    def _get_geo_feature(self, locations, geo_feature_table_name_dic, conn):
        """
        Get all the geographic features for all locations from database

        :param locations: selected location
        :param geo_feature_table_name_dic: {geo_feature: related_table_name}
        :param conn: database connection
        :return: a DataFrame columned ["id", "geo_feature", "feature_type", "buffer_size", "value"]
        """

        additional_features = self._config['additional_features']
        geo_feature_df_list = []

        for geo_feature in self._geo_feature_set:
            this_geo_feature_table_name = geo_feature_table_name_dic[geo_feature]
            geo_feature_data = conn.read(this_geo_feature_table_name, COLUMN_SET)
            geo_feature_df = pd.DataFrame(geo_feature_data, columns=COLUMN_SET)
            # NOTE: filter the location based on the provided air quality locations
            geo_feature_df = geo_feature_df.loc[geo_feature_df[GID_COL].isin(locations)]
            geo_feature_df_list.append(geo_feature_df)

        for geo_feature in additional_features.keys():
            this_geo_feature_table_name = additional_features[geo_feature]['table_name']
            column_list = additional_features[geo_feature]['column_set']
            column_set = ['{} as '.format(column_list[0]) + GID_COL,
                          '\'location\' as ' + GEO_FEATURE_COL,
                          '\'{}\' as '.format(geo_feature) + FEATURE_TYPE_COL,
                          '0 as ' + BUFFER_SIZE_COL,
                          '{} as '.format(column_list[1]) + VALUE_COL]

            geo_feature_data = conn.read(this_geo_feature_table_name, column_set)
            geo_feature_df = pd.DataFrame(geo_feature_data, columns=COLUMN_SET)
            geo_feature_df.drop_duplicates(inplace=True)
            # NOTE: filter the location based on the provided air quality locations
            geo_feature_df = geo_feature_df.loc[geo_feature_df[GID_COL].isin(locations)]
            geo_feature_df_list.append(geo_feature_df)

        all_geo_feature_df = pd.concat(geo_feature_df_list)
        return all_geo_feature_df
