import pandas as pd


KEY_COL = 'station_id'
TIME_COL = 'date_observed'
VALUE_COL = 'value'


class AirModel:

    def __init__(self, config, conn):
        self._config = config
        self._raw_air_quality_df = self._get_air_quality(conn)
        self.air_quality_df = self._df_simple_cleaning(self._raw_air_quality_df)
        self.time_series = None

    def _get_air_quality(self, conn):
        """
        Read air quality data.
        Construct air quality data to a DataFrame, columned by [KEY_COL, TIME_COL, VALUE_COL]

        """

        table_name = self._config['table_name']
        column_set = self._config['column_set']
        request_condition = self._config['request_condition']
        data = conn.read(table_name, column_set, request_condition)
        df = pd.DataFrame(data, columns=[KEY_COL, TIME_COL, VALUE_COL])
        return df

    @staticmethod
    def _df_simple_cleaning(df):
        """
        Simply filter by values
        Step1: Filter out the duplicates
        Step2: Filter out all negative values

        :param df: input data
        :return: a simply cleaned DataFrame
        """
        cleaned_df = df.drop_duplicates()
        cleaned_df = cleaned_df[cleaned_df[VALUE_COL] > 0.0]
        return cleaned_df

    def remove_stations(self, removers):
        """
        Get a new air quality Dataframe without given removers

        """
        self.air_quality_df = self.air_quality_df[~self.air_quality_df[KEY_COL].isin(removers)]

    def select_stations(self, stations):
        """
        Get a new air quality Dataframe given certain stations

        """
        self.air_quality_df = self.air_quality_df[self.air_quality_df[KEY_COL].isin(stations)]

    def get_locations(self):
        """
        Get distinct locations from a air quality Dataframe

        """
        locations = self.air_quality_df[KEY_COL].drop_duplicates()
        return list(locations)
