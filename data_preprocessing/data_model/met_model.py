from utils.learning_lib import norm_to_zero_one

import pandas as pd


KEY_COL = 'station_id'
TIME_COL = 'date_observed'


class MetModel:

    def __init__(self, locations, config, conn, met_df=None):
        self.config = config
        self.VALUE_COLS = self.config['column_set'][2:]
        self.locations = locations

        if met_df is None:
            self.met_df = self._get_met_data(locations, conn)
        else:
            self.met_df = met_df

        self.scaled_met_df = self._scaling(self.met_df)
        self.time_series = None

    def _get_met_data(self, locations, conn):

        table_name = self.config['table_name']
        column_set = self.config['column_set']
        request_condition = self.config['request_condition']

        data = conn.read(table_name, column_set, request_condition)
        df = pd.DataFrame(data, columns=[KEY_COL, TIME_COL] + self.VALUE_COLS)
        df = df.loc[df[KEY_COL].isin(locations)]
        return df

    def _scaling(self, df):
        """
        Scale meterological features to (0,1)

        """
        scaled_data = norm_to_zero_one(df[self.VALUE_COLS].values)
        scaled_df = pd.DataFrame(scaled_data, columns=self.VALUE_COLS)
        scaled_df[KEY_COL] = list(df[KEY_COL])
        scaled_df[TIME_COL] = list(df[TIME_COL])

        return scaled_df

