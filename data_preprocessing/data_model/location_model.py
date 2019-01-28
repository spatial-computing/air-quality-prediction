import pandas as pd

KEY_COL = 'station_id'
LON_COL = 'lon'
LAT_COL = 'lat'


class LocModel:

    def __init__(self, config, conn):
        self._config = config
        self.locations = self._get_locations(self._config, conn)

    @staticmethod
    def _get_locations(config, conn):

        table_name = config['table_name']
        column_set = config['column_set']

        data = conn.read(table_name, column_set, "")
        df = pd.DataFrame(data, columns=[KEY_COL, LON_COL, LAT_COL])
        df = df.drop_duplicates()
        return df


class LOCATIONS:

    def __init__(self, locations, ilocations):
        pass
