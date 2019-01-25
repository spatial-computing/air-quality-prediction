import math
import json

R = 6373.0


def geo_distance(lon1, lat1, lon2, lat2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def load_json(file_path):
    try:
        json_data = open(file_path).read()
        data = json.loads(json_data)
        return data
    except IOError as e:
        print('{}. Fail to load {} json file.'.format(e, file_path))
        return False


def write_csv(data, file_path):
    try:
        data.to_csv(file_path, header=True, index=False, sep=',', mode='w')
        return True
    except IOError as e:
        print('{}. Fail to write csv file.'.format(e))
        return False

