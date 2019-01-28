import math
import json

R = 6373.0






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

