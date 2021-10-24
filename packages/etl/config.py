import json


# db credentials {"hostname": "***", "port":"", "login":, "password":""}
with open('./credentials/mysql_user.json', encoding="utf8") as file:
    config = json.load(file)
ssl_args = {'ssl_ca': './credentials/CA.pem'}

# paths to source data
PATH_STATION_COORDINATES = './IN/02_координаты_станций.csv'
PATH_STATION_METRICS = './IN/01_данные станций'
PATH_PROFILE = './IN/04_данные_профилемера'

SCHEMA_RAW = 'raw'
SCHEMA_FILTERED = 'filtered'
SCHEMA_PROD = 'production'
TABLE_NAME_COO = 'station_coordinates'
TABLE_NAME_METRICS = 'station_metrics'
TABLE_NAME_METRICS_ML = 'station_metrics_ml'
TABLE_NAME_TEMPERATURE = 'out_temperature'
TABLE_NAME_PROFILE = 'profile_metrics'
TABLE_NAME_ALERTS = 'incidents'

# conditions to raise incident
ALERT_CONFIG = {
    'notnull_metrics': ['pressure'],  # alert if metric <= 0
    'positive_metrics': ['CO', 'NO', 'NO2', 'PM2.5', 'PM10', 'humidity', 'precipitation', '| V |', '_V_'],  # alert if metric < 0
    'upper_alert': ['pressure', '| V |', 'CO', 'NO', 'NO2', 'PM2.5', 'PM10'],  # alert on upper percentile
    'upper_percentiles': [98, 99, 99.9, 100],  # alert on this percentile ranges
    'lower_alert': ['pressure'],  # alert on lower percentile
    'lower_percentiles': [0, 0.1, 1, 2]  # alert on this percentile ranges
}

# stations that not in "02_координаты_станций.csv"
EXTRA_STATIONS = {
    'station_name': 'Высотный пункт 253 м ветер'
}