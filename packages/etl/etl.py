import json
import os
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timezone

from tools import *


with open('./credentials/mysql_user.json', encoding="utf8") as file:
    config = json.load(file)

SCHEMA_RAW = 'raw'
SCHEMA_FILTERED = 'filtered'
SCHEMA_PROD = 'production'
TABLE_NAME_COO = 'station_coordinates'
TABLE_NAME_METRICS = 'station_metrics'
TABLE_NAME_TEMPERATURE = 'out_temperature'
TABLE_NAME_PROFILE = 'profile_metrics'

PATH_STATION_COORDINATES = './IN/02_координаты_станций.csv'
PATH_STATION_METRICS = './IN/01_данные станций'
PATH_PROFILE = './IN/04_данные_профилемера'

ssl_args = {'ssl_ca': './credentials/CA.pem'}
db_connection_str = f"mysql+pymysql://{config['login']}:{config['password']}@{config['hostname']}:{config['port']}/{SCHEMA_RAW}"
db_connection = create_engine(db_connection_str, connect_args=ssl_args)

logging.getLogger().setLevel('INFO')


def upload_coordinates(path_to_file):
    """
    Upload stations coordinates to db
    path_to_file: path to 02_координаты_станций.csv
    """
    logging.info('Processing station coordinates')
    df_coor = pd.read_csv(path_to_file, delimiter=';', header=None,
                          names=['station_name', 'latitude', 'longitude'], dtype=str)
    df_coor['created_at'] = datetime.now(tz=timezone.utc)
    df_to_pg(df_coor, TABLE_NAME_COO, SCHEMA_RAW, pk='clear table', engine=db_connection)
    logging.info('Done!')


def upload_station_metrics(path_to_directory):
    """
    Upload station metrics to db
    path_to_directory: path to directory 01_данные станций
    """
    logging.info('Processing station metrics')
    renames = {
        'Дата и время': 'report_dt',
        'PM25': 'PM2.5',
        'Давление': 'pressure',
        'Влажность': 'humidity',
        'Осадки': 'precipitation'
    }

    df_station_metrics = pd.DataFrame()
    for filename in os.listdir(path_to_directory):
        logging.info(filename)
        source, file_extension = os.path.splitext(filename)
        if file_extension not in ['.xlsx', '.xls']:
            logging.info('Not a Excel file. Skipping...')
            continue
        engine_excel = 'openpyxl' if file_extension == '.xlsx' else None
        df_input_i = pd.read_excel(os.path.join(path_to_directory, filename), engine=engine_excel)

        df_input_i = df_input_i[list(filter(lambda x: 'Unnamed' not in x, df_input_i.columns))]
        df_input_i = df_input_i.rename(columns=renames).set_index('report_dt')
        df_input_i = df_input_i[~df_input_i.index.isnull()]

        df_flat = pd.DataFrame()
        for column in df_input_i.columns:
            df_flat_i = df_input_i[[column]].rename(columns={column: 'metric_value'})
            df_flat_i['metric_name'] = column
            df_flat = df_flat.append(df_flat_i.reset_index(), ignore_index=False)

        df_flat['filename'] = source
        df_station_metrics = df_station_metrics.append(df_flat)

    df_station_metrics['created_at'] = datetime.now(tz=timezone.utc)

    # Parse dates, convert timestamp to UTC timezone
    dates1 = pd.to_datetime(df_station_metrics.report_dt, format='%d/%m/%Y %H:%M', errors='coerce')
    dates2 = pd.to_datetime(df_station_metrics.report_dt, errors='coerce')
    df_station_metrics['report_dt_parsed'] = dates1.combine_first(dates2)
    df_station_metrics['report_dt_parsed'] = df_station_metrics.report_dt_parsed.dt.tz_localize(
        'Europe/Moscow').dt.tz_convert('UTC')
    df_station_metrics = df_station_metrics.drop(columns='report_dt').rename(
        columns={'report_dt_parsed': 'report_dt'}).copy()

    # Insert to database
    df_to_pg(df_station_metrics, TABLE_NAME_METRICS, SCHEMA_RAW, pk='clear table', engine=db_connection)
    logging.info('Done!')


def upload_profile(path_to_directory):
    """
    Upload station metrics to db
    path_to_directory: path to directory 04_данные_профилемера
    """
    logging.info('Processing profiler')
    df_out_temperature = pd.DataFrame()
    df_profile = pd.DataFrame()

    for filename in os.listdir(path_to_directory):
        logging.info(f'Parsing file {filename}')
        with open(os.path.join(path_to_directory, filename), 'r') as file:
            lines = file.readlines()

            # Find key line index
            for index, line in enumerate(lines):
                if 'End Of Commentary' in line:
                    break

        # Parse file
        df_profile_i = pd.read_csv(os.path.join(path_to_directory, filename), skiprows=index + 3, delimiter='\t',
                                   decimal=b',')

        # Cut OutsideTemperature to another df
        df_temperature_i = df_profile_i[['data time', 'OutsideTemperature', 'Quality']] \
            .rename(columns={'OutsideTemperature': 'outside_temperature', 'Quality': 'quality', 'data time': 'report_dt'}) \
            .assign(filename=filename).copy()
        df_out_temperature = df_out_temperature.append(df_temperature_i, sort=False)

        # Cut profile
        df_profile_i = df_profile_i.drop(columns=['OutsideTemperature', 'Quality']) \
            .rename(columns={'data time': 'report_dt'}) \
            .set_index('report_dt')

        # Flatting table
        df_profile_i_merge = pd.DataFrame()
        for c in df_profile_i.columns:
            df_altitude = df_profile_i[[c]].copy()
            df_altitude.set_axis(['temperature'], axis=1, inplace=True)
            df_altitude['altitude'] = c

            df_profile_i_merge = df_profile_i_merge.append(df_altitude, sort=False)
        df_profile_i_merge['filename'] = filename
        df_profile = df_profile.append(df_profile_i_merge, sort=False)

    # Parse dates, convert timestamp to UTC timezone
    logging.info('Parsing dates...')
    df_out_temperature['report_dt'] = pd.to_datetime(df_out_temperature.report_dt, format='%d/%m/%Y %H:%M:%S')
    df_out_temperature['report_dt'] = df_out_temperature.report_dt.dt.tz_localize('Europe/Moscow').dt.tz_convert('UTC')

    df_profile = df_profile.reset_index().copy()
    df_profile['report_dt'] = pd.to_datetime(df_profile.report_dt, format='%d/%m/%Y %H:%M:%S')
    df_profile['report_dt'] = df_profile.report_dt.dt.tz_localize('Europe/Moscow').dt.tz_convert('UTC')

    # created_at timestamp
    df_out_temperature['created_at'] = datetime.now(tz=timezone.utc)
    df_profile['created_at'] = datetime.now(tz=timezone.utc)

    # profiler sourse. In DEMO we have only Останкино.
    df_out_temperature['source'] = 'Останкино'
    df_profile['source'] = 'Останкино'

    # insert to database
    df_to_pg(df_out_temperature, TABLE_NAME_TEMPERATURE, SCHEMA_RAW, pk='clear table', engine=db_connection)
    df_to_pg(df_profile, TABLE_NAME_PROFILE, SCHEMA_RAW, pk='clear table', engine=db_connection)
    logging.info('Done!')


def update_prod_station():
    """
    Update prod station coordinates if new found in raw
    """
    logging.info('UPDATING PRODUCTION station coordinates')
    dt = datetime.now(tz=timezone.utc).strftime('%m-%d-%Y %H:%M:%S')
    query = f"""
        INSERT INTO {SCHEMA_PROD}.{TABLE_NAME_COO}
            (station_name, latitude, longitude, created_at)
        SELECT station_name, 
               latitude, 
               longitude, 
               STR_TO_DATE('{dt}','%%m-%%d-%%Y %%H:%%i:%%s') as created_at
        FROM {SCHEMA_RAW}.{TABLE_NAME_COO}
        WHERE station_name NOT IN (
            SELECT station_name 
            FROM {SCHEMA_PROD}.{TABLE_NAME_COO}
        )
    """

    with db_connection.connect() as con:
        con.execute(query)

    logging.info('Done!')


def update_filtered_metrics():
    """
    Matching station_id by filename for station metrics
    Updating filtered.station_metrics
    """
    logging.info('Updating filtered station metrics')

    logging.info('Getting data from raw...')
    station_ids = pd.read_sql(f'SELECT station_name, id FROM {SCHEMA_PROD}.{TABLE_NAME_COO}', con=db_connection)
    df_metric = pd.read_sql(f'SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME_METRICS}', con=db_connection) # TO DO: update only new rows

    # Mapping

    logging.info('Mapping stations...')
    station_ids['station_name'] = station_ids['station_name'].str.lower()
    station_ids = {i['station_name']: i['id'] for _, i in station_ids.iterrows()}  # {station_name: id}

    df_mapping = df_metric[['filename']].drop_duplicates()
    df_mapping['id'] = df_mapping['filename'].apply(lambda x: map_id(x, station_ids))  # {filename: id}

    df_metric = df_metric.merge(df_mapping, how='left', left_on='filename', right_on='filename')\
                         .rename(columns={'id': 'station_id'})\
                         .drop(columns=['filename', 'created_at'])
    df_metric['created_at'] = datetime.now(tz=timezone.utc)

    df_to_pg(df_metric, TABLE_NAME_METRICS, SCHEMA_FILTERED, pk='clear table', engine=db_connection)
    logging.info('Done!')


def update_filtered_profile():
    """
    Matching station_id by filename for station metrics
    Updating filtered.station_metrics
    """
    logging.info('Updating filtered station profiler')

    logging.info('Getting data from raw...')
    station_ids = pd.read_sql(f'SELECT station_name, id FROM {SCHEMA_PROD}.{TABLE_NAME_COO}', con=db_connection)
    df_profile = pd.read_sql(f'SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME_PROFILE}', con=db_connection) # TO DO: update only new rows
    df_temperature = pd.read_sql(f'SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME_TEMPERATURE}', con=db_connection)

    # Mapping
    logging.info('Mapping stations...')
    station_ids['station_name'] = station_ids['station_name'].str.lower()
    station_ids = {i['station_name']: i['id'] for _, i in station_ids.iterrows()}  # {station_name: id}

    df_mapping = df_profile[['source']].append(df_temperature[['source']]).drop_duplicates()
    df_mapping['id'] = df_mapping['source'].apply(lambda x: map_id(x, station_ids))  # {source: id}

    df_profile = df_profile.merge(df_mapping, how='left', left_on='source', right_on='source')\
                         .rename(columns={'id': 'station_id'})\
                         .drop(columns=['source', 'created_at', 'filename'])
    df_temperature = df_temperature.merge(df_mapping, how='left', left_on='source', right_on='source')\
                         .rename(columns={'id': 'station_id'})\
                         .drop(columns=['source', 'created_at', 'filename'])

    df_profile['created_at'] = datetime.now(tz=timezone.utc)
    df_temperature['created_at'] = datetime.now(tz=timezone.utc)

    df_to_pg(df_profile, TABLE_NAME_PROFILE, SCHEMA_FILTERED, pk='clear table', engine=db_connection)
    df_to_pg(df_temperature, TABLE_NAME_TEMPERATURE, SCHEMA_FILTERED, pk='clear table', engine=db_connection)
    logging.info('Done!')


def main():
    upload_coordinates(PATH_STATION_COORDINATES)
    upload_station_metrics(PATH_STATION_METRICS)
    upload_profile(PATH_PROFILE)

    update_prod_station()
    update_filtered_metrics()
    update_filtered_profile()


if __name__ == "__main__":
    main()

