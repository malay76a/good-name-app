import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from scipy.optimize import minimize
from sqlalchemy import create_engine

from config import *


def get_connection():
    db_connection_str = f"mysql+pymysql://{config['login']}:{config['password']}@{config['hostname']}:{config['port']}/{SCHEMA_RAW}"
    db_connection = create_engine(db_connection_str, connect_args=ssl_args)
    return db_connection


def df_to_pg(df, table_name, schema, pk, engine, attempts=1):
    """
    Insert df to table_name deleting outdated data by pk
    :param df: input df
    :param table_name: table to insert
    :param schema: table schema
    :param pk: delete from table where pk in input df.
        pk = 'clear table' : delete all data from table
        pk = None : do nothing
    :param engine: connection engine
    :param attempts: number of retry
    :return:
    """
    if pk is not None and pk.lower() == 'clear table':
        logging.info(f'All data will be deleted from {table_name}')
        delete_query = """DELETE FROM {table_name}""" \
            .format(table_name=schema + '.' + table_name)
    elif pk:
        logging.info(f'Data with be deleted from {table_name} by pk {pk}')
        delete_query = ("""
            DELETE FROM 
                {table_name} 
            WHERE 
                "{pk}" IN ('{pk_list}')
            """.format(
            pk=pk,
            pk_list="', '".join([str(i) for i in df[pk].unique()]),
            table_name=schema + '.' + table_name
        )
        )
    else:
        delete_query = None

    # INSERT
    for t in range(attempts):
        with engine.connect() as con:
            try:
                if delete_query:
                    logging.info(f'Deleting obsolete data from {schema}.{table_name}')
                    con.execute(delete_query)

                logging.info(f'Inserting to {schema}.{table_name}')
                df.to_sql(name=table_name, schema=schema, con=engine, if_exists='append', index=False, chunksize=5000)
                logging.info(f'Inserting to {schema}.{table_name} was successful!')
                break
            except Exception as e:
                logging.warning(f'Attempt {t + 1} failed. Reason: {e}')


def map_id(target_string: str, substring_to_id: dict):
    """
    Find in target string substring from substring_to_id keys and return its id
    :param target_string: any_string
    :param substring_to_id: {substing: unique_id}
    :return: id of matched substring
    """
    for key in substring_to_id.keys():
        if key[:-1] in target_string.lower().replace('.', ' '):  # [:-1] awesome crutch for extra whitespace in the end
            return substring_to_id[key]


# ----------------------------------------------------------------------------------------------------------------------
# predictions
# ----------------------------------------------------------------------------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def timeseriesCVscore(params, series, loss_function, slen):
    """
        Returns error on CV

        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
    """
    # errors array
    errors = []

    values = series.values
    alpha, beta, gamma = params

    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):

        model = HoltWinters(series=values[train], slen=slen,
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)

    return abs(np.mean(np.array(errors)))


class HoltWinters:

    """
    Holt-Winters model with the anomalies detection using Brutlag method

    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)

    """


    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        if slen > int(len(series) / 2):
            self.slen = int(len(series) / 2) - 1

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor


    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals


    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(self.result[0] +
                                      self.scaling_factor *
                                      self.PredictedDeviation[0])

                self.LowerBond.append(self.result[0] -
                                      self.scaling_factor *
                                      self.PredictedDeviation[0])
                continue

            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i])
                                               + (1-self.gamma)*self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] +
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] -
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


def plotHoltWinters(series, model, plot_intervals=False, plot_anomalies=False):
    """
        series - dataset with timeseries
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """

    plt.figure(figsize=(20, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(series.values, label = "Actual")
    error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

    if plot_anomalies:
        anomalies = np.array([np.NaN]*len(series))
        anomalies[series.values<model.LowerBond[:len(series)]] = \
            series.values[series.values<model.LowerBond[:len(series)]]
        anomalies[series.values>model.UpperBond[:len(series)]] = \
            series.values[series.values>model.UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    if plot_intervals:
        plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(model.LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond,
                         y2=model.LowerBond, alpha=0.2, color = "grey")

    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);


# ----------------------------------------------------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------------------------------------------------
def upload_coordinates(path_to_file):
    """
    Upload stations coordinates to db
    path_to_file: path to 02_координаты_станций.csv
    """
    db_connection = get_connection()
    logging.info('Processing station coordinates')
    df_coor = pd.read_csv(path_to_file, delimiter=';', header=None,
                          names=['station_name', 'latitude', 'longitude'], dtype=str)
    df_coor = df_coor.append(EXTRA_STATIONS, ignore_index=True)
    df_coor['created_at'] = datetime.now(tz=timezone.utc)
    df_to_pg(df_coor, TABLE_NAME_COO, SCHEMA_RAW, pk='clear table', engine=db_connection)
    logging.info('Done!')


def upload_station_metrics(path_to_directory):
    """
    Upload station metrics to db
    path_to_directory: path to directory 01_данные станций
    """
    db_connection = get_connection()
    logging.info('Processing station metrics')
    renames = {
        'Дата и время': 'report_dt',
        'PM25': 'PM2.5',
        'Давление': 'pressure',
        'Влажность': 'humidity',
        'Осадки': 'precipitation',
        'Направление ветра': '_V_',
        'Скорость ветра': '| V |'
    }

    df_station_metrics = pd.DataFrame()
    for filename in os.listdir(path_to_directory):
        logging.info(filename)
        source, file_extension = os.path.splitext(filename)
        if file_extension not in ['.xlsx', '.xls']:
            logging.info('Not a Excel file. Skipping...')
            continue
        if filename == 'Высотный пункт 253 м ветер.xls':
            df_input_i = pd.read_excel(os.path.join(PATH_STATION_METRICS, filename), engine=None, skiprows=1)
            df_input_i.rename(columns={'Unnamed: 0': 'report_dt'}, inplace=True)
        else:
            df_input_i = pd.read_excel(os.path.join(PATH_STATION_METRICS, filename), engine='openpyxl')

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
    logging.info('Parsing...')
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
    db_connection = get_connection()
    logging.info('Processing profiler')
    df_out_temperature = pd.DataFrame()
    df_profile = pd.DataFrame()

    for filename in os.listdir(path_to_directory):
        logging.info(f'Parsing file {filename}')
        with open(os.path.join(path_to_directory, filename), 'r') as f:
            lines = f.readlines()

            # Find key line index
            for index, line in enumerate(lines):
                if 'End Of Commentary' in line:
                    break

        # Parse file
        df_profile_i = pd.read_csv(os.path.join(path_to_directory, filename), skiprows=index + 3, delimiter='\t',
                                   decimal=b',')

        # Cut OutsideTemperature to another df
        df_temperature_i = df_profile_i[['data time', 'OutsideTemperature', 'Quality']] \
            .rename(
            columns={'OutsideTemperature': 'outside_temperature', 'Quality': 'quality', 'data time': 'report_dt'}) \
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

    # profiler source. In DEMO we have only Останкино.
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
    db_connection = get_connection()
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
        result = con.execute(query)
        logging.info(f"Inserted {result.rowcount} rows")

    logging.info('Done!')


def update_filtered_metrics():
    """
    Matching station_id by filename for station metrics
    Updating filtered.station_metrics
    """
    db_connection = get_connection()
    logging.info('Updating filtered station metrics')

    logging.info('Getting data from raw...')
    station_ids = pd.read_sql(f'SELECT station_name, id FROM {SCHEMA_PROD}.{TABLE_NAME_COO}', con=db_connection)
    df_metric = pd.read_sql(f'SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME_METRICS}', con=db_connection)

    # Mapping
    logging.info('Mapping stations...')
    station_ids['station_name'] = station_ids['station_name'].str.lower()
    station_ids = {i['station_name']: i['id'] for _, i in station_ids.iterrows()}  # {station_name: id}

    df_mapping = df_metric[['filename']].drop_duplicates()
    df_mapping['id'] = df_mapping['filename'].apply(lambda x: map_id(x, station_ids))  # {filename: id}

    df_metric = df_metric.merge(df_mapping, how='left', left_on='filename', right_on='filename') \
        .rename(columns={'id': 'station_id'}) \
        .drop(columns=['filename', 'created_at'])
    df_metric['created_at'] = datetime.now(tz=timezone.utc)

    df_to_pg(df_metric, TABLE_NAME_METRICS, SCHEMA_FILTERED, pk='clear table', engine=db_connection)
    logging.info('Done!')


def update_filtered_profile():
    """
    Matching station_id by filename for station metrics
    Updating filtered.station_metrics
    """
    db_connection = get_connection()
    logging.info('Updating filtered station profiler')

    logging.info('Getting data from raw...')
    station_ids = pd.read_sql(f'SELECT station_name, id FROM {SCHEMA_PROD}.{TABLE_NAME_COO}', con=db_connection)
    df_profile = pd.read_sql(f'SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME_PROFILE}', con=db_connection)
    df_temperature = pd.read_sql(f'SELECT * FROM {SCHEMA_RAW}.{TABLE_NAME_TEMPERATURE}', con=db_connection)

    # Mapping
    logging.info('Mapping stations...')
    station_ids['station_name'] = station_ids['station_name'].str.lower()
    station_ids = {i['station_name']: i['id'] for _, i in station_ids.iterrows()}  # {station_name: id}

    df_mapping = df_profile[['source']].append(df_temperature[['source']]).drop_duplicates()
    df_mapping['id'] = df_mapping['source'].apply(lambda x: map_id(x, station_ids))  # {source: id}

    df_profile = df_profile.merge(df_mapping, how='left', left_on='source', right_on='source') \
        .rename(columns={'id': 'station_id'}) \
        .drop(columns=['source', 'created_at', 'filename'])
    df_temperature = df_temperature.merge(df_mapping, how='left', left_on='source', right_on='source') \
        .rename(columns={'id': 'station_id'}) \
        .drop(columns=['source', 'created_at', 'filename'])

    df_profile['created_at'] = datetime.now(tz=timezone.utc)
    df_temperature['created_at'] = datetime.now(tz=timezone.utc)

    df_to_pg(df_profile, TABLE_NAME_PROFILE, SCHEMA_FILTERED, pk='clear table', engine=db_connection)
    df_to_pg(df_temperature, TABLE_NAME_TEMPERATURE, SCHEMA_FILTERED, pk='clear table', engine=db_connection)
    logging.info('Done!')


def define_anomaly(dt_from: datetime, dt_to: datetime):
    """
    Get anomalies in station_metrics for specified date range
    :param dt_from: find anomalies from this date
    :param dt_to: to this date
    :return:
    """
    db_connection = get_connection()
    logging.info(f'Defining anomalies in range {dt_from} - {dt_to}')

    logging.info('Getting data from filtered...')
    query = f"""
        SELECT report_dt,
               station_id,
               metric_name,
               metric_value
        FROM {SCHEMA_FILTERED}.{TABLE_NAME_METRICS}
    """
    df_all = pd.read_sql(query, con=db_connection)
    df_window = df_all[(df_all.report_dt > dt_from) & (df_all.report_dt <= dt_to)]

    df_alerts = pd.DataFrame(columns=['report_dt', 'station_id', 'source', 'metric_name', 'metric_value',
                                      'incident_type', 'incident_level', 'is_alert_send'])

    # TO DO: Проверку целостности данных, как и сверку с ПДК можно вынести в отдельную функцию, ибо не нужен весь датасет
    # NULL ALERT
    alert_i = df_window[(df_window.metric_name.isin(ALERT_CONFIG['notnull_metrics'])) & (df_window.metric_value == 0)]
    alert_i = alert_i.assign(incident_type='null_value', source='station_metrics')
    logging.info(f'{alert_i.shape[0]} incidents with zero values')
    df_alerts = df_alerts.append(alert_i)

    # NEGATIVE ALERT
    alert_i = df_window[(df_window.metric_name.isin(ALERT_CONFIG['positive_metrics'])) &
                        (df_window.metric_value < 0)]
    alert_i = alert_i.assign(incident_type='negative_value', source='station_metrics')
    logging.info(f'{alert_i.shape[0]} incidents with negative values')
    df_alerts = df_alerts.append(alert_i)

    # TO DO: Проверку, что значения на приборе перестали приходить (но приходили раньше)
    # PERCENTILE ALERTS
    df_all = df_all[((df_all.metric_name.isin(ALERT_CONFIG['notnull_metrics'])) & (df_all.metric_value > 0)) |
                    (df_all.metric_name.isin(ALERT_CONFIG['positive_metrics'])) & (df_all.metric_value >= 0)].copy()

    # upper percentile
    for metric_i in ALERT_CONFIG['upper_alert']:
        df_all_segment = df_all[df_all.metric_name == metric_i].copy()

        percentiles_vals = [np.percentile(df_all_segment.metric_value.sort_values().to_list(), p)
                            for p in ALERT_CONFIG['upper_percentiles']]

        alert_i = pd.DataFrame()
        for p in range(len(percentiles_vals) - 1):
            alert_p_i = df_window[(df_window.metric_name == metric_i) &
                                  (df_window.metric_value > percentiles_vals[p]) &
                                  (df_window.metric_value <= percentiles_vals[p + 1])]
            alert_p_i = alert_p_i.assign(
                incident_type=f"{ALERT_CONFIG['upper_percentiles'][p]}-{ALERT_CONFIG['upper_percentiles'][p + 1]}_percentile_value",
                source='station_metrics')
            alert_i = alert_i.append(alert_p_i)
        logging.info(f'{alert_i.shape[0]} incidents with upper percentile')
        df_alerts = df_alerts.append(alert_i)

    # lower percentile
    for metric_i in ALERT_CONFIG['lower_alert']:
        df_all_segment = df_all[df_all.metric_name == metric_i].copy()

        percentiles_vals = [np.percentile(df_all_segment.metric_value.sort_values().to_list(), p)
                            for p in ALERT_CONFIG['lower_percentiles']]

        alert_i = pd.DataFrame()
        for p in range(len(percentiles_vals) - 1):
            alert_p_i = df_window[(df_window.metric_name == metric_i) &
                                  (df_window.metric_value >= percentiles_vals[p]) &
                                  (df_window.metric_value < percentiles_vals[p + 1])]
            alert_p_i = alert_p_i.assign(
                incident_type=f"{ALERT_CONFIG['lower_percentiles'][p]}-{ALERT_CONFIG['lower_percentiles'][p + 1]}_percentile_value",
                source='station_metrics')
            alert_i = alert_i.append(alert_p_i)
        logging.info(f'{alert_i.shape[0]} incidents with lower percentile')
        df_alerts = df_alerts.append(alert_i)

    df_alerts = df_alerts.assign(
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
        is_alert_send=False
    )

    df_to_pg(df_alerts, TABLE_NAME_ALERTS, SCHEMA_PROD, pk='clear table', engine=db_connection)


def get_predictions(metric_name, only_future=True):
    """
    Predict metric_value for all stations for metric_name
    :param metric_name: what metric to predict
    :param only_future: write to database only future prediction
    result inserted to database
    """
    db_connection = get_connection()
    logging.info(f'Getting predictions for all stations for metric {metric_name}')
    event_hour = 3  # number of events in hour
    n_preds = 24 * event_hour  # 24 hours

    # Get data
    df = pd.read_sql(f"""
        SELECT *
        FROM {SCHEMA_FILTERED}.{TABLE_NAME_METRICS}
        WHERE metric_name = '{metric_name}'
            AND metric_value IS NOT NULL
    """, con=db_connection)

    # Filter incorrect values
    if metric_name in ALERT_CONFIG['notnull_metrics']:
        df = df[df.metric_value > 0].copy()
    if metric_name in ALERT_CONFIG['positive_metrics']:
        df = df[df.metric_value >= 0].copy()

    for station_id in df.station_id.unique():
        logging.info(f'Station #{station_id}')
        df_i = df[df.station_id == station_id].set_index('report_dt').copy()
        if not df_i.shape[0]:
            continue
        data = df_i[:-int(n_preds / 2)].metric_value.copy()

        # initializing model parameters alpha, beta and gamma
        logging.info('Minimizing the loss function')
        x = [0, 0, 0]
        slen = event_hour * 24 * 365
        logging.info(f'Season len set to {slen}')
        opt = minimize(timeseriesCVscore, x0=x,
                       args=(data, mean_absolute_percentage_error, slen),
                       method="TNC", bounds=((0, 1), (0, 1), (0, 1))
                       )
        alpha_final, beta_final, gamma_final = opt.x
        logging.info(f'alpha: {alpha_final}, beta: {beta_final}, gamma: {gamma_final}')

        # training
        logging.info('Training the model')
        model = HoltWinters(data, slen=slen,
                            alpha=alpha_final,
                            beta=beta_final,
                            gamma=gamma_final,
                            n_preds=n_preds + int(n_preds / 2), scaling_factor=3)
        model.triple_exponential_smoothing()

        # applying results
        df_result = pd.DataFrame(model.result, columns=['metric_value_ml']) \
            .merge(df_i.reset_index(), how='left', left_index=True, right_index=True)
        dates = list(df_result[~df_result.report_dt.isnull()].report_dt)
        future_dates = [
            df_result.report_dt.max() + (i + 20) * timedelta(minutes=20)
            for i in range(df_result[df_result.report_dt.isnull()].shape[0])
        ]
        dates.extend(future_dates)
        df_result['report_dt_pred'] = dates
        df_result[['station_id', 'metric_name']] = [station_id, metric_name]

        if only_future:
            df_result = df_result[df_result.report_dt.isnull()]
        df_result = df_result[['report_dt_pred', 'station_id', 'metric_name', 'metric_value_ml']] \
            .rename(columns={'report_dt_pred': 'report_dt'})

        df_result['created_at'] = datetime.now(tz=timezone.utc)
        if metric_name in ALERT_CONFIG['positive_metrics']:
            df_result.loc[df_result.metric_value_ml < 0, 'metric_value_ml'] = 0
        df_to_pg(df_result, TABLE_NAME_METRICS_ML, SCHEMA_FILTERED, pk=None, engine=db_connection)
    logging.info('Done!')


def update_prod(update_actual=True, update_predictions=True):
    """
    Insert to production new data and predictions for all dates
    """
    if not update_actual and not update_predictions:
        logging.warning('Set at least one of params update_actual|update_predictions as True')
        return
    db_connection = get_connection()
    logging.info('UPDATING PRODUCTION station metrics!')

    logging.info('Station metrics')
    query_delete_metric = f"""DELETE FROM {SCHEMA_PROD}.{TABLE_NAME_METRICS} WHERE fl_ml = 0"""
    query_delete_metric_ml = f"""DELETE FROM {SCHEMA_PROD}.{TABLE_NAME_METRICS} WHERE fl_ml = 1"""
    query_metrics = f"""
        INSERT INTO {SCHEMA_PROD}.{TABLE_NAME_METRICS}
        (report_dt, station_id, metric_name, metric_value, created_at, prediction_created_at, fl_ml)
        SELECT filt.report_dt ,
                filt.station_id ,
                filt.metric_name ,
                filt.metric_value ,
                CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at ,
                NULL AS prediction_created_at ,
                0 AS fl_ml
        FROM {SCHEMA_FILTERED}.{TABLE_NAME_METRICS} filt
        """
    query_metrics_ml = f"""
        INSERT INTO {SCHEMA_PROD}.{TABLE_NAME_METRICS}
        (report_dt, station_id, metric_name, metric_value, created_at, prediction_created_at, fl_ml)
        SELECT ml.report_dt ,
                ml.station_id ,
                ml.metric_name ,
                ml.metric_value_ml AS metric_value,
                CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at ,
                ml.created_at AS prediction_created_at ,
                1 AS fl_ml
        FROM {SCHEMA_FILTERED}.{TABLE_NAME_METRICS_ML} ml
    """
    with db_connection.connect() as con:
        if update_actual:
            logging.info('Updating actual data')
            result = con.execute(query_delete_metric)
            logging.info(f"Deleted {result.rowcount} rows")
            result = con.execute(query_metrics)
            logging.info(f"Inserted {result.rowcount} rows")

        if update_predictions:
            logging.info('Updating predicted data')
            result = con.execute(query_delete_metric_ml)
            logging.info(f"Deleted {result.rowcount} rows")
            result = con.execute(query_metrics_ml)
            logging.info(f"Inserted {result.rowcount} rows")

    logging.info('Profile metrics')
    query_delete_profile = f"""DELETE FROM {SCHEMA_PROD}.{TABLE_NAME_PROFILE} WHERE fl_ml = 0"""
    query_profile = f"""
        INSERT INTO {SCHEMA_PROD}.{TABLE_NAME_PROFILE}
        (report_dt, station_id, altitude, temperature, created_at, prediction_created_at, fl_ml)
        SELECT filt.report_dt ,
                filt.station_id ,
                filt.altitude ,
                filt.temperature ,
                CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at ,
                NULL AS prediction_created_at ,
                0 AS fl_ml
        FROM {SCHEMA_FILTERED}.{TABLE_NAME_PROFILE} filt
        """
    with db_connection.connect() as con:
        if update_actual:
            logging.info('Updating actual data')
            result = con.execute(query_delete_profile)
            logging.info(f"Deleted {result.rowcount} rows")
            result = con.execute(query_profile)
            logging.info(f"Inserted {result.rowcount} rows")

        if update_predictions:
            pass

    logging.info('Out temperature')
    query_delete_out_temp = f"""DELETE FROM {SCHEMA_PROD}.{TABLE_NAME_TEMPERATURE} WHERE fl_ml = 0"""
    query_out_temp = f"""
        INSERT INTO {SCHEMA_PROD}.{TABLE_NAME_TEMPERATURE}
        (report_dt, station_id, outside_temperature, quality, created_at, prediction_created_at, fl_ml)
        SELECT filt.report_dt ,
                filt.station_id ,
                filt.outside_temperature ,
                filt.quality ,
                CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at ,
                NULL AS prediction_created_at ,
                0 AS fl_ml 
        FROM {SCHEMA_FILTERED}.{TABLE_NAME_TEMPERATURE} filt
        """
    with db_connection.connect() as con:
        if update_actual:
            logging.info('Updating actual data')
            result = con.execute(query_delete_out_temp)
            logging.info(f"Deleted {result.rowcount} rows")
            result = con.execute(query_out_temp)
            logging.info(f"Inserted {result.rowcount} rows")

        if update_predictions:
            pass

    logging.info('Done!')

