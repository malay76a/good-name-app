from tools import *

logging.getLogger().setLevel('INFO')


def main():
    ## Загрузка сырых данных
    upload_coordinates(PATH_STATION_COORDINATES)
    upload_station_metrics(PATH_STATION_METRICS)
    upload_profile(PATH_PROFILE)

    ## Обработка входных данных
    update_prod_station()
    update_filtered_metrics()
    update_filtered_profile()

    ## Определение аномалий
    define_anomaly(datetime(2020, 11, 1), datetime(2020, 11, 15))

    ## Получить предсказания
    for metric_name in ['CO', 'NO', 'NO2', 'PM2.5', 'PM10']:
        get_predictions(metric_name)

    ## Обновить продакшн
    update_prod()


if __name__ == "__main__":
    main()



