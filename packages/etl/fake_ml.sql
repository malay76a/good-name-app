INSERT INTO filtered.station_metrics_ml
(report_dt, station_id, metric_name, metric_value_ml, created_at)
SELECT
	report_dt ,
	station_id ,
	metric_name ,
	metric_value + RAND() * metric_value AS metric_value_ml,
	CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at 
FROM filtered.station_metrics sm ;



INSERT INTO filtered.station_metrics_ml
(report_dt, station_id, metric_name, metric_value_ml, created_at)
SELECT
	ADDDATE(
		MAX(report_dt) OVER(PARTITION BY station_id), 
		DATEDIFF((MAX(report_dt) OVER(PARTITION BY station_id)), report_dt)
	),
	station_id,
	metric_name ,
	metric_value + RAND() * metric_value AS metric_value_ml,
	CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at 
FROM filtered.station_metrics sm ;




INSERT INTO filtered.profile_metrics_ml
(report_dt, station_id, altitude, temperature_ml, created_at)
SELECT
	report_dt ,
	station_id ,
	altitude , 
	temperature + RAND() * temperature AS temperature_ml,
	CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at 
FROM filtered.profile_metrics sm ;


INSERT INTO filtered.profile_metrics_ml
(report_dt, station_id, altitude, temperature_ml, created_at)
SELECT
	ADDDATE(
		MAX(report_dt) OVER(PARTITION BY station_id), 
		DATEDIFF((MAX(report_dt) OVER(PARTITION BY station_id)), report_dt)
	),
	station_id ,
	altitude , 
	temperature + RAND() * temperature AS temperature_ml,
	CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at 
FROM filtered.profile_metrics sm ;



INSERT INTO filtered.out_temperature_ml
(report_dt, station_id, outside_temperature_ml, quality, created_at)
SELECT
	report_dt ,
	station_id ,
	outside_temperature + RAND() * outside_temperature AS outside_temperature_ml,
	quality , 
	CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at 
FROM filtered.out_temperature sm ;


INSERT INTO filtered.out_temperature_ml
(report_dt, station_id, outside_temperature_ml, quality, created_at)
SELECT
	ADDDATE(
		MAX(report_dt) OVER(PARTITION BY station_id), 
		DATEDIFF((MAX(report_dt) OVER(PARTITION BY station_id)), report_dt)
	),
	station_id ,
	outside_temperature + RAND() * outside_temperature AS outside_temperature_ml,
	quality , 
	CONVERT_TZ(NOW(),'SYSTEM','UTC') AS created_at 
FROM filtered.out_temperature sm ;

