CREATE DATABASE raw;
CREATE DATABASE filtered;
CREATE DATABASE production;
 
-- raw
CREATE TABLE raw.station_coordinates (
     station_name TEXT,
     latitude TEXT,
     longitude TEXT,
     created_at TIMESTAMP
 );

CREATE TABLE raw.station_metrics (
	report_dt TIMESTAMP,
	filename TEXT,
	metric_name TEXT,
	metric_value FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE raw.out_temperature (
	report_dt TIMESTAMP,
	filename TEXT,
	source TEXT,
	outside_temperature FLOAT,
	quality	FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE raw.profile_metrics (
	report_dt TIMESTAMP,
	filename TEXT,
	source TEXT,
	altitude FLOAT,
	temperature FLOAT,
    created_at TIMESTAMP
);


-- filtered

CREATE TABLE filtered.station_metrics (
	report_dt TIMESTAMP,
	station_id INT,
	metric_name TEXT,
	metric_value FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE filtered.out_temperature (
	report_dt TIMESTAMP,
	station_id INT,
	outside_temperature FLOAT,
	quality	FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE filtered.profile_metrics (
	report_dt TIMESTAMP,
	station_id INT,
	altitude FLOAT,
	temperature FLOAT,
    created_at TIMESTAMP
);


-- filtered ml
CREATE TABLE filtered.station_metrics_ml (
	report_dt TIMESTAMP,
	station_id INT,
	metric_name TEXT,
	metric_value_ml FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE filtered.out_temperature_ml (
	report_dt TIMESTAMP,
	station_id INT,
	outside_temperature_ml FLOAT,
	quality	FLOAT,
    created_at TIMESTAMP
);

CREATE TABLE filtered.profile_metrics_ml (
	report_dt TIMESTAMP,
	station_id INT,
	altitude FLOAT,
	temperature_ml FLOAT,
    created_at TIMESTAMP
);


-- production
 CREATE TABLE production.station_coordinates (
     station_name TEXT,
     id INT NOT NULL AUTO_INCREMENT,
     latitude TEXT,
     longitude TEXT,
     created_at TIMESTAMP,
     PRIMARY KEY (id)
 );
 
CREATE TABLE production.station_metrics (
	report_dt TIMESTAMP,
	station_id INT, 
	metric_name TEXT,
	metric_value FLOAT,
	metric_value_ml FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE production.out_temperature (
	report_dt TIMESTAMP,
	station_id INT,
	outside_temperature FLOAT,
	outside_temperature_ml FLOAT,
	quality	FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE production.profile_metrics (
	report_dt TIMESTAMP,
	station_id INT,
	altitude FLOAT,
	temperature FLOAT,
	temperature_ml FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE production.incident (
	report_dt TIMESTAMP,
	station_id INT,
	metric_name TEXT,
	metric_value FLOAT,
	incident_type TEXT,
	incident_level FLOAT,
	is_alert_send BOOL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
