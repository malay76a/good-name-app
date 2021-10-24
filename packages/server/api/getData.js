const db = require("./db");

module.exports = (time, station_id, metric_name) => db
    .promise()
    .query(`SELECT * 
                FROM production.station_metrics 
                WHERE report_dt = "${time}"
                ${station_id ? `AND station_id = ${station_id}` : ''}
                ${metric_name ? `AND metric_name = "${metric_name}"` : ''}
                ORDER BY report_dt DESC`)
