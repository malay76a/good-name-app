const db = require("./db");

module.exports = () => db
    .promise()
    .query(`SELECT DISTINCT
                  sm.station_id,
                  sc.station_name,
                  metric_name
                FROM production.station_metrics sm
                LEFT JOIN production.station_coordinates sc
                  ON sm.station_id = sc.id
                WHERE station_id IS NOT NULL
                ORDER BY 1, 2, 3`)
