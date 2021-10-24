const db = require("./db");

module.exports = () => db
    .promise()
    .query(`SELECT DISTINCT report_dt 
                FROM production.station_metrics
                WHERE MINUTE(report_dt) IN (0, 20,40)
                ORDER BY report_dt DESC 
                LIMIT 216`)
