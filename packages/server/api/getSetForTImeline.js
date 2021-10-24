const db = require("./db");

module.exports = () => db
    .promise()
    .query(`SELECT DISTINCT report_dt FROM production.station_metrics ORDER BY report_dt DESC LIMIT 432`)
