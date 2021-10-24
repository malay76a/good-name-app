const db = require("./db");

module.exports = () => db
    .promise()
    .query('SELECT * FROM production.station_coordinates')
