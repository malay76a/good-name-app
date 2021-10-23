const db = require('../db');

module.exports = function (req, res) {
    db
        .promise()
        .query('SELECT * FROM production.station_coordinates')
        .then(([rows]) => {
            const preparedData = rows.map(i => {
                return {
                    id: i.id,
                    name: i['station_name'].trim(),
                    latitude: parseFloat(i.latitude),
                    longitude: parseFloat(i.longitude),
                }
            })
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify(preparedData))
        })
        .catch(err => {
            console.log(err)
            res.statusCode(500);
            res.end('[]');
        });
}
