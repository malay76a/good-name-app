const data = require('../api/getData');

module.exports = function (req, res) {
    const {time, station_id, metric_name} = req.query;

    data(time, station_id, metric_name)
        .then(([rows]) => {
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify(rows))
        })
        .catch(err => {
            console.log(err)
            res.statusCode(500);
            res.end('[]');
        });
}
