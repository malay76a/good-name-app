const db = require('../db');

module.exports = function (req, res)  {
    db
        .promise()
        .query(`SELECT DISTINCT
                      sm.station_id,
                      sc.station_name,
                      metric_name
                    FROM filtered.station_metrics sm
                    LEFT JOIN production.station_coordinates sc
                      ON sm.station_id = sc.id
                    WHERE station_id IS NOT NULL
                    ORDER BY 1, 2, 3`)
        .then(([rows]) => {
            const preparedData = Array.from(new Set(rows.map(i => i['metric_name'])))
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify(preparedData))
        })
        .catch(err => {
            console.log(err)
            res.statusCode(500);
            res.end('[]');
        });
}
