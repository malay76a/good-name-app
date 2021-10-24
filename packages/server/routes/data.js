const db = require("../db");

module.exports = function (req, res) {
    const {time, station_id, metric_name} = req.query;
    console.log(req.query)

    db
        .promise()
        .query(`SELECT * 
                    FROM filtered.station_metrics 
                    WHERE report_dt = "${time}"
                    ${station_id ? `AND station_id = ${station_id}` : ''}
                    ${metric_name ? `AND metric_name = "${metric_name}"` : ''}
                    ORDER BY report_dt DESC`)
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


// {
//     "report_dt": "2020-12-31T18:00:00.000Z",
//     "station_id": 1,
//     "metric_name": "CO",
//     "metric_value": 0.5,
//     "created_at": "2021-10-23T08:09:47.000Z"
// }
