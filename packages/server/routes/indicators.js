const setOfIndicators = require('../api/getSetOfIndicators');

module.exports = function (req, res)  {
    setOfIndicators()
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
