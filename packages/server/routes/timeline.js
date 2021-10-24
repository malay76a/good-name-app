const setForTimeline = require('../api/getSetForTImeline');

module.exports = function (req, res)  {
    setForTimeline()
        .then(([rows]) => {
            const preparedData = rows.map(i => i['report_dt'])
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify(preparedData))
        })
        .catch(err => {
            console.log(err)
            res.statusCode(500);
            res.end('[]');
        });
}
