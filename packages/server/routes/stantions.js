const setOfStantions = require('../api/getSetOfStantions');

module.exports = function (req, res) {
    setOfStantions()
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
