const pdk = require('../api/getPDK');

module.exports = function (req, res)  {
    pdk()
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
