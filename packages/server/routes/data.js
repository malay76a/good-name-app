const db = require("../db");

module.exports = function (req, res) {
    db
        .promise()
        .query(`SELECT * FROM filtered.station_metrics ORDER BY station_id DESC LIMIT 10`)
        .then(([rows]) => {
            // const preparedData = rows.map(i => {
            //     return {
            //         id: i.id,
            //         name: i['station_name'].trim(),
            //         latitude: parseFloat(i.latitude),
            //         longitude: parseFloat(i.longitude),
            //     }
            // })
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify(rows))
        })
        .catch(err => {
            console.log(err)
            res.statusCode(500);
            res.end('[]');
        });
}

// /data:
//     get:
//       summary: 'Возвращает данные по станциям, времени и веществам'
//       parameters:
//         - name: stantion_id
//           in: query
//           description: ID станции
//           required: false
//           schema:
//             type: integer
//             format: int64
//         - name: time
//           in: query
//           description: время в UTC
//           required: false
//           schema:
//             type: integer
//             format: int64
//         - name: indicator_id
//           in: query
//           description: ID вещества
//           required: false
//           schema:
//             type: integer
//             format: int64
//       responses:
//         '200':
//           description: 'Запрос за данными таймлайна'
//           content:
//             application/json:
//               schema:
//                 type: object
//                 properties:
//                   stantion:
//                     type: object
//                     properties:
//                       id:
//                         type: integer
//                         format: int64
//                       time:
//                         type: integer
//                         format: int64
//                       data:
//                         type: array
//                         $ref: '#/components/schemas/Indicator'
