const mysql = require('mysql2');
const con = mysql.createConnection({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    // database: 'production',
    ssl  : {
        // DO NOT DO THIS
        // set up your ca correctly to trust the connection
        rejectUnauthorized: false
    }
});

module.exports = con;

// [
//     { Tables_in_production: 'incidents' },
//     { Tables_in_production: 'out_temperature' },
//     { Tables_in_production: 'profile_metrics' },
//     { Tables_in_production: 'station_coordinates' },
//     { Tables_in_production: 'station_metrics' }
// ]
