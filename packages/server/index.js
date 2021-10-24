const express = require('express');
const app = express();
const cors = require('cors')
require('dotenv').config();
const stantions = require('./routes/stantions');
const indicators = require('./routes/indicators');
const data = require('./routes/data');
const timeline = require('./routes/timeline');
const pdk = require('./routes/pdk');

const port = process.env.PORT || 3000;

app.use(cors());

app.get('/indicators', indicators);
app.get('/stantions', stantions);
app.get('/timeline', timeline);
app.get('/data', data);
app.get('/pdk', pdk);
app.use(function(req, res, next) {
    res.header('Access-Control-Allow-Origin', '*');
    res.header(
      'Access-Control-Allow-Headers',
      'Origin, X-Requested-With, Content-Type, Accept'
    );
    next();
  });

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`);
})
