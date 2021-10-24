const express = require('express');
const app = express();
require('dotenv').config();
const stantions = require('./routes/stantions');
const indicators = require('./routes/indicators');
const data = require('./routes/data');
const timeline = require('./routes/timeline');

const port = process.env.PORT || 3000;

app.get('/indicators', indicators);
app.get('/stantions', stantions);
app.get('/timeline', timeline);
app.get('/data', data);

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`);
})
