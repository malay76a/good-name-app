const express = require('express');
const app = express();
require('dotenv').config();
const stantions = require('./routes/stantions');
const indicators = require('./routes/indicators');
const data = require('./routes/data');

const port =  3004;

app.get('/indicators', indicators);
app.get('/stantions', stantions);
app.get('/data', data);

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`);
})