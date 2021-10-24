import { createContext, useState, useEffect } from 'react';

import NormCard from './NormCard';
import DataChoice from './DataChoiceCard';
import BottomData from './BottomData';
import YaMap from './YaMap';
import './App.css';

export const AppContext = createContext(null);

export const makeUrl = (path) => {
  return `${process.env.REACT_APP_SERVER_URL}/${path}`;
};

const servData = [
  {
  "report_dt": "2019-12-31T18:00:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.21,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T18:20:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.2,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T18:40:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.19,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T19:00:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.19,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T19:20:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.2,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T19:40:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.19,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T20:00:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.2,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T20:20:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.2,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T20:40:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.2,
  "created_at": "2021-10-23T08:09:47.000Z"
  },
  {
  "report_dt": "2019-12-31T21:00:00.000Z",
  "station_id": 10,
  "metric_name": "CO",
  "metric_value": 0.22,
  "created_at": "2021-10-23T08:09:47.000Z"
  }
  ];

  const getFetchData =(path, callback)=>{
    fetch(makeUrl(path))
    .then((response) => response.json())
    .then((res) => {
      callback(res);
    })
    .catch((err) => console.error(err));
  }
function App() {
  const [station, setStation] = useState('');
  const [polutionType, setPolutionType] = useState('');
  const [data, setData] = useState(servData);
  const [indicators, setIndicators] = useState([]);
  const [stations, setStations] = useState([]);
  const [timelines, setTimelines] = useState([]);

  useEffect(() => {
    if (window) {
      getFetchData('pdk', setIndicators);
      getFetchData('stantions', setStations);
      getFetchData('timeline', setTimelines);
    }
  }, [window]);

  useEffect(()=>{
    if(timelines.length){
      console.log(timelines);
      const [time] = timelines;
      getFetchData(`data?time=${time}`, setData)
    }

  }, [timelines]);

  return (
    <div className="App">
      <div className="App-map-wrapper">
        <AppContext.Provider
          value={{
            station,
            setStation,
            polutionType,
            setPolutionType,
            indicators,
            data,
            stations,
          }}
        >
          <NormCard type={polutionType} indicators={indicators} />
          <DataChoice indicators={indicators} />
          <YaMap />
          <BottomData />
        </AppContext.Provider>
      </div>
    </div>
  );
}

export default App;
