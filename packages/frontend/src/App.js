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

const getFetchData = (path, callback) => {
  fetch(makeUrl(path))
    .then((response) => response.json())
    .then((res) => {
      callback(res);
    })
    .catch((err) => console.error(err));
};

function App() {
  const [station, setStation] = useState('');
  const [polutionType, setPolutionType] = useState('');
  const [data, setData] = useState([]);
  const [indicators, setIndicators] = useState([]);
  const [stations, setStations] = useState([]);
  const [timelines, setTimelines] = useState([]);

  useEffect(() => {
    if (window) {
      getFetchData('pdk', setIndicators);
      getFetchData('stantions', (stations) =>
        setStations(stations.filter((item) => item.latitude))
      );
      getFetchData('timeline', setTimelines);
    }
  }, [window]);

  useEffect(() => {
    if (timelines.length) {
      const [time] = timelines;
      getFetchData(`data?time=${time}`, setData);
    }
  }, [timelines]);
  const dataReady =  !!(indicators.length && stations.length && data.length);

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
          <NormCard visible={!!polutionType} />

          {dataReady && <DataChoice />}
          <YaMap />
          {dataReady && <BottomData />}
        </AppContext.Provider>
      </div>
    </div>
  );
}

export default App;
