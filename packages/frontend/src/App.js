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
  const [selectDate, setSelectDate] = useState('');

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
      if(selectDate) getFetchData(`data?time=${selectDate}`, setData);
  }, [selectDate]);

  useEffect(() => {
      if (timelines.length) {
          const [time] = timelines;
          setSelectDate(time)
      }
  }, [timelines])

  const dataReady =  !!(indicators.length && stations.length && data.length);

  const prepereTime = (item) => {
      const [date, tail] = item.split('T');
      const [hours, minutes] = tail.split(':');
      return `${date} ${hours}:${minutes}`
  }

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
          <ul className="Date-list">
              {timelines.map(i => <li className={i === selectDate ? 'active' : ''}  onClick={() => setSelectDate(i)}>{prepereTime(i)}</li>)}
          </ul>
          {dataReady && <DataChoice />}
          <YaMap />
          {dataReady && <BottomData />}
        </AppContext.Provider>
      </div>
    </div>
  );
}

export default App;
