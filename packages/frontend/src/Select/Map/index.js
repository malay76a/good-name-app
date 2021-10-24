import { useContext, useEffect } from 'react';
import {AppContext} from '../App';

function YaMap() {
   const {station, polutionType} = useContext(AppContext);
  console.log(station, polutionType);
  
  const init = () => {
    const { ymaps } = window;
    const myMap = new ymaps.Map(
      'map',
      {
        center: [55.76, 37.64],
        zoom: 10,
        controls: [],
      },
      { minZoom: 7 }
    );

    return myMap;
  };

  useEffect(() => {
    if (window.ymaps) {
      const { ymaps } = window;
      ymaps.ready(init);
    }
  }, [window.ymaps]);

  return <div className="App-map" id="map"></div>;
}

export default YaMap;
