import { useContext, useEffect, useCallback } from 'react';
import ReactDOMServer from 'react-dom/server';

import { AppContext } from '../App';
import PolutionList from '../PolutionList';

function YaMap() {
  const { polutionType, stations, indicators } = useContext(AppContext);
  const dataReady = stations.length; // !!(indicators.length && stations.length);

  const init = () => {
    const findDistance = (firstCoords, secondCoords) => {
      const [x1, y1] = firstCoords;
      const [x2, y2] = secondCoords;
      return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    };

    const findClosestStationByCoords = (coords) => {
      const diffLength = stations
        .map((station) => {
          const { latitude, longitude } = station;
          return {
            distance: findDistance([latitude, longitude], coords),
            ...station,
          };
        })
        .sort((a, b) => a.distance - b.distance);

      return diffLength[0];
    };
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
    stations.map((station) => {
      const { latitude, longitude, name } = station;

      const markup = ReactDOMServer.renderToStaticMarkup(<PolutionList />);
      const additionalProps = {
          preset: 'islands#circleIcon',
          iconColor: '#3caa3c',
        };
      const place = new ymaps.Placemark(
        [latitude, longitude],
        {
          balloonContentHeader: name,
          balloonContentBody: markup,
        }
      );
      myMap.geoObjects.add(place);
    });

    myMap.events.add('click', function (e) {
      if (myMap.baloon?.isOpen()) myMap.baloon.close();
      const currentCoords = e.get('coords');
      const station = findClosestStationByCoords(currentCoords);

      const { name } = station;
      const markup = ReactDOMServer.renderToStaticMarkup(<PolutionList />);

      myMap.balloon.open(currentCoords, {
        contentHeader: name,
        contentBody: markup,
      });
    });

    return myMap;
  };

  useEffect(() => {
    if (dataReady) {
      const { ymaps } = window;
      ymaps.ready(init);
    }
  }, [dataReady]);

  return <div className="App-map" id="map"></div>;
}

export default YaMap;
