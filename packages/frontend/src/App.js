import { YMaps, Map, ZoomControl } from 'react-yandex-maps';
import NormCard from './NormCard';

import DataChoice from './DataChoiceCard';
import BottomData from './BottomData';
import './App.css';

function App() {
  return (
    <div className="App">
      <div className="App-map-wrapper">
        <NormCard />

        <DataChoice />
        <YMaps className="App-map">
          <Map
            className="App-map"
      
            defaultState={{ center: [55.75, 37.57], zoom: 9, controls: [] }}
            options={{ maxZoom: 10, minZoom: 1 }}
          >
            <ZoomControl
              options={{
                position: {
                  right: 36,
                  left: 'auto',
                  top: '50%',
                  bottom: 'auto',
                  position: 'absolute',
                },

                size: 'small',
              }}
              state={{ maxZoom: 5, minZoom: 1 }}
            />
          </Map>
        </YMaps>
        <BottomData />
      </div>
    </div>
  );
}

export default App;
