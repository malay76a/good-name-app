
import { createContext, useState, useEffect } from 'react';

import NormCard from './NormCard';
import DataChoice from './DataChoiceCard';
import BottomData from './BottomData';
import YaMap from './YaMap';
import './App.css';

export const AppContext = createContext(null);

function App() {
  const [station, setStation] = useState('');
  const [polutionType, setPolutionType] = useState('');
  useEffect(() => {
    if(window){
      fetch('http://localhost:3005/data')
      //.then(res=>res.json)
      .then(response => response.json())
      .then(res=>{
        console.log(res);
      })
      .catch(err=>console.error(err))
    }
  }, [])

  return (
    <div className="App">
      <div className="App-map-wrapper">
        <AppContext.Provider value={{station, setStation, polutionType, setPolutionType}}>
        <NormCard />
        <DataChoice />
        <YaMap />
        <BottomData />
        </AppContext.Provider>
      </div>
    </div>
  );
}

export default App;
