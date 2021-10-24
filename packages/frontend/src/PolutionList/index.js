import Dot from '../ListDot';

function PolutionList({ stationId, name, ...context }) {
  const { polutionType, indicators, data } = context;
  const indicatorsMap = indicators.reduce((acc, item) => {
    acc[item.metric_name] = item;
    return acc;
  }, {});
  const filteredData = data
  .filter(i => i.fl_ml)
  .filter((item) => {
    return item.station_id === stationId && (polutionType
      ? item.metric_name === polutionType
      : true);
  }).filter(i => indicatorsMap[i.metric_name]);
 
  if (!filteredData.length) return null;

  return (
    <div key={name} className={'BottomModalItem'}>
      {name && <div className={'PolutionListTitle'}>{name}</div>}
      <ul className={'List'}>
        {filteredData.map((dataItem, index) => {
          const indicator = indicatorsMap[dataItem.metric_name];
 
          if (!indicator) return null;

          const getStatus = (pdk, value) => {
            if (value < pdk) return 'norm';
            if (value < pdk * 1.5) return 'warning';
            return 'danger';
          };
          return (
            <li key={index}>
              <Dot status={getStatus(indicator.PDK, dataItem.metric_value)} />{' '}
              <span style={{ fontWeight: 'bold' }}>
                {indicator?.metric_name}
              </span>{' '}
              - {dataItem?.metric_value}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
export default PolutionList;
