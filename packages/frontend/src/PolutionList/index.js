import Dot from '../ListDot';

function PolutionList({ stationId, name, ...context }) {
  const { polutionType, indicators, data } = context;
  const indicatorsMap = indicators.reduce((acc, item) => {
    acc[item.metric_name] = item;
    return acc;
  }, {});
  const filteredData = data.filter((item) => {
    return item.station_id === stationId && polutionType
      ? item.metric_name === polutionType
      : true;
  });
  if (!filteredData.length) return null;

  return (
    <>
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
              {indicator?.metric_name} - {indicator?.metric_name_ru}
            </li>
          );
        })}
      </ul>
    </>
  );
}
export default PolutionList;
