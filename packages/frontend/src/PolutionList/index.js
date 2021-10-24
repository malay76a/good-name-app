import Dot from '../ListDot';

function PolutionList({ station, type, data }) {
  return (
    <ul className={'List'}>
      <li>
        <Dot big status="norm" /> ниже ПДК
      </li>
    </ul>
  );
}
export default PolutionList;
