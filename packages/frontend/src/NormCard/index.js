import Dot from '../ListDot';
import './NormCard.css';

function NormCard({type}) {
  if(!type) return null;
  return (
    <div className="Card NormCard">
      <ul className="List NormCard-list">
        <li>
          <Dot big status="norm" /> ниже ПДК
        </li>
        <li>
          <Dot big status="warning" /> 1,0 – 1,5 ПДК
        </li>
        <li>
          <Dot big status="danger" /> более 1,5 ПДК
        </li>
      </ul>
    </div>
  );
}

export default NormCard;
