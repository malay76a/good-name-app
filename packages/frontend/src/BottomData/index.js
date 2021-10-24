import { useContext, useState } from 'react';
import cn from 'classnames';

import { ReactComponent as Icon } from './Vector.svg';
import { AppContext } from '../App';
import PolutionList from '../PolutionList';

import './BottomCard.css';

function BottomData() {
  const [open, setOpen] = useState(false);

  const context = useContext(AppContext);
  const { stations } = context;
  const toggleText = open ? '' : 'Смотреть результаты по всем станциям';

  return (
    <div className="BottomPanel">
      <button
        className={cn('BottomCardToggleBtn', {
          'BottomCardToggleBtn-active': open,
        })}
        onClick={() => setOpen(!open)}
      >
        {toggleText}{' '}
        <Icon
          className={cn('BottomCardToggleBtnIcon', {
            'BottomCardToggleBtnIcon-active': open,
          })}
        />
      </button>

      <div
        className="BottomModal"
        style={{
          height: open ? 'auto' : 0,
          visibility: open ? 'visible' : 'hidden',
          maxHeight: 400,
        }}
      >
        {open &&
          stations.map((station) => {
            const { name, id } = station;
            return (
              <div key={name}>
                <PolutionList stationId={id} {...context} name={name}/>
              </div>
            );
          })}
      </div>
    </div>
  );
}

export default BottomData;
