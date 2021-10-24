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
        className={cn('BottomModal', {
          'BottomModal-open': open,
        })}
      >
        <div className="BottomModalList">
          {open &&
            stations.map((station) => {
              const { name, id } = station;
              return (
                <div key={name} className={'BottomModalItem'}>
                  <PolutionList stationId={id} {...context} name={name} />
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}

export default BottomData;
