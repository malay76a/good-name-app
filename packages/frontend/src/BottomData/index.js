import { useState } from 'react';
import cn from 'classnames';
import { ReactComponent as Icon } from './Vector.svg';

import './BottomCard.css';

function BottomData() {
  const [open, setOpen] = useState(false);

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

      <div className="BottomModal">asd</div>
    </div>
  );
}

export default BottomData;
