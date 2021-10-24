import { useState } from 'react';
import cn from 'classnames';
import './Switch.css';

const SWITCH_VALUES = {
  air: 'air',
  meteo: 'meteo',
};

function Switch() {
  const [active, setValue] = useState(SWITCH_VALUES.air);

  return (
    <div className="SwitchCard">
      <button
        className={cn('SwitchBtn', {
          'SwitchBtn-active': active === SWITCH_VALUES.air,
        })}
        onClick={() => setValue(SWITCH_VALUES.air)}
      >
        Воздух
      </button>
      <button
        className={cn('SwitchBtn', {
          'SwitchBtn-active': active === SWITCH_VALUES.meteo,
        })}
        onClick={() => setValue(SWITCH_VALUES.meteo)}
      >
        Метеоданные
      </button>
    </div>
  );
}

export default Switch;
