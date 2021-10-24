import cn from 'classnames';
import './Dot.css';

function Dot({ big, status }) {
  const classes = cn('Dot', `Dot-${status}`, {
    'Dot-big': big,
  });
  return <span className={classes}></span>;
}

export default Dot;
