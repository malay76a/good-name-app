import { useContext, useState, useEffect, useCallback } from 'react';
import { Swiper, SwiperSlide } from 'swiper/react';
import SwiperCore, { Navigation } from 'swiper';

import { AppContext } from '../App';

import './Select.css';

SwiperCore.use([Navigation]);

function SelectComponent() {
  const [value, setValue] = useState('');
  const [open, setOpen] = useState(false);
  const [swiper, setSwiper] = useState(null);
  const { polutionType, setPolutionType, indicators } = useContext(AppContext);

  const slideTo = useCallback((index) => swiper.slideTo(index), [swiper]);

  useEffect(() => {
    if (swiper && value) {
      slideTo(+value);
      setValue('');
    }
  }, [swiper, value, slideTo]);

  const allIndicators = [
    { metric_name: '', metric_name_ru: 'Все показатели' },
  ].concat(indicators);

  const handleChange = (value) => {
    setValue(value+1);
    setOpen(false);
    setPolutionType(allIndicators[value].metric_name);
  };
  const onSlideClick = () => setOpen(true);

  return (
    <div className="SelectCard">
      <div className="Swiper-wrapper">
        <Swiper
          slidesPerView={1}
          spaceBetween={0}
          loop={true}
          navigation={true}
          className="mySwiper"
          onSwiper={setSwiper}
          simulateTouch={false}
          onSlideChange={(e) => {
            handleChange(e.realIndex);
          }}
        >
          {allIndicators.map((item, index) => {
            return (
              <SwiperSlide key={index} onClick={onSlideClick}>
                {item.metric_name_ru}
              </SwiperSlide>
            );
          })}
        </Swiper>
      </div>
      <ul className={'List'} style={{ display: open ? 'block' : 'none' }}>
        {allIndicators.map((item, index) => {
          return (
            <li
              key={index}
              className={
                item.metric_name === polutionType
                  ? 'ListItem ListItem-active'
                  : 'ListItem'
              }
              onClick={() => handleChange(index)}
            >
              {item.metric_name_ru}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default SelectComponent;
