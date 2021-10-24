// import cn from 'classnames';
import { useState, useEffect, useCallback } from 'react';

import { Swiper, SwiperSlide } from 'swiper/react';
import SwiperCore, { Navigation } from 'swiper';

import './Select.css';

SwiperCore.use([Navigation]);

function SelectComponent() {
  const [value, setValue] = useState('');
  const [open, setOpen] = useState(false);
  const [swiper, setSwiper] = useState(null);

  const slideTo = useCallback((index) => swiper.slideTo(index), [swiper]);

  useEffect(() => {
    if (swiper && value) {
      slideTo(+value);
      setValue('');
    }
  }, [swiper, value, slideTo]);

  const handleChange = (value) => {
    setValue(value);
    setOpen(false);
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
        >
          <SwiperSlide onClick={onSlideClick}>Slide 1</SwiperSlide>
          <SwiperSlide onClick={onSlideClick}>Slide 2</SwiperSlide>
          <SwiperSlide onClick={onSlideClick}>Slide 3</SwiperSlide>
          <SwiperSlide onClick={onSlideClick}>Slide 4</SwiperSlide>
        </Swiper>
      </div>
      <ul className={'List'} style={{ display: open ? 'block' : 'none' }}>
        <li onClick={() => handleChange(1)}>1</li>
        <li onClick={() => handleChange(2)}>2</li>
        <li onClick={() => handleChange(3)}>3</li>
      </ul>
    </div>
  );
}

export default SelectComponent;
