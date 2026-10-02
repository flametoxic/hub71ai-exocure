import React, { useLayoutEffect, useRef } from 'react';
import { mount } from './phone-controller.js';
import './phone.css';
import colorLogo from '../logocolor.svg';
export default function Phone({ search = "", onResident = () => {} }) {
const root = useRef(null);
const controller = useRef(null);
useLayoutEffect(() => {
controller.current = mount(root.current, search, onResident);
return () => controller.current?.dispose();
}, [search]);
return <div ref={root} className={'view phone-view' + (new URLSearchParams(search).has('embed') ? ' embed' : '')}>
<div className="phone">
  <div className="device-island" aria-hidden="true"><span /></div>
  <div className="status"><img className="device-logo" src={colorLogo} alt="CURE" /><span id="clock" className="muted"></span><span id="who" className="muted"></span></div>
  <div id="screen" className="screen"></div>
  <div id="chatbox" className="chatbox hidden"></div>
  <nav id="tabs" className="tabs hidden"></nav>
</div>
</div>;
}
