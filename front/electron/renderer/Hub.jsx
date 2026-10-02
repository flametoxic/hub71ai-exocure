import React, { useLayoutEffect, useRef } from 'react';
import { mount } from './hub-controller.js';
import './hub.css';
import colorLogo from '../logocolor.svg';
export default function Hub({ search = "" }) {
const root = useRef(null);
const controller = useRef(null);
useLayoutEffect(() => {
controller.current = mount(root.current, search);
return () => controller.current?.dispose();
}, [search]);
return <div ref={root} className={'view hub-view' + (new URLSearchParams(search).has('embed') ? ' embed' : '')}>
<div className="hub">
  <div className="col home-overview" style={{"gap": "0"}}>
    <div className="row"><img className="device-logo" src={colorLogo} alt="CURE Home" /><span className="grow"></span></div>
    <div className="home-heading">Home</div>
    <div id="clock" className="clock" style={{"marginTop": "14px"}}>--:--</div>
    <div id="date" className="date"></div>
    <div id="air" className="air"></div>
  </div>
  <div className="col" style={{"minHeight": "0"}}>
    <div id="now" className="panel" style={{"flex": "0 0 auto"}}></div>
    <div className="panel assistant-panel" style={{"flex": "1"}}><div className="assistant-heading"><img className="assistant-logo" src={colorLogo} alt="" /><div><b>CURE</b></div></div><div id="say" className="say" role="log" aria-label="Conversation"></div><div className="row" id="ask"></div></div>
  </div>
  <div className="col" style={{"minHeight": "0"}}>
    <div className="panel approval-panel" style={{"flex": "1"}}><div className="tag" style={{"marginBottom": "8px"}}>You're in control</div><h3>Waiting for your yes</h3><div id="appr" className="appr"></div></div>
  </div>
  <div className="bottom" id="bottom"></div>
</div>
</div>;
}
