import React, { useLayoutEffect, useRef, useState } from 'react';
import Phone from './Phone.jsx';
import Hub from './Hub.jsx';
import BackendStatus from './BackendStatus.jsx';
import { mount } from './console-controller.js';
import './console.css';
import logo from '../logo.svg';

export default function Console({ search = '' }) {
  const root = useRef(null);
  const controller = useRef(null);
  const [rid, setRid] = useState(new URLSearchParams(search).get('rid') || '');
  const [device, setDevice] = useState('phone');
  const [rightOpen, setRightOpen] = useState(false);
  const [switchingProfile, setSwitchingProfile] = useState(false);
  async function chooseProfile(profile) {
    if (switchingProfile || !controller.current) return;
    setSwitchingProfile(true);
    try {
      if (profile === 'leila') await controller.current.selectLeila();
      else { controller.current.selectNew(); setDevice('phone'); }
    } catch { /* The API client displays the backend error. */ }
    finally { setSwitchingProfile(false); }
  }
  useLayoutEffect(() => {
    controller.current = mount(root.current, search, setRid);
    return () => controller.current?.dispose();
  }, [search]);
  useLayoutEffect(() => {
    const element = root.current;
    const stage = element.querySelector('.stage');
    const navigation = element.querySelector('.device-switch');
    const city = element.querySelector('#city');
    const fit = () => {
      const cityHeight = city.children.length ? city.offsetHeight + 16 : 0;
      const width = Math.max(0, stage.clientWidth - 32);
      const height = Math.max(0, stage.clientHeight - 32 - navigation.offsetHeight - 16 - cityHeight);
      const expanded = element.classList.contains('right-collapsed');
      const phoneScale = Math.max(.25, Math.min((width - 12) / 390, (height - 12) / 844, expanded ? 1.4 : .8513));
      const hubScale = Math.max(.25, Math.min((width - 12) / 1024, (height - 12) / 600, expanded ? 1.65 : 1));
      element.style.setProperty('--phone-scale', phoneScale);
      element.style.setProperty('--hub-scale', hubScale);
    };
    const observer = new ResizeObserver(fit);
    observer.observe(stage);
    observer.observe(city);
    fit();
    return () => observer.disconnect();
  }, [rightOpen]);
  return <div ref={root} className={`view console-view${rightOpen ? '' : ' right-collapsed'}`}>
    <header>
      <img className="company-logo" src={logo} alt="" />
      <span className="brand">CURE</span>
      <span className="grow" />
      <BackendStatus rid={rid} />
      <nav className="profile-switch" aria-label="Profile">
        <button disabled={switchingProfile} aria-pressed={rid !== 'leila'} onClick={() => chooseProfile('new')}>New</button>
        <button disabled={switchingProfile} aria-pressed={rid === 'leila'} onClick={() => chooseProfile('leila')}>Leila’s</button>
      </nav>
      <button className="danger" onClick={() => controller.current?.resetAll()}>Reset</button>
    </header>
    <main>
      <section className="stage">
        <nav className="device-switch" aria-label="Central device view">
          <button aria-pressed={device === 'phone'} onClick={() => setDevice('phone')}>Phone</button>
          <button aria-pressed={device === 'hub'} onClick={() => setDevice('hub')}>Home hub</button>
        </nav>
        <div className="device-panel" hidden={device !== 'phone'}>
          <div className="phonebox"><Phone key={rid} search={`?embed=1&rid=${encodeURIComponent(rid)}`} onResident={value => controller.current?.setRid(value)} /></div>
        </div>
        <div className="device-panel" hidden={device !== 'hub'}>
          <div className="hubbox"><Hub key={rid} search={`?embed=1&rid=${encodeURIComponent(rid)}`} /></div>
        </div>
        <div id="city" className="col" style={{maxWidth:640}} />
      </section>
      <section className="right">
        <button className="sidebar-toggle" aria-expanded={rightOpen} aria-controls="insights" title={rightOpen ? 'Hide insights' : 'Show insights'} onClick={() => setRightOpen(value => !value)}>
          {rightOpen ? 'Insights  ›' : '‹'}
        </button>
        <div id="insights" hidden={!rightOpen}>
        <div className="tag" style={{marginTop:10}}>Under the hood · last reply</div><div id="hood" />
        </div>
      </section>
    </main>
  </div>;
}

