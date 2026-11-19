"""Offline single-file HTML dashboard (inline CSS/JS/SVG, no network, no CDN) showing alerts on a timeline for one log."""
import json, math
import numpy as np
from .detectors import DET_ORDER

NAMES = {"cn0_agc": "C/N0 + AGC", "uniformity": "Power uniformity", "clock": "Clock jump", "raim": "RAIM residuals",
         "innovation": "Innovation (Kalman)", "cusum": "CUSUM drag"}


class Recorder:
    """Collects per-epoch rows and alert events while a log is streamed."""
    def __init__(self, thr):
        self.thr = thr; self.rows = []; self.alerts = []; self.label_seen = False

    def add(self, ep, scores, events):
        c = ep.cn0[np.isfinite(ep.cn0)]
        self.rows.append(dict(t=ep.t, n=ep.n_sat, cn=float(c.mean()) if len(c) else None,
                              sp=float(c.std()) if len(c) > 1 else None,
                              agc=ep.agc_db if math.isfinite(ep.agc_db) else None,
                              noi=ep.noise_db if math.isfinite(ep.noise_db) else None,
                              clk=ep.clock_m if math.isfinite(ep.clock_m) else None,
                              s=[round(float(x), 3) for x in scores], lab=ep.label))
        if ep.label: self.label_seen = True
        for a in events: self.alerts.append(dict(t=a.t, d=a.detector, k=a.kind, s=round(a.score, 2), msg=a.msg))


def build_html(rec, title="log", notes=()):
    rows = rec.rows
    data = dict(title=title, thr=[rec.thr[n] for n in DET_ORDER], det=DET_ORDER, names=[NAMES[n] for n in DET_ORDER],
                t=[r["t"] for r in rows], n=[r["n"] for r in rows], cn=[r["cn"] for r in rows], sp=[r["sp"] for r in rows],
                agc=[r["agc"] for r in rows], noi=[r["noi"] for r in rows], clk=[r["clk"] for r in rows],
                s=[r["s"] for r in rows], lab=[r["lab"] for r in rows] if rec.label_seen else None,
                alerts=rec.alerts, notes=list(notes))
    js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("/*__DATA__*/null", js)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GNSS-Sentinel dashboard</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#d9d8d2;--grid:#ececE6;--bad:#b3261e;
--c0:#0072B2;--c1:#E69F00;--c2:#009E73;--c3:#CC79A7;--c4:#D55E00;--c5:#56B4E9;--truth:#8a8a84}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ecebe6;--mut:#a09f98;--card:#1f1f1d;--line:#3a3a36;--grid:#2a2a27;--bad:#ff8a80;--truth:#9a9a92}}
:root[data-theme=dark]{--bg:#161615;--fg:#ecebe6;--mut:#a09f98;--card:#1f1f1d;--line:#3a3a36;--grid:#2a2a27;--bad:#ff8a80;--truth:#9a9a92}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:13px;margin:0 0 6px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mut);margin-bottom:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.chip{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;display:flex;gap:6px;align-items:center}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.note{border-left:3px solid var(--c1);padding:4px 10px;margin:6px 0;color:var(--mut);font-size:13px}
svg{display:block;width:100%;height:auto;overflow:visible}text{fill:var(--mut);font-size:11px}
.grid{stroke:var(--grid)}.axis{stroke:var(--line)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}
tr.clk{cursor:pointer}tr.clk:hover{background:var(--grid)}.end{color:var(--mut)}
#readout{font-variant-numeric:tabular-nums;font-size:12px;color:var(--mut);min-height:34px;margin-top:4px}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer;font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--mut);margin-top:4px}
@media (max-width:600px){.hide-sm{display:none}}
</style></head><body><main>
<header><h1>GNSS-Sentinel</h1><div class="sub" id="sub"></div><div class="chips" id="chips"></div><div id="notes"></div></header>
<section class="card"><h2>Alert timeline</h2><div id="tl"></div><div class="legend" id="tlleg"></div><div id="readout"></div></section>
<section class="card"><h2>Detector score / threshold (alert when above 1)</h2><div id="sc"></div></section>
<section class="card"><h2>Mean C/N0 (band = +/- 1 std across satellites), dB-Hz</h2><div id="cn"></div></section>
<section class="card"><h2>AGC and noise-floor rise, dB</h2><div id="ag"></div><div class="legend"><span>AGC</span><span>noise floor</span></div></section>
<section class="card"><h2>Satellites used</h2><div id="ns"></div></section>
<section class="card"><h2>Receiver clock bias increment per epoch, m</h2><div id="ck"></div></section>
<section class="card"><h2>Alert log <button id="theme">toggle theme</button></h2><div style="overflow-x:auto"><table id="tab"><thead><tr><th>time (s)</th><th>detector</th><th>event</th><th>score / thr</th><th>explanation</th></tr></thead><tbody></tbody></table></div></section>
<p class="sub">Offline report generated by GNSS-Sentinel. Detectors learn a baseline from the first 60 epochs (assumed clean) and were validated only on synthetic data; treat alerts as situational-awareness hints, not proof.</p>
</main>
<script id="data" type="application/json">/*__DATA__*/null</script>
<script>
"use strict";
var D=JSON.parse(document.getElementById('data').textContent);
var COL=['var(--c0)','var(--c1)','var(--c2)','var(--c3)','var(--c4)','var(--c5)'];
var NS='http://www.w3.org/2000/svg',N=D.t.length,T0=D.t[0],T1=D.t[N-1]||1;
var W=1000,M={l:46,r:12,t:6,b:20},cross=[];
function el(n,a,p,txt){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;if(p)p.appendChild(e);return e}
function X(t){return M.l+(t-T0)/((T1-T0)||1)*(W-M.l-M.r)}
function ticks(a,b,n){var s=(b-a)/n,m=Math.pow(10,Math.floor(Math.log10(s||1))),r=s/m,st=(r>5?10:r>2?5:r>1?2:1)*m,o=[];for(var v=Math.ceil(a/st)*st;v<=b+1e-9;v+=st)o.push(+v.toFixed(6));return o}
function frame(host,H,y0,y1,fmt){
  var s=el('svg',{viewBox:'0 0 '+W+' '+H,role:'img'},host);
  var Y=function(v){return M.t+(1-(v-y0)/((y1-y0)||1))*(H-M.t-M.b)};
  ticks(y0,y1,4).forEach(function(v){el('line',{x1:M.l,x2:W-M.r,y1:Y(v),y2:Y(v),'class':'grid'},s);el('text',{x:M.l-6,y:Y(v)+4,'text-anchor':'end'},s,fmt?fmt(v):v)});
  ticks(T0,T1,8).forEach(function(v){el('line',{x1:X(v),x2:X(v),y1:H-M.b,y2:H-M.b+4,'class':'axis'},s);el('text',{x:X(v),y:H-4,'text-anchor':'middle'},s,Math.round(v))});
  el('line',{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,'class':'axis'},s);
  var cl=el('line',{y1:M.t,y2:H-M.b,stroke:'var(--fg)','stroke-width':1,opacity:0,'pointer-events':'none'},s);cross.push(cl);
  var hit=el('rect',{x:M.l,y:M.t,width:W-M.l-M.r,height:H-M.t-M.b,fill:'transparent'},s);
  hit.addEventListener('mousemove',function(ev){var r=s.getBoundingClientRect();var px=(ev.clientX-r.left)/r.width*W;show(T0+(px-M.l)/(W-M.l-M.r)*(T1-T0))});
  hit.addEventListener('mouseleave',function(){show(null)});
  return {s:s,Y:Y,H:H};
}
function line(f,arr,color,w){var d='',pen=false;for(var i=0;i<N;i++){var v=arr[i];if(v==null||!isFinite(v)){pen=false;continue}d+=(pen?'L':'M')+X(D.t[i]).toFixed(1)+' '+f.Y(v).toFixed(1);pen=true}
  if(d)el('path',{d:d,fill:'none',stroke:color,'stroke-width':w||1.4,'stroke-linejoin':'round'},f.s)}
function rng(arrs,pad,lo){var a=Infinity,b=-Infinity;arrs.forEach(function(x){x.forEach(function(v){if(v!=null&&isFinite(v)){if(v<a)a=v;if(v>b)b=v}})});if(!isFinite(a)){a=0;b=1}if(lo!=null)a=Math.min(a,lo);var p=(b-a)*(pad||.08)||1;return[a-p,b+p]}
function intervals(d){var out=[],on=null;D.alerts.forEach(function(a){if(a.d!==D.det[d])return;if(a.k==='start'&&on===null)on=a;else if(a.k==='end'&&on!==null){out.push([on.t,a.t,on.msg,on.s]);on=null}});if(on!==null)out.push([on.t,T1,on.msg,on.s]);return out}
// timeline
(function(){var lanes=D.det.length+(D.lab?1:0),H=lanes*26+30,f=frame(document.getElementById('tl'),H,0,lanes,function(){return ''});
 var yl=function(i){return M.t+i*26};
 D.det.forEach(function(n,i){el('text',{x:M.l-6,y:yl(i)+16,'text-anchor':'end'},f.s,D.names[i].split(' ')[0]);
  intervals(i).forEach(function(iv){var r=el('rect',{x:X(iv[0]),y:yl(i)+3,width:Math.max(3,X(iv[1])-X(iv[0])),height:18,rx:3,fill:COL[i],opacity:.9},f.s);var tt=el('title',{},r,D.names[i]+' '+iv[0]+'-'+iv[1]+' s: '+iv[2])})});
 if(D.lab){var i=D.det.length;el('text',{x:M.l-6,y:yl(i)+16,'text-anchor':'end'},f.s,'truth');var s0=null,lb='';
  for(var k=0;k<=N;k++){var l=k<N?D.lab[k]:'clean';if(l!=='clean'&&s0===null){s0=k;lb=l}if((l==='clean'||k===N)&&s0!==null){var r=el('rect',{x:X(D.t[s0]),y:yl(i)+3,width:Math.max(3,X(D.t[k-1])-X(D.t[s0])),height:18,rx:3,fill:'none',stroke:'var(--truth)','stroke-width':1.5,'stroke-dasharray':'4 3'},f.s);el('title',{},r,'synthetic ground truth: '+lb);s0=null}}}
 var lg=document.getElementById('tlleg');D.det.forEach(function(n,i){var s=document.createElement('span');s.innerHTML='<span class="dot" style="background:'+COL[i]+'"></span> '+D.names[i];lg.appendChild(s)});
 if(D.lab){var s=document.createElement('span');s.textContent='dashed box = synthetic ground truth';lg.appendChild(s)}})();
// scores
(function(){var f=frame(document.getElementById('sc'),210,0,4,function(v){return v+'x'});
 var thr=el('line',{x1:M.l,x2:W-M.r,y1:f.Y(1),y2:f.Y(1),stroke:'var(--bad)','stroke-dasharray':'5 4'},f.s);
 D.det.forEach(function(n,i){line(f,D.s.map(function(r){return Math.min(Math.max(r[i]/D.thr[i],0),4)}),COL[i],1.3)});
 el('text',{x:W-M.r,y:f.Y(1)-4,'text-anchor':'end'},f.s,'threshold (scores above 4x are clipped)')})();
(function(){var lo=rng([D.cn.map(function(v,i){return v!=null&&D.sp[i]!=null?v-D.sp[i]:null}),D.cn.map(function(v,i){return v!=null&&D.sp[i]!=null?v+D.sp[i]:null})],.05),f=frame(document.getElementById('cn'),190,lo[0],lo[1]);
 var up='',dn=[];for(var i=0;i<N;i++){if(D.cn[i]==null||D.sp[i]==null)continue;up+=(up?'L':'M')+X(D.t[i]).toFixed(1)+' '+f.Y(D.cn[i]+D.sp[i]).toFixed(1);dn.unshift('L'+X(D.t[i]).toFixed(1)+' '+f.Y(D.cn[i]-D.sp[i]).toFixed(1))}
 if(up)el('path',{d:up+dn.join('')+'Z',fill:COL[0],opacity:.18},f.s);line(f,D.cn,COL[0],1.5)})();
(function(){var r=rng([D.agc,D.noi],.1,0),f=frame(document.getElementById('ag'),150,r[0],r[1]);line(f,D.agc,COL[1],1.4);line(f,D.noi,COL[2],1.2);
 if(D.agc.every(function(v){return v==null}))el('text',{x:W/2,y:60,'text-anchor':'middle'},f.s,'no AGC / noise-floor fields in this log (e.g. NMEA input)')})();
(function(){var r=rng([D.n],.1,0),f=frame(document.getElementById('ns'),120,r[0],r[1]);line(f,D.n,COL[3],1.6)})();
(function(){var inc=D.clk.map(function(v,i){return i&&v!=null&&D.clk[i-1]!=null?v-D.clk[i-1]:null}),r=rng([inc],.1),f=frame(document.getElementById('ck'),150,r[0],r[1]);line(f,inc,COL[4],1.2);
 if(D.clk.every(function(v){return v==null}))el('text',{x:W/2,y:60,'text-anchor':'middle'},f.s,'no clock-bias field in this log')})();
function nearest(t){var lo=0,hi=N-1;while(hi-lo>1){var m=(lo+hi)>>1;if(D.t[m]<=t)lo=m;else hi=m}return Math.abs(D.t[lo]-t)<=Math.abs(D.t[hi]-t)?lo:hi}
function f1(v,d){return v==null?'-':v.toFixed(d)}
function show(t){var ro=document.getElementById('readout');if(t==null||N===0){cross.forEach(function(c){c.setAttribute('opacity',0)});return}
 var i=nearest(Math.min(Math.max(t,T0),T1)),x=X(D.t[i]);cross.forEach(function(c){c.setAttribute('x1',x);c.setAttribute('x2',x);c.setAttribute('opacity',.6)});
 ro.textContent='t='+D.t[i]+' s  sats '+D.n[i]+'  C/N0 '+f1(D.cn[i],1)+' (sd '+f1(D.sp[i],1)+')  AGC '+f1(D.agc[i],2)+'  noise '+f1(D.noi[i],2)+'  score/thr: '+D.det.map(function(n,k){return n+' '+(D.s[i][k]/D.thr[k]).toFixed(2)}).join(', ')+(D.lab&&D.lab[i]?'  truth: '+D.lab[i]:'')}
// header + table
(function(){var cnt={};D.alerts.forEach(function(a){if(a.k==='start')cnt[a.d]=(cnt[a.d]||0)+1});
 document.getElementById('sub').textContent=D.title+' - '+N+' epochs, '+Math.round(T1-T0)+' s'+(D.lab?' - SYNTHETIC log with ground-truth labels':'');
 var ch=document.getElementById('chips');D.det.forEach(function(n,i){var c=document.createElement('span');c.className='chip';c.innerHTML='<span class="dot" style="background:'+COL[i]+'"></span>'+D.names[i]+': <b>'+(cnt[n]||0)+'</b> alert'+((cnt[n]||0)===1?'':'s');ch.appendChild(c)});
 D.notes.forEach(function(t){var d=document.createElement('div');d.className='note';d.textContent=t;document.getElementById('notes').appendChild(d)});
 var tb=document.querySelector('#tab tbody');if(!D.alerts.length){var r=tb.insertRow();var c=r.insertCell();c.colSpan=5;c.textContent='No alerts.'}
 D.alerts.forEach(function(a){var r=tb.insertRow();r.className='clk'+(a.k==='end'?' end':'');var k=D.det.indexOf(a.d);var th=D.thr[k];
  [a.t,D.names[k],a.k==='start'?'ALERT':'cleared',a.k==='start'?(a.s/th).toFixed(1)+'x':'',a.msg].forEach(function(v,j){var c=r.insertCell();c.textContent=v;if(j===1)c.style.color=COL[k]});
  r.addEventListener('click',function(){show(a.t);document.getElementById('tl').scrollIntoView({behavior:'smooth',block:'center'})})});
 document.getElementById('theme').addEventListener('click',function(){var r=document.documentElement,dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);r.setAttribute('data-theme',dark?'light':'dark')})})();
</script></body></html>
"""
