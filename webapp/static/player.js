"use strict";
// Parse each paced JPEG part so the timeline follows displayed frames, not a timer.
const player = {
  key:null, url:null, available:0, position:0, controller:null, paused:false,
  dragging:false, imageURL:null, finished:false, resumeWait:null, renderedPosition:null,
  clock(seconds){seconds=Math.max(0,Math.floor(seconds));return `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;},
  controls(){
    const range=$("play-progress");
    range.max=Math.max(0,this.available-0.001);
    range.disabled=!this.available;
    if(!this.dragging)range.value=Math.min(this.position,Number(range.max));
    $("play-position").textContent=this.clock(this.dragging?Number(range.value):this.position);
    $("play-available").textContent=this.clock(this.available);
    $("play-toggle").disabled=!this.available;
    $("play-toggle").textContent=this.paused||this.finished?"播放":"暂停";
    $("play-hint").textContent=this.available?`已生成 ${this.available.toFixed(2)} 秒 · 只能拖到已生成部分，尚未生成时等待`:'等待视频生成；进度条仅开放已生成的部分，可以向前拖拽回看。';
  },
  stop(){
    this.controller?.abort();this.controller=null;
    this.resumeWait?.();this.resumeWait=null;
    if(this.imageURL){URL.revokeObjectURL(this.imageURL);this.imageURL=null;}
  },
  reset(){this.stop();this.key=null;this.url=null;this.position=0;this.renderedPosition=null;this.available=0;this.paused=false;this.finished=false;this.dragging=false;$("preview").removeAttribute('src');this.controls();},
  async configure(key,url,available){
    const changed=key!==this.key;
    if(changed){this.reset();this.key=key;this.url=url;}
    this.available=available;this.controls();
    if(available&&!this.controller&&!this.paused&&!this.finished)this.start(this.position);
  },
  async start(position){
    this.stop();this.position=Math.max(0,Math.min(position,this.available-0.001));
    this.finished=false;
    const controller=new AbortController();this.controller=controller;
    this.controls();
    const concat=(a,b)=>{const c=new Uint8Array(a.length+b.length);c.set(a);c.set(b,a.length);return c;};
    const delimiter=new Uint8Array([13,10,13,10]);
    const headerEnd=bytes=>{outer:for(let i=0;i<=bytes.length-4;i++){for(let j=0;j<4;j++)if(bytes[i+j]!==delimiter[j])continue outer;return i;}return -1;};
    try{
      const response=await fetch(`${this.url}&offset=${this.position}`,{signal:controller.signal});
      if(!response.ok)throw new Error('播放请求失败');
      const reader=response.body.getReader();let buffer=new Uint8Array(),header=null,length=0;
      let displayed=0,lastShown=0;
      while(!controller.signal.aborted){
        if(!header){const end=headerEnd(buffer);if(end>=0){header=new TextDecoder().decode(buffer.slice(0,end));length=Number(/Content-Length:\s*(\d+)/i.exec(header)?.[1]);if(!length||length>20*1024*1024)throw new Error('无效视频帧');buffer=buffer.slice(end+4);}}
        if(header&&buffer.length>=length){
          while(this.paused&&displayed&&!controller.signal.aborted){
            await new Promise(resolve=>{const done=()=>{controller.signal.removeEventListener('abort',done);resolve();};this.resumeWait=done;controller.signal.addEventListener('abort',done,{once:true});});
            this.resumeWait=null;lastShown=0;
          }
          if(controller.signal.aborted)break;
          const interval=Number(/X-Frame-Interval:\s*([\d.]+)/i.exec(header)?.[1]||0.04)*1000;
          const delay=Math.max(0,interval-(performance.now()-lastShown));
          if(delay)await new Promise(resolve=>setTimeout(resolve,delay));
          if(controller.signal.aborted)break;
          if(this.paused&&displayed)continue;
          const jpeg=buffer.slice(0,length);buffer=buffer.slice(length);
          const position=Number(/X-Playback-Time:\s*([\d.]+)/i.exec(header)?.[1]||0);header=null;
          const nextURL=URL.createObjectURL(new Blob([jpeg],{type:'image/jpeg'}));
          const image=$("preview");const old=this.imageURL;
          image.src=nextURL;
          try{await image.decode();}catch(error){URL.revokeObjectURL(nextURL);if(controller.signal.aborted)break;throw error;}
          if(controller.signal.aborted){URL.revokeObjectURL(nextURL);break;}
          this.imageURL=nextURL;if(old)URL.revokeObjectURL(old);
          this.renderedPosition=position;
          lastShown=performance.now();displayed++;
          if(!this.paused||displayed===1)this.position=position;
          this.controls();
          continue;
        }
        const chunk=await reader.read();if(chunk.done){this.finished=true;this.controls();break;}
        buffer=concat(buffer,chunk.value);
      }
    }catch(error){if(error.name!=='AbortError'&&!controller.signal.aborted)toast(error.message,true);}
  }
};
$("play-progress").addEventListener('input',()=>{player.dragging=true;player.controls();});
$("play-progress").addEventListener('change',()=>{player.dragging=false;player.start(Number($("play-progress").value));});
$("play-toggle").onclick=()=>{if(player.finished){player.paused=false;player.start(0);}else if(player.paused){player.paused=false;player.resumeWait?.();player.controls();}else{player.paused=true;player.controls();}};
window.addEventListener('pagehide',()=>player.stop());
