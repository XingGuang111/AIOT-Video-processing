const assert=require('node:assert/strict');
const {chromium}=require('../logs/web_browser_qa/node_modules/playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page=await browser.newPage();const base='http://127.0.0.1:8080';let hold=false,started=false;
  const template={device_no:0,status:'completed',running:false,source:'sample.mp4',segments:1,created_at:100,start_ts:100,elapsed_seconds:1,total_tracks:1,pending_seconds:0,pending_frames:0,stages:{capture:'completed',inference:'completed',tag:'completed'}};
  const first={...template,id:'a'.repeat(32)},second={...template,id:'b'.repeat(32),segments:0,start_ts:200};
  try{
    await page.route('**/api/overview',route=>route.fulfill({json:{jobs:[first,second],devices:[0],samples:[]}}));
    await page.route('**/api/playback/**',route=>route.fulfill({json:{items:[],available:0}}));
    await page.route('**/thumbnail',route=>route.abort());
    await page.route('**/api/results/**',async route=>{
      const old=route.request().url().includes(first.id);
      if(old&&hold){started=true;await new Promise(resolve=>setTimeout(resolve,800));}
      await route.fulfill({json:old?{items:[{name:'100.mp4',ts:100,kind:'segment',size:100,thumbnail_url:'/thumbnail',download_url:'/download'}],total:1}:{items:[],total:0}});
    });
    await page.goto(base);
    await page.waitForFunction(()=>document.getElementById('connection').textContent==='服务已连接');
    await page.locator('#show-previous-tasks').check();await page.waitForSelector('.job');
    await page.locator('.job').first().click();await page.waitForSelector('#results .segment');
    hold=true;
    await page.evaluate(()=>{refresh();});
    for(let i=0;!started&&i<40;i++)await page.waitForTimeout(50);
    assert.ok(started);
    await page.evaluate(id=>{state.selected=id;state.sessionJobIds.add(id);setDevice(0);refresh();},second.id);
    assert.equal(await page.locator('#results .segment').count(),0);
    await page.waitForTimeout(1000);
    assert.equal(await page.locator('#results .segment').count(),0,'late previous request repopulated results');
    await page.waitForTimeout(1600);
    assert.equal(await page.locator('#result-count').textContent(),'0');
    assert.equal(await page.locator('#show-history').count(),0);
    console.log('PASS: task switch clears immediately, delayed previous responses discarded, new task results isolated');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
