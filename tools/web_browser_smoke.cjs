/* 浏览器真实交互：上传、启动、播放、回看、停止、窄屏布局。 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('../logs/web_browser_qa/node_modules/playwright');

(async () => {
  const [base, output, sample] = process.argv.slice(2);
  const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  try {
    await page.goto(base, {waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.getElementById('connection').textContent==='服务已连接');
    await page.screenshot({path:path.join(output,'website-empty.png'),fullPage:true});
    const invalid=await page.request.post(`${base}/api/tasks`,{data:{source:'missing-file.mp4',device_no:923010}});
    assert.equal(invalid.status(),400);
    const cross=await page.request.post(`${base}/api/tasks`,{headers:{Origin:'https://example.com'},data:{source:sample,device_no:923010}});
    assert.equal(cross.status(),403);
    await page.locator('#file-input').setInputFiles(sample);
    await page.waitForFunction(()=>document.getElementById('upload-status').textContent.startsWith('已上传：'));
    await page.locator('#device').fill('923010');
    assert.equal(await page.locator('#buffer-mode').inputValue(),'test');
    await page.locator('#buffer-mode').selectOption('runtime');
    const started=page.waitForResponse(response=>response.url()===`${base}/api/tasks`&&response.status()===201);
    await page.locator('#start-button').click();
    const job=await (await started).json();
    console.log(`Started website job ${job.id}`);
    const duplicate=await page.request.post(`${base}/api/tasks`,{data:{source:sample,device_no:923010}});
    assert.equal(duplicate.status(),409);
    let completed;
    const deadline=Date.now()+240000;
    while(Date.now()<deadline){
      completed=await (await page.request.get(`${base}/api/tasks/${job.id}`)).json();
      if(!completed.running)break;
      await page.waitForTimeout(1000);
    }
    assert.equal(completed.status,'completed',JSON.stringify(completed));
    assert.equal(completed.buffer_mode,'runtime');
    for(const folder of ['capture_buffer','buffer_101','buffer_201']){
      const dir=path.join(output,folder,'device_923010');
      const entries=fs.existsSync(dir)?fs.readdirSync(dir):[];
      assert.ok(!entries.some(name=>/^\d/.test(name)),`remaining buffers: ${dir} ${entries}`);
    }
    assert.ok(completed.segments>0);
    assert.equal(completed.full_video.kind,'full');
    assert.equal(completed.metrics.tag.full_frames,45);
    assert.equal(completed.metrics.capture.frames,45);
    assert.ok(completed.metrics.inference.timings.yolo.calls>0);
    const fullDownload=await page.request.get(base+completed.full_video.download_url);
    assert.equal(fullDownload.status(),200);
    assert.ok((await fullDownload.body()).length>100);
    await page.waitForFunction(()=>document.getElementById('full-output').textContent.includes('下载完整标注视频'));
    assert.ok(await page.locator('#timing-rows tr').count()>=16);
    await page.waitForFunction(()=>document.getElementById('preview').naturalWidth>0,{},{timeout:30000});
    await page.waitForFunction(()=>Number(document.getElementById('segments').textContent)>0);
    const results=await (await page.request.get(`${base}/api/results/923010?start_ts=${completed.start_ts}`)).json();
    const thumbnail=await page.request.get(base+results.items[0].thumbnail_url);
    assert.equal(thumbnail.status(),200);
    assert.ok(thumbnail.headers()['content-type'].includes('image/jpeg'));
    const download=await page.request.get(base+results.items[0].download_url);
    assert.equal(download.status(),200);
    assert.ok((await download.body()).length>100);
    const traversal=await page.request.get(`${base}/api/results/923010/bad.mp4/download`);
    assert.equal(traversal.status(),404);
    await page.screenshot({path:path.join(output,'website-results.png'),fullPage:true});
    await page.locator('.segment').first().click();
    await page.waitForFunction(()=>document.getElementById('preview-state').textContent==='片段回看');
    await page.waitForFunction(()=>document.getElementById('preview').naturalWidth>0);
    await page.locator('#live-button').click();
    await page.waitForFunction(()=>document.getElementById('preview-state').textContent==='顺序播放');
    await page.locator('.log-panel > details > summary').click();
    await page.waitForFunction(()=>document.getElementById('logs').textContent.includes('汇总'));
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'website-mobile.png'),fullPage:true});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'mobile overflow');
    await page.setViewportSize({width:1440,height:1050});
    await page.locator('#source').fill('VideoCapture/demo_videos/video3.mp4');
    await page.locator('#device').fill('923011');
    await page.locator('#buffer-mode').selectOption('test');
    const secondStart=page.waitForResponse(response=>response.url()===`${base}/api/tasks`&&response.status()===201);
    await page.locator('#start-button').click();
    const second=await (await secondStart).json();
    await page.locator('.job.selected .stop-button').click();
    await page.waitForFunction(()=>document.querySelector('.job.selected .status')?.textContent==='已停止',{},{timeout:60000});
    const stopped=await (await page.request.get(`${base}/api/tasks/${second.id}`)).json();
    assert.equal(stopped.status,'stopped');
    assert.equal(stopped.buffer_mode,'test');
    assert.deepEqual(errors,[]);
    const report={passed:['upload','start','duplicate_device_rejected','invalid_source_rejected','cross_origin_rejected','real_pipeline_complete','mjpeg_preview','thumbnail','download','replay','follow_latest','logs','mobile_layout','stop'],job:completed,stopped,console_errors:errors};
    fs.writeFileSync(path.join(output,'verification.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({passed:report.passed,segments:completed.segments,screenshots:output}));
  } finally {await page.close();await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
