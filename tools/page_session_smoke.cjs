/* Read-only browser verification against an existing website with historical jobs. */
const assert = require('node:assert/strict');
const {chromium} = require('../logs/web_browser_qa/node_modules/playwright');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8080';
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const context=await browser.newContext();
  const errors=[];
  async function blank(page){
    await page.waitForFunction(()=>document.getElementById('connection').textContent==='服务已连接');
    await page.waitForTimeout(1800); // Verify a later polling refresh stays blank too.
    assert.equal(await page.locator('#jobs .job').count(),0);
    assert.equal(await page.locator('#results .segment').count(),0);
    assert.equal(await page.locator('#preview').getAttribute('src'),null);
    assert.equal(await page.locator('#segments').textContent(),'—');
    assert.equal(await page.locator('#process-events').textContent(),'');
    assert.equal(await page.locator('#logs').textContent(),'选择任务后显示运行日志。');
    assert.equal(await page.locator('#show-previous-tasks').isChecked(),false);
    assert.equal(await page.locator('#show-history').count(),0);
  }
  try{
    const page=await context.newPage();
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(base);
    await blank(page);
    await page.locator('#show-previous-tasks').check();
    await page.waitForSelector('#jobs .job');
    assert.equal(await page.locator('#preview').getAttribute('src'),null);
    await page.locator('#jobs .job').first().click();
    await page.waitForFunction(()=>document.getElementById('segments').textContent!=='—');
    await page.locator('.log-panel > details > summary').click();
    await page.waitForFunction(()=>document.getElementById('logs').textContent!=='选择任务后显示运行日志。');
    await page.reload();
    await blank(page);
    const reopened=await context.newPage();
    await reopened.goto(base);
    await blank(reopened);
    assert.deepEqual(errors,[]);
    console.log('PASS: fresh page, polling, explicit history selection, reload and reopened tab');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
