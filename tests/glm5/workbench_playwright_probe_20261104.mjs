import {createRequire} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
const root=process.cwd(),out=path.join(root,'tests/glm5/result/workbench_playwright_20261104');
await fs.mkdir(out,{recursive:true});
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.AI2050_PLAYWRIGHT||'C:/Users/Admin/.workbuddy/binaries/node/workspace/node_modules/playwright-core');
const browser=await chromium.launch({headless:true,executablePath:'C:/Users/Admin/AppData/Local/ms-playwright/chromium-1247/chrome-win64/chrome.exe'});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:5173/workbench?lens=route',{waitUntil:'domcontentloaded'});
 await page.getByRole('navigation',{name:'工作台界面',exact:true}).waitFor();
 await page.screenshot({path:path.join(out,'01-overview.png'),fullPage:false});
 const text=await page.locator('body').innerText();await fs.writeFile(path.join(out,'probe.txt'),text);
 console.log(JSON.stringify({url:page.url(),errors,primary:await page.getByRole('navigation',{name:'工作台界面',exact:true}).innerText(),text:text.slice(0,1600)}));
}finally{await browser.close()}
