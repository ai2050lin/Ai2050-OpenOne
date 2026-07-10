import fs from 'node:fs';


const endpoint = process.argv[2] || 'http://127.0.0.1:9223';
const output = process.argv[3];
if (!output) throw new Error('Output PNG path is required');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const targets = await (await fetch(`${endpoint}/json`)).json();
const target = targets.find((item) => item.type === 'page');
if (!target) throw new Error('No Chromium page target');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', reject, { once: true });
});

let nextId = 1;
const pending = new Map();
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  const waiter = pending.get(message.id);
  if (waiter) {
    pending.delete(message.id);
    if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
    else waiter.resolve(message.result);
  }
});

function command(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

await command('Runtime.enable');
await command('Page.enable');
await sleep(3500);
const selection = await evaluate(`(() => {
  const select = [...document.querySelectorAll('select')].find((item) =>
    [...item.options].some((option) => option.value.includes('forward_pass_demo.json'))
  );
  if (!select) return { selected: false };
  const option = [...select.options].find((item) => item.value.includes('forward_pass_demo.json'));
  select.value = option.value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return { selected: true, value: option.value };
})()`);
await sleep(2500);
const interaction = await evaluate(`(() => {
  const visible = (element) => element && element.getBoundingClientRect().width > 0;
  const run = [...document.querySelectorAll('button')].find((item) =>
    visible(item) && item.textContent.trim() === '运行'
  );
  run?.click();
  const competition = [...document.querySelectorAll('button')].find((item) =>
    visible(item) && item.textContent.trim() === '竞争路径'
  );
  competition?.click();
  return { runClicked: Boolean(run), competitionClicked: Boolean(competition) };
})()`);
await sleep(5000);
const canvas = await evaluate(`(() => {
  const items = [...document.querySelectorAll('canvas')].map((item) => {
    const rect = item.getBoundingClientRect();
    return { width: item.width, height: item.height, cssWidth: rect.width, cssHeight: rect.height };
  });
  const activeButton = [...document.querySelectorAll('button')].find((item) =>
    item.textContent.trim() === '竞争路径' && item.classList.contains('is-active')
  );
  return { items, competitionActive: Boolean(activeButton) };
})()`);
const screenshot = await command('Page.captureScreenshot', { format: 'png', fromSurface: true });
fs.mkdirSync(new URL('.', `file://${output}`).pathname, { recursive: true });
fs.writeFileSync(output, Buffer.from(screenshot.data, 'base64'));
ws.close();
process.stdout.write(JSON.stringify({ selection, interaction, canvas, output }, null, 2));
