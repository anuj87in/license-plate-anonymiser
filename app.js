// License plate anonymiser: YOLOv8s (ONNX) runs in the browser; the image never leaves the device.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

const MODEL_URL = 'model/plate_fp16.onnx';
const SIZE = 640;          // model input (letterboxed square)
const IOU = 0.5;           // NMS overlap threshold, as in the report
const PAD = 0.15;          // each box is enlarged by 15% per side before blurring
const MIN_CONF = 0.05;     // candidates kept after inference; the slider filters from here
const REVIEW_BELOW = 0.5;  // detections under this confidence are flagged for human review
const MAX_SIDE = 4096;     // very large photos are scaled down to keep memory in check

const $ = (id) => document.getElementById(id);
const els = {
  drop: $('drop'), file: $('file'), status: $('status'), result: $('result'),
  view: $('view'), conf: $('conf'), confVal: $('confVal'), boxes: $('showBoxes'),
  toggle: $('toggle'), download: $('download'), summary: $('summary'), samples: $('samples'),
  pasteRow: $('paste-row'), pasteBtn: $('pasteBtn'),
};

let session = null;
let backend = '';
let current = null;  // { name, source: canvas, candidates: [{box, score}], ms }
let showOriginal = false;

// ---------- model ----------
async function loadModel() {
  setStatus('Loading model (about 22 MB, cached after the first visit)…');
  const forced = new URLSearchParams(location.search).get('backend');  // e.g. ?backend=wasm for testing
  const providers = forced ? [forced] : navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [ep] });
      backend = ep === 'webgpu' ? 'WebGPU' : 'WebAssembly';
      break;
    } catch (err) {
      console.warn(`Backend ${ep} unavailable:`, err);
    }
  }
  if (!session) throw new Error('This browser cannot run the model.');
  setStatus(`Model ready (${backend}). Choose a photo to anonymise.`);
  els.drop.classList.remove('disabled');
  els.samples?.classList.remove('disabled');
  if (navigator.clipboard?.read) els.pasteRow.hidden = false;  // the keyboard shortcut works everywhere
}

// ---------- pre-processing: letterbox to 640 x 640, grey (114) padding ----------
function letterbox(src) {
  const r = Math.min(SIZE / src.height, SIZE / src.width);
  const nw = Math.round(src.width * r), nh = Math.round(src.height * r);
  const left = Math.floor((SIZE - nw) / 2), top = Math.floor((SIZE - nh) / 2);
  const c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, left, top, nw, nh);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 4] / 255;
    input[i + plane] = px[i * 4 + 1] / 255;
    input[i + 2 * plane] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]), r, left, top };
}

// ---------- post-processing: decode (1, 5, 8400), NMS, map back to the original image ----------
function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const area = (q) => (q[2] - q[0]) * (q[3] - q[1]);
  return inter / (area(a) + area(b) - inter + 1e-9);
}

function decode(out, n, lb, imgW, imgH) {
  let cands = [];
  for (let i = 0; i < n; i++) {
    const score = out[4 * n + i];
    if (score < MIN_CONF) continue;
    const cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i];
    cands.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score });
  }
  cands.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cands) {
    if (kept.every((k) => iou(k.box, c.box) <= IOU)) kept.push(c);
  }
  // Greedy NMS in descending score order: filtering by a higher threshold afterwards gives the same
  // result as filtering first, so the slider needs no new inference.
  return kept.map(({ box, score }) => ({
    score,
    box: [
      clamp((box[0] - lb.left) / lb.r, 0, imgW), clamp((box[1] - lb.top) / lb.r, 0, imgH),
      clamp((box[2] - lb.left) / lb.r, 0, imgW), clamp((box[3] - lb.top) / lb.r, 0, imgH),
    ],
  }));
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function padBox([x1, y1, x2, y2], w, h) {
  const dx = (x2 - x1) * PAD, dy = (y2 - y1) * PAD;
  return [Math.max(0, Math.trunc(x1 - dx)), Math.max(0, Math.trunc(y1 - dy)),
          Math.min(w, Math.trunc(x2 + dx)), Math.min(h, Math.trunc(y2 + dy))];
}

// ---------- Gaussian blur inside the box only (same kernel rule as the report) ----------
// OpenCV: kernel k = largest box side (odd), sigma = 0.3 * ((k - 1) / 2 - 1) + 0.8.
// Approximated by three box-blur passes, which works in every browser (canvas filters do not).
function blurRegion(ctx, x1, y1, x2, y2) {
  const w = x2 - x1, h = y2 - y1;
  if (w < 2 || h < 2) return;
  const k = Math.floor(Math.max(w, h) / 2) * 2 + 1;
  const sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;
  const img = ctx.getImageData(x1, y1, w, h);
  const radii = boxRadii(sigma, 3);
  const a = img.data, b = new Uint8ClampedArray(a.length);
  for (const r of radii) {
    boxPass(a, b, w, h, r, true);
    boxPass(b, a, w, h, r, false);
  }
  ctx.putImageData(img, x1, y1);
}

function boxRadii(sigma, n) {
  const wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let wl = Math.floor(wIdeal);
  if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
  return Array.from({ length: n }, (_, i) => ((i < m ? wl : wu) - 1) / 2);
}

// One horizontal or vertical running-sum pass; edges are clamped to the region.
function boxPass(src, dst, w, h, r, horizontal) {
  const len = horizontal ? w : h, lines = horizontal ? h : w;
  const step = horizontal ? 4 : w * 4;
  const span = 2 * r + 1;
  for (let line = 0; line < lines; line++) {
    const base = horizontal ? line * w * 4 : line * 4;
    for (let ch = 0; ch < 3; ch++) {
      const at = (i) => src[base + clamp(i, 0, len - 1) * step + ch];
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += at(i);
      for (let i = 0; i < len; i++) {
        dst[base + i * step + ch] = acc / span;
        acc += at(i + r + 1) - at(i - r);
      }
    }
    for (let i = 0; i < len; i++) dst[base + i * step + 3] = src[base + i * step + 3];
  }
}

// ---------- pipeline ----------
async function process(file, name) {
  if (!session) return;
  setStatus('Detecting plates…');
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    setStatus('This file could not be read as an image. Please try a JPG or PNG photo.', true);
    return;
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const source = document.createElement('canvas');
  source.width = Math.round(bitmap.width * scale);
  source.height = Math.round(bitmap.height * scale);
  source.getContext('2d').drawImage(bitmap, 0, 0, source.width, source.height);
  bitmap.close?.();

  const t0 = performance.now();
  const lb = letterbox(source);
  const res = await session.run({ images: lb.tensor });
  const out = res[session.outputNames[0]];
  const candidates = decode(out.data, out.dims[2], lb, source.width, source.height);
  const ms = performance.now() - t0;

  current = { name: (name || 'image').replace(/\.[^.]+$/, ''), source, candidates, ms };
  showOriginal = false;
  // Every new image starts at the default (lowest) threshold, so a setting chosen for the previous
  // image cannot hide plates in this one.
  els.conf.value = els.conf.defaultValue;
  els.confVal.textContent = Number(els.conf.value).toFixed(2);
  render();
  setStatus('Done. Check the result below, or choose another photo.');
  els.result.hidden = false;
  els.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function selected() {
  const t = Number(els.conf.value);
  return current.candidates.filter((c) => c.score >= t);
}

function anonymised() {
  const { source } = current;
  const c = document.createElement('canvas');
  c.width = source.width;
  c.height = source.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0);
  for (const d of selected()) blurRegion(ctx, ...padBox(d.box, c.width, c.height));
  return c;
}

function render() {
  if (!current) return;
  const dets = selected();
  const base = showOriginal ? current.source : anonymised();
  const v = els.view;
  v.width = base.width;
  v.height = base.height;
  const ctx = v.getContext('2d');
  ctx.drawImage(base, 0, 0);
  if (els.boxes.checked) {
    const lw = Math.max(2, Math.round(v.width / 400));
    const fs = Math.max(12, Math.round(v.width / 60));
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    for (const d of dets) {
      const [x1, y1, x2, y2] = padBox(d.box, v.width, v.height);
      const review = d.score < REVIEW_BELOW;
      const color = review ? '#f59e0b' : '#22c55e';
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${d.score.toFixed(2)}${review ? ' review' : ''}`;
      const tw = ctx.measureText(label).width + 8;
      const ty = y1 - fs - 6 >= 0 ? y1 - fs - 6 : y2;
      ctx.fillStyle = color;
      ctx.fillRect(x1, ty, tw, fs + 6);
      ctx.fillStyle = '#111';
      ctx.fillText(label, x1 + 4, ty + fs);
    }
  }
  const review = dets.filter((d) => d.score < REVIEW_BELOW).length;
  els.summary.innerHTML =
    `<strong>${dets.length}</strong> plate${dets.length === 1 ? '' : 's'} blurred` +
    (review ? ` · <span class="warn">${review} low-confidence, please review</span>` : '') +
    ` · ${Math.round(current.ms)} ms on your device (${backend})`;
  els.toggle.textContent = showOriginal ? 'Show anonymised' : 'Show original';
  els.toggle.setAttribute('aria-pressed', String(showOriginal));
}

function download() {
  if (!current) return;
  anonymised().toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${current.name}_anonymised.jpg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, 'image/jpeg', 0.92);
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle('error', isError);
}

// ---------- UI wiring ----------
els.file.addEventListener('change', () => {
  const f = els.file.files[0];
  if (f) process(f, f.name);
  els.file.value = '';
});
els.drop.addEventListener('dragover', (e) => { e.preventDefault(); els.drop.classList.add('over'); });
els.drop.addEventListener('dragleave', () => els.drop.classList.remove('over'));
els.drop.addEventListener('drop', (e) => {
  e.preventDefault();
  els.drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) process(f, f.name);
});
els.conf.addEventListener('input', () => {
  els.confVal.textContent = Number(els.conf.value).toFixed(2);
  render();
});
els.boxes.addEventListener('change', render);
els.toggle.addEventListener('click', () => { showOriginal = !showOriginal; render(); });
els.download.addEventListener('click', download);
els.samples?.addEventListener('click', async (e) => {
  const btn = e.target.closest('[data-src]');
  if (!btn) return;
  setStatus('Loading sample…');
  const blob = await (await fetch(btn.dataset.src)).blob();
  process(blob, btn.dataset.src.split('/').pop());
});

// Paste: Cmd/Ctrl + V anywhere on the page, or the button (clipboard API, where the browser has it).
document.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (!item) {
    if (!e.clipboardData?.types?.includes('text/plain')) setStatus('The clipboard does not contain an image. Copy an image or take a screenshot first.', true);
    return;
  }
  e.preventDefault();
  process(item.getAsFile(), 'pasted-image');
});
els.pasteBtn.addEventListener('click', async () => {
  try {
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find((t) => t.startsWith('image/'));
      if (type) {
        process(await item.getType(type), 'pasted-image');
        return;
      }
    }
    setStatus('The clipboard does not contain an image. Copy an image or take a screenshot first.', true);
  } catch {
    setStatus('The browser blocked clipboard access. Press ⌘/Ctrl + V instead, or choose a file.', true);
  }
});

loadModel().catch((err) => setStatus(`${err.message} Please try a recent version of Chrome, Edge, Firefox or Safari.`, true));

// Exposed for the automated parity test only.
window.__plates = { ready: () => !!session, backend: () => backend,
  detect: async (blob) => { await process(blob, 'test'); return current.candidates; } };
