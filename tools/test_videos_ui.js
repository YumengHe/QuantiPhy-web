// Minimal DOM regression for independent groups; the legacy WebGL renderer is not under test.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

async function main() {
  const root = path.resolve(__dirname, '..');
  const dom = new JSDOM(fs.readFileSync(path.join(root, 'videos.html'), 'utf8'), {
    runScripts: 'outside-only', url: 'http://localhost/videos.html'
  });
  const w = dom.window;
  const questions = [0, 1].map((i) => ({ harness: true, row: i, q: 'Same question under independent priors',
    cat: 'S2', type: 'static', prior: `prior-${i}`, depth: '', answer: 1, pred: 1, mra: 1,
    src: i ? 'measurement' : 'scene', conf: .8, status: 'resolved', scene_id: `scene-${i}`, group: String(i),
    binding_status: 'matched', measurement: { reasons: [`diagnostic-${i}`] }, version: { pipeline: 'v77' },
    runtime_overlays: [{ scene_id: `scene-${i}`, path: `data/runtime/v77/scene-${i}.jpg`, t: 0, renderer_version: 'v77' }]
  }));
  const groups = questions.map((q) => ({ group: q.group, scene_id: q.scene_id, prior: q.prior,
    scene: null, scene2d: null, critique: { status: {}, iterations: [] } }));
  const video = { id: 'synthetic', harness: true, source: 'synthetic', video_type: '3D', dim: 3, fps: 24,
    duration: 2, cats: ['S2'], size: [854, 480], questions, groups, orig: '#', version: { label: 'v77' } };
  const data = { generated: 'synthetic', val: { file: 'val_v77_answers.csv', harness_tag: 'v77', videos: [video] },
    test: { file: 'retained.csv', videos: [] } };
  w.fetch = async () => ({ ok: true, json: async () => data });
  w.eval(fs.readFileSync(path.join(root, 'videos.js'), 'utf8'));
  await new Promise((resolve) => setTimeout(resolve, 20));
  const card = w.document.querySelector('#val-list details.vid');
  assert(card, w.document.querySelector('#val-list').textContent);
  card.open = true;
  card.dispatchEvent(new w.Event('toggle'));
  assert.strictEqual(card.querySelector('.group-select').options.length, 2);
  assert.strictEqual(card.querySelector('.qtable .runtime-frame img').getAttribute('src'), 'data/runtime/v77/scene-0.jpg');
  assert(card.querySelector('.qtable').textContent.includes('diagnostic-0'));
  assert(!card.querySelector('.qtable').textContent.includes('diagnostic-1'));
  const selector = card.querySelector('.group-select');
  selector.value = '1';
  selector.dispatchEvent(new w.Event('change'));
  assert.strictEqual(card.querySelector('.qtable .runtime-frame img').getAttribute('src'), 'data/runtime/v77/scene-1.jpg');
  assert(card.querySelector('.qtable').textContent.includes('diagnostic-1'));
  assert(!card.querySelector('.qtable').textContent.includes('diagnostic-0'));
  assert(card.querySelector('.qtable').textContent.includes('不是渲染几何'));
  assert(w.document.querySelector('#val-src-file').textContent.includes('v77'));
  dom.window.close();
  console.log('Group selection and exact per-question scene diagnostics: OK');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
