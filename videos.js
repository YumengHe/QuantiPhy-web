// evaluation.html: one row per question (unit <video_id>_q<k>). Data: data/videos.json (built by tools/build_videos.py).
(function () {
  'use strict';
  // UI text in English; I18N (i18n.js) translates it: Ht for templates, tr() for single strings
  const Ht = typeof I18N !== 'undefined' ? I18N.H : (s, ...v) => s.reduce((r, x, i) => r + v[i - 1] + x);
  const tr = typeof I18N !== 'undefined' ? I18N.t : (s, p) => (p ? s.replace(/\{(\w+)\}/g, (m, k) => (k in p ? p[k] : m)) : s);
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  let DATA = null, OURS = null;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  function fmtNum(x) {
    if (x == null || !isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return Number(x).toExponential(2);
    return String(Number(x.toPrecision(4)));
  }
  function mraClass(m) { return m >= 0.8 ? 'm4' : m >= 0.6 ? 'm3' : m >= 0.4 ? 'm2' : m >= 0.2 ? 'm1' : 'm0'; }
  function mraBadge(m) { return isFinite(m) ? Ht`<span class="mra ${mraClass(m)}">${fmt(m, 2)}</span>` : Ht`<span class="muted">—</span>`; }
  // questions of the selected set : DEV20 = docs/DEV20.md in its order, All = every question of the run
  function setQuestions() {
    if ($('v-set').value !== 'dev') return DATA.val.questions;
    const byId = Object.fromEntries(DATA.val.questions.map((q) => [q.id, q]));
    return DATA.val.dev.map((id) => byId[id]);
  }
  // the read-out and the answer of the scene-program run (q.sp): answer_scene, the strict read-out (undetermined = 0), and
  // answer_final, the agent's answer (the reported one); a unit without an answer scores 0 in both
  const fMRA = (q) => q.sp.final_mra, sMRA = (q) => q.sp.scene_mra;
  // the per-question mean of both MRAs and the determined count of a set of questions
  const spCells = (qs) => Ht`<td class="n">${qs.length}</td><td class="n">${fmt(mean(qs.map(sMRA)), 3)}</td>` +
    Ht`<td class="n"><b>${fmt(mean(qs.map(fMRA)), 3)}</b></td><td class="n">${qs.filter((q) => q.sp.determined).length}</td>`;
  const SP_HEAD = Ht`<th class="n">n</th><th class="n">Read-out</th><th class="n">Answer</th><th class="n">determined</th>`;
  // ---------- summary of the selected set: both MRAs, the set shared with the baselines, categories ----------
  function valSummary() {
    const all = setQuestions(), dev = $('v-set').value === 'dev';
    $('val-final').textContent = fmt(mean(all.map(fMRA)), 3);
    $('val-scene').textContent = fmt(mean(all.map(sMRA)), 3);
    $('val-n').textContent = [Ht`${all.length} questions`, Ht`${new Set(all.map((q) => q.video)).size} videos`].join(' · ');
    const shared = all.filter((q) => OURS.questions[q.id]);
    $('t-sets').innerHTML = Ht`<tr><th>Set</th>${SP_HEAD}</tr><tr><td>${dev ? 'DEV20' : tr('Validation')}</td>${spCells(all)}</tr>` +
      (shared.length !== all.length ? Ht`<tr><td>Shared with the baselines</td>${spCells(shared)}</tr>` : '');
    $('t-cats').innerHTML = Ht`<tr><th>Category</th>${SP_HEAD}</tr>` +
      CATS.map((c) => all.filter((q) => q.cat === c)).map((qs, i) => qs.length ? Ht`<tr><td>${CATS[i]}</td>${spCells(qs)}</tr>` : '').join('');
  }

  // ---------- one question ----------
  function summaryRow(q, v) {
    const sp = q.sp;
    const nums = Ht`<span class="qv" data-k="GT">${fmtNum(q.answer)}</span>` +
      Ht`<span class="qv" data-k="Read-out">${sp.determined ? fmtNum(sp.scene) : ''}</span>` +
      Ht`<span class="qv qsm" data-k="Read-out MRA">${mraBadge(sp.scene_mra)}</span>` +
      Ht`<span class="qv pred" data-k="Answer">${fmtNum(sp.final)}</span>`;
    return Ht`<summary><img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}"><span class="qmain"><span class="qid">${esc(q.id)}<span class="qver">${esc(q.cat)}</span></span><span class="qtext">${esc(q.q)}</span></span>` +
      Ht`<span class="qnums sp">${nums}</span><span class="qm">${mraBadge(sp.final_mra)}</span></summary>`;
  }
  function questionCard(q, v) {
    const d = document.createElement('details');
    d.className = 'qrow';
    d.dataset.id = q.id;
    d.innerHTML = summaryRow(q, v);
    let built = false, video = null;
    const onToggle = () => {
      if (d.open && !built) {
        built = true;
        const body = document.createElement('div');
        body.className = 'qbody';
        body.innerHTML = priorBlock(q) + spBlock(q) + mediaBlock(v, q) + timeBlock(q);
        d.appendChild(body);
        video = body.querySelector('video');
        body.addEventListener('click', (e) => {
          const b = e.target.closest('.tq-btn');
          if (b) jumpTo(body, v, b.dataset.t === '' ? null : Number(b.dataset.t));
        });
      }
      // the <video> only holds a src while its question is open, so collapsed rows keep no connections or buffers
      if (video) {
        // preload="auto" + a tiny seek decodes the first frame right away
        if (d.open && !video.getAttribute('src')) {
          video.preload = 'auto'; video.src = v.video;
          video.addEventListener('loadedmetadata', () => { if (video.readyState < 2) video.currentTime = 0.001; }, { once: true });
        }
        if (!d.open) { video.pause(); video.removeAttribute('src'); video.load(); }
      }
    };
    d.addEventListener('toggle', onToggle);
    return d;
  }

  // the scene-program products of the question: the read-out and the answer, the fitted program's overlays, the plan grid and the shapes
  function spBlock(q) {
    const sp = q.sp;
    const u = sp.unit ? Ht` ${esc(sp.unit)}` : '';
    const ans = Ht`<div class="scroll-x"><table class="spans"><tr><th></th><th class="n">Value</th><th class="n">GT</th><th class="n">MRA</th><th></th></tr>` +
      Ht`<tr><td><code>answer_scene</code></td><td class="n">${sp.determined ? Ht`${fmtNum(sp.scene)}${sp.sigma != null ? Ht` ± ${fmtNum(sp.sigma)}` : ''}${u}` : tr('undetermined')}</td>` +
      Ht`<td class="n">${fmtNum(q.answer)}</td><td class="n">${mraBadge(sp.scene_mra)}</td><td>${esc(sp.reason || '')}</td></tr>` +
      Ht`<tr><td><code>answer_final</code></td><td class="n"><b>${fmtNum(sp.final)}</b>${u}</td><td class="n">${fmtNum(q.answer)}</td>` +
      Ht`<td class="n">${mraBadge(sp.final_mra)}</td><td></td></tr></table></div>` +
      (sp.rationale ? Ht`<p class="sprat">${esc(sp.rationale)}</p>` : '');
    const img = (src, alt) => Ht`<a class="ovimg" href="${esc(src)}"><img loading="lazy" src="${esc(src)}" alt="${esc(alt)}"></a>`;
    const r = sp.renders;
    const frames = r.overlays.map((p) => Ht`<figure class="runtime-frame">${img(p, p.split('/').pop())}<figcaption>${esc(p.match(/overlay_f(\d+)/)[1].replace(/^0+(?=\d)/, ''))}</figcaption></figure>`).join('');
    return Ht`<h4>Scene program</h4>` + ans +
      Ht`<div class="runtime-frames spframes">${frames}</div>` +
      Ht`<div class="spplan">${img(r.plan_grid, tr('plan grid'))}${img(r.shapes, tr('shapes'))}</div>`;
  }

  // the question's own prior: text facts and distance anchors, one per line
  function priorBlock(q) {
    const lines = [q.prior, q.depth].filter(Boolean).flatMap((s) => String(s).split('\n')).map((s) => s.trim()).filter(Boolean);
    return Ht`<div class="qfull">${esc(q.q)}<span class="qver">${esc(q.cat)}</span></div>` + (lines.length ? Ht`<h4>prior</h4><ul class="prior">${lines.map((l) => Ht`<li>${esc(l)}</li>`).join('')}</ul>` : '');
  }

  // ---------- question time t_q ----------
  const tqLabel = (q) => (q.tq.length ? q.tq.map((x) => fmt(x, 2)).join('–') + ' s' : q.tjump != null ? Ht`≈${fmt(q.tjump, 2)} s` : '');
  // the question's time on the original video (its middle when the question names none)
  function jumpTo(body, v, t) {
    const T = v.duration;
    if (t == null || !isFinite(t)) t = T / 2;
    const video = body.querySelector('video.orig');
    if (video && !video.paused) video.pause();
    if (video && video.duration > 0) video.currentTime = Math.min(Math.max(t / T, 0), 1) * video.duration;
    const media = body.querySelector('.media');
    if (media) media.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ---------- media: the original video ----------
  function mediaBlock(v, q) {
    let vid;
    if (v.video) {
      const note = v.video_from !== 'val' ? Ht`<p class="muted vnote">H.264 copy from the test set; the validation file is MPEG-4 Part 2. <a href="${esc(v.orig)}">Original file</a></p>` : '';
      vid = Ht`<h4>Video</h4><video class="orig" controls muted loop playsinline preload="metadata"></video>${note}`;
    } else {
      vid = Ht`<h4>Video</h4><div class="noscene">MPEG-4 Part 2 (<code>${esc(v.codec)}</code>) does not play in browsers. <a href="${esc(v.orig)}">Original file</a></div>`;
    }
    const tl = tqLabel(q);
    const jump = tl ? Ht`<button type="button" class="tq-btn" data-t="${q.tjump == null ? '' : q.tjump}">t = ${esc(tl)}</button>` : '';
    if (jump) vid += Ht`<div class="viewbar below">${jump}</div>`;
    return Ht`<div class="media"><div>${vid}</div></div>`;
  }

  // ---------- time per step ----------
  // wall time of the question's own run, per pipeline step (answer record "timing"); reused = served from a sibling
  // question's vision memo (SAM 2 propagation, floor masks), cached = this question's file from an earlier run kept
  const STEP_LABEL = { objects: 'Objects', scenegraph_grid: 'Scene graph (grid)', track: 'Tracking', focal: 'Focal' };
  function reuseText(x) {
    if (x.cached) return tr('cached');
    if (!x.reused) return '';
    const sh = x.shared || {};
    const hits = sh.sam2_hits != null ? Ht` (SAM 2 ${sh.sam2_hits}/${sh.sam2_hits + (sh.sam2_misses || 0)})`
      : sh.ground_floor === 'shared' ? ` (${tr('floor')})` : sh.support_masks === 'shared' ? ` (${tr('support masks')})` : '';
    return (x.reused === true ? tr('reused') : tr('partly reused')) + hits;
  }
  // harness steps (answer record timing) and the scene-program steps this unit records (build, fit, answer)
  const SP_STEP_LABEL = { build: 'Scene program', fit: 'Physics-guided Fitting', answer: 'Answer' };
  function timeBlock(q) {
    const t = q.sp.time;
    if (!q.time && !t) return '';
    const marks = q.time ? q.time.steps.map(reuseText) : [], any = marks.some(Boolean);
    const rows = q.time ? q.time.steps.map((x, i) => Ht`<tr${marks[i] ? ' class="reused"' : ''}><td>${esc(STEP_LABEL[x.step] ? tr(STEP_LABEL[x.step]) : x.step)}</td><td class="n">${fmt(x.s, 1)}</td>` +
      (any ? Ht`<td>${esc(marks[i])}</td>` : '') + Ht`</tr>`).join('') +
      Ht`<tr class="ttotal"><td>Harness</td><td class="n">${fmt(q.time.total, 1)}</td>${any ? Ht`<td></td>` : ''}</tr>` : '';
    const sprows = t ? Object.keys(SP_STEP_LABEL).filter((k) => k in t).map((k) => Ht`<tr><td>${tr(SP_STEP_LABEL[k])}</td><td class="n">${fmt(t[k], 1)}</td>${any ? Ht`<td></td>` : ''}</tr>`).join('') : '';
    return Ht`<h4>Time</h4><div class="scroll-x"><table class="tsteps"><tr><th>Step</th><th class="n">s</th>` + (any ? Ht`<th></th>` : '') + Ht`</tr>${rows}${sprows}</table></div>`;
  }

  // ---------- filtering ----------
  function renderList(items) {
    const list = $('val-list'); list.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'qhead';
    head.innerHTML = Ht`<span></span><span>Question</span><span class="qnums sp"><span class="qv">GT</span><span class="qv">Read-out</span><span class="qv">MRA</span><span class="qv">Answer</span></span><span class="qm">MRA</span>`;
    if (items.length) list.appendChild(head);
    const frag = document.createDocumentFragment();
    items.forEach(({ q, v }) => frag.appendChild(questionCard(q, v)));
    list.appendChild(frag);
  }
  function fillSourceSelect(sel, videos) {
    [...new Set(videos.map((v) => v.source))].sort().forEach((x) => { const o = document.createElement('option'); o.textContent = x; sel.appendChild(o); });
  }
  const matchText = (q, needle) => !needle || q.id.toLowerCase().includes(needle) || q.q.toLowerCase().includes(needle);
  function renderVal() {
    const vsrc = $('v-vsrc').value, mf = $('v-mra').value, sort = $('v-sort').value;
    const needle = $('v-q').value.trim().toLowerCase(), vids = DATA.val.vmap;
    const okM = (m) => !mf || (mf === 'lt0.2' ? m < 0.2 : mf === 'lt0.5' ? m < 0.5 : m >= 0.8);
    // setQuestions() is already in set order (DEV20 order, or video then k)
    const items = setQuestions().filter((q) => (!vsrc || vids[q.video].source === vsrc) && okM(fMRA(q)) && matchText(q, needle)).map((q) => ({ q, v: vids[q.video] }));
    const key = { 'mra-asc': fMRA, 'mra-desc': fMRA, 'smra-asc': sMRA, 'smra-desc': sMRA }[sort];
    if (key) items.sort((a, b) => (sort.endsWith('asc') ? 1 : -1) * (key(a.q) - key(b.q)));
    $('v-status').textContent = Ht`${items.length} questions` + (items.length ? Ht` · Read-out ${fmt(mean(items.map((x) => sMRA(x.q))), 3)} · Answer ${fmt(mean(items.map((x) => fMRA(x.q))), 3)}` : '');
    renderList(items);
  }
  const renderAll = () => { valSummary(); renderVal(); };
  // a language switch re-renders the list in place, the open questions stay open
  function rerender(listId, renderFn) {
    const list = $(listId), y = window.scrollY;
    const open = [...list.querySelectorAll('details.qrow[open]')];
    const ids = open.map((x) => x.dataset.id);
    open.forEach((x) => { x.open = false; });
    renderFn();
    list.querySelectorAll('details.qrow').forEach((x) => { if (ids.includes(x.dataset.id)) x.open = true; });
    window.scrollTo(0, y);
  }
  function debounce(fn, ms) { let t = 0; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(Ht`${u}: ${r.status}`); return r.json(); });
  Promise.all([get('data/videos.json'), get('data/comparison_ours.json')]).then(([d, ours]) => {
    DATA = d; OURS = ours;
    d.val.vmap = Object.fromEntries(d.val.videos.map((v) => [v.id, v]));
    fillSourceSelect($('v-vsrc'), d.val.videos);
    $('v-set').addEventListener('change', renderAll);
    ['v-vsrc', 'v-mra', 'v-sort'].forEach((id) => $(id).addEventListener('change', renderVal));
    $('v-q').addEventListener('input', debounce(renderVal, 200));
    renderAll();
    // deep link: #val/<video_id>_q<k> opens that question (switching to All when it is not a DEV20 question)
    function openFromHash() {
      const m = decodeURIComponent(location.hash.slice(1)).match(/^val\/(.+)$/);
      if (!m || !DATA.val.questions.some((q) => q.id === m[1])) return;
      if (!DATA.val.dev.includes(m[1])) $('v-set').value = '';
      ['v-vsrc', 'v-mra', 'v-q'].forEach((id) => { $(id).value = ''; });
      renderAll();
      const card = [...$('val-list').querySelectorAll('details.qrow')].find((x) => x.dataset.id === m[1]);
      card.open = true; card.scrollIntoView();
    }
    openFromHash();
    addEventListener('hashchange', openFromHash);
    if (typeof I18N !== 'undefined') I18N.on(() => rerender('val-list', renderAll));
  }).catch((e) => {
    $('val-list').innerHTML = Ht`<p class="noscene">${esc(e.message)}</p>`;
  });
})();
