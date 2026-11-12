// baseline.html: other methods on the QuantiPhy validation set, one row per question as on evaluation.html.
// Data: data/baseline.json (built by tools/build_baseline.py from the methods' prediction CSVs).
(function () {
  'use strict';
  // UI text in English; I18N (i18n.js) translates it: Ht for templates, tr() for single strings
  const Ht = typeof I18N !== 'undefined' ? I18N.H : (s, ...v) => s.reduce((r, x, i) => r + v[i - 1] + x);
  const tr = typeof I18N !== 'undefined' ? I18N.t : (s, p) => (p ? s.replace(/\{(\w+)\}/g, (m, k) => (k in p ? p[k] : m)) : s);
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  let DATA = null;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  // each method's colour is --method-<key> in site.css, the same on every page
  const dot = (key) => Ht`<span class="mdot" style="background:var(--method-${key})"></span>`;
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  function fmtNum(x) {
    if (x == null || !isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return Number(x).toExponential(2);
    return String(Number(x.toPrecision(4)));
  }
  function mraClass(m) { return m >= 0.8 ? 'm4' : m >= 0.6 ? 'm3' : m >= 0.4 ? 'm2' : m >= 0.2 ? 'm1' : 'm0'; }
  function mraBadge(m) { return isFinite(m) ? Ht`<span class="mra ${mraClass(m)}">${fmt(m, 2)}</span>` : Ht`<span class="muted">—</span>`; }
  // our row's label is translated; the other methods' names stay as they are
  const lab = (m) => (m.key === 'ours' ? tr(m.label) : m.label);
  const mraOf = (q, key) => q.preds[key].mra;
  // the benchmark's overall score: mean over the four categories of the per-category mean (as on evaluation.html)
  const catMean = (qs, key, c) => mean(qs.filter((q) => q.cat === c).map((q) => mraOf(q, key)));
  const macro = (qs, key) => mean(CATS.map((c) => catMean(qs, key, c)).filter((v) => isFinite(v)));

  // ---------- summary: one big number per method, then the per-category table ----------
  function summary() {
    const qs = DATA.questions;
    $('b-stats').innerHTML = DATA.methods.map((m) =>
      Ht`<div class="m" style="--mc:var(--method-${m.key})"><div class="muted">${esc(lab(m))}</div><div class="big">${fmt(macro(qs, m.key), 3)}</div></div>`).join('');
    $('b-n').textContent = Ht`${qs.length} questions · ${DATA.videos.length} videos · overall = mean of the four category means`;
    const head = Ht`<tr><th>Method</th><th class="n">Overall</th>` + CATS.map((c) => Ht`<th class="n">${c}</th>`).join('') +
      Ht`<th class="n">Reported in<br>original paper</th></tr>`;
    const counts = Ht`<tr><td class="muted">Questions</td><td class="n muted">` + qs.length + Ht`</td>` +
      CATS.map((c) => Ht`<td class="n muted">${qs.filter((q) => q.cat === c).length}</td>`).join('') + Ht`<td></td></tr>`;
    const link = (m) => m.key === 'ours' ? Ht`<a href="${esc(m.source)}">${esc(lab(m))}</a>` : Ht`<a href="${esc(m.source)}" target="_blank" rel="noopener">${esc(lab(m))}</a>`;
    const rows = DATA.methods.map((m) => Ht`<tr${m.key === 'ours' ? ' class="ours"' : ''}><td>${dot(m.key)}${link(m)}</td><td class="n"><b>${fmt(macro(qs, m.key))}</b></td>` +
      CATS.map((c) => Ht`<td class="n">${fmt(catMean(qs, m.key, c))}</td>`).join('') +
      Ht`<td class="n muted">${fmt(m.paper)}</td></tr>`).join('');
    $('b-table').innerHTML = head + rows + counts;
    const left = DATA.left_out.length ? Ht` Not in the HF validation release, so not run: ${DATA.left_out.map((x) => Ht`<code>${esc(x)}</code>`).join(', ')}.` : '';
    $('b-notes').innerHTML = Ht`Ground truth and questions from the HF validation release (<code>PaulineLi/QuantiPhy-validation</code>); ` +
      tr('MRA as on the Evaluation page. Reported in original paper = the validation score published by the method’s authors; the other columns are our re-run of their released code.') + left + Ht`<br>` +
      DATA.methods.map((m) => Ht`${esc(lab(m))}: ${esc(tr(m.setup))}`).join(Ht`<br>`);
  }

  // ---------- one question ----------
  function summaryRow(q, v, key) {
    const p = q.preds[key];
    const nums = Ht`<span class="qv" data-k="GT">${fmtNum(q.answer)}</span><span class="qv pred" data-k="Pred">${fmtNum(p.pred)}</span>` +
      Ht`<span class="qsrc" title="${esc(p.raw)}">${esc(p.raw)}</span>`;
    return Ht`<summary><img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}"><span class="qmain"><span class="qid">${esc(q.id)}<span class="qver">${esc(q.cat)}</span></span><span class="qtext">${esc(q.q)}</span></span>` +
      Ht`<span class="qnums">${nums}</span><span class="qm">${mraBadge(p.mra)}</span></summary>`;
  }
  function videoBlock(v) {
    if (!v.video) return Ht`<div class="noscene">MPEG-4 Part 2 (<code>${esc(v.codec)}</code>) does not play in browsers. <a href="${esc(v.orig)}">Original file</a></div>`;
    const note = v.video_from !== 'val' ? Ht`<p class="muted vnote">H.264 copy from the test set; the validation file is MPEG-4 Part 2. <a href="${esc(v.orig)}">Original file</a></p>` : '';
    return Ht`<video class="orig" controls muted loop playsinline preload="metadata"></video>${note}`;
  }
  function body(q, v) {
    const lines = [q.prior, q.depth].filter(Boolean).flatMap((s) => String(s).split('\n')).map((s) => s.trim()).filter(Boolean);
    const prior = lines.length ? Ht`<h4>prior</h4><ul class="prior">${lines.map((l) => Ht`<li>${esc(l)}</li>`).join('')}</ul>` : '';
    const table = Ht`<h4>Answers</h4><div class="scroll-x"><table><tr><th>Method</th><th class="n">Pred</th><th class="n">GT</th><th class="n">MRA</th><th>Response</th></tr>` +
      DATA.methods.map((m) => { const p = q.preds[m.key]; return Ht`<tr><td>${dot(m.key)}${esc(lab(m))}</td><td class="n">${fmtNum(p.pred)}</td><td class="n">${fmtNum(q.answer)}</td><td class="n">${mraBadge(p.mra)}</td><td><pre>${esc(p.raw)}</pre></td></tr>`; }).join('') +
      Ht`</table></div>`;
    return Ht`<div class="qfull">${esc(q.q)}</div>${prior}<div class="media"><div><h4>Video</h4>${videoBlock(v)}</div><div>${table}</div></div>`;
  }
  function questionCard(q, v, key) {
    const d = document.createElement('details');
    d.className = 'qrow';
    d.dataset.id = q.id;
    d.innerHTML = summaryRow(q, v, key);
    let built = false;
    d.addEventListener('toggle', () => {
      if (d.open && !built) {
        built = true;
        const b = document.createElement('div');
        b.className = 'qbody';
        b.innerHTML = body(q, v);
        d.appendChild(b);
      }
      // the <video> only holds a src while its question is open (as on evaluation.html)
      const video = d.querySelector('video');
      if (!video) return;
      if (d.open) { video.src = v.video; } else { video.pause(); video.removeAttribute('src'); video.load(); }
    });
    return d;
  }

  // ---------- filtering ----------
  function renderList(items, key) {
    const list = $('b-list'); list.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'qhead';
    head.innerHTML = Ht`<span></span><span>Question</span><span class="qnums"><span class="qv">GT</span><span class="qv">Pred</span><span>Response</span></span><span class="qm">MRA</span>`;
    if (items.length) list.appendChild(head);
    const frag = document.createDocumentFragment();
    items.forEach(({ q, v }) => frag.appendChild(questionCard(q, v, key)));
    list.appendChild(frag);
  }
  const matchText = (q, needle) => !needle || q.id.toLowerCase().includes(needle) || q.q.toLowerCase().includes(needle);
  function render() {
    const key = $('b-method').value, cat = $('b-cat').value, vsrc = $('b-vsrc').value, mf = $('b-mra').value, sort = $('b-sort').value;
    const needle = $('b-q').value.trim().toLowerCase();
    const okM = (m) => !mf || (mf === 'lt0.2' ? m < 0.2 : mf === 'lt0.5' ? m < 0.5 : m >= 0.8);
    const items = DATA.questions.filter((q) => (!cat || q.cat === cat) && (!vsrc || DATA.vmap[q.video].source === vsrc) &&
      okM(mraOf(q, key)) && matchText(q, needle)).map((q) => ({ q, v: DATA.vmap[q.video] }));
    if (sort === 'mra-asc') items.sort((a, b) => mraOf(a.q, key) - mraOf(b.q, key));
    else if (sort === 'mra-desc') items.sort((a, b) => mraOf(b.q, key) - mraOf(a.q, key));
    $('b-status').textContent = Ht`${items.length} questions` + (items.length ? Ht` · mean MRA ${fmt(mean(items.map((x) => mraOf(x.q, key))), 2)}` : '');
    renderList(items, key);
  }
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

  // ours: answer_final of the scene-program run for every question (data/comparison_ours.json, tools/build_videos.py)
  function addOurs(d, ours) {
    const miss = d.questions.filter((q) => !ours.questions[q.id]).map((q) => q.id);
    if (miss.length) throw new Error(Ht`comparison_ours.json lacks ${miss.join(', ')}`);
    d.methods.push({ key: 'ours', label: ours.label, paper: null, source: 'evaluation.html',
      setup: 'answer_final of every question, as on the Evaluation page' });
    d.questions.forEach((q) => { const o = ours.questions[q.id]; q.preds.ours = { pred: o.final, mra: o.final_mra, raw: o.final_source }; });
  }
  const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(Ht`${u}: ${r.status}`); return r.json(); });
  Promise.all([get('data/baseline.json'), get('data/comparison_ours.json')]).then(([d, ours]) => {
    addOurs(d, ours);
    DATA = d;
    d.vmap = Object.fromEntries(d.videos.map((v) => [v.id, v]));
    d.methods.forEach((m) => { const o = document.createElement('option'); o.value = m.key; o.textContent = lab(m); $('b-method').appendChild(o); });
    [...new Set(d.videos.map((v) => v.source))].sort().forEach((x) => { const o = document.createElement('option'); o.textContent = x; $('b-vsrc').appendChild(o); });
    summary();
    ['b-method', 'b-cat', 'b-vsrc', 'b-mra', 'b-sort'].forEach((id) => $(id).addEventListener('change', render));
    $('b-q').addEventListener('input', debounce(render, 200));
    render();
    if (typeof I18N !== 'undefined') I18N.on(() => rerender('b-list', () => { [...$('b-method').options].forEach((o, i) => { o.textContent = lab(d.methods[i]); }); summary(); render(); }));
  }).catch((e) => {
    $('b-list').innerHTML = Ht`<p class="noscene">${esc(e.message)}</p>`;
  });
})();
