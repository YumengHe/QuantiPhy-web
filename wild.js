// wild.html: the pipeline on real videos outside the benchmark (no ground truth), one block per video, best aligned
// first. Data: data/wild.json (built by tools/build_wild.py from the run's answers table and its final programs).
(function () {
  'use strict';
  // UI text in English; I18N (i18n.js) translates it: Ht for templates, tr() for single strings
  const Ht = typeof I18N !== 'undefined' ? I18N.H : (s, ...v) => s.reduce((r, x, i) => r + v[i - 1] + x);
  const tr = typeof I18N !== 'undefined' ? I18N.t : (s, p) => (p ? s.replace(/\{(\w+)\}/g, (m, k) => (k in p ? p[k] : m)) : s);
  const GROUPS = {
    ball: ['Ball videos', 'regulation ball diameter'],
    quadruped: ['Quadruped robots', 'published body length'],
    lab: ['Lab recordings', 'standard ceiling light panel length'],
  };
  const STATUS = { failed: 'failed', stopped: 'unfinished' };
  let DATA = null;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function fmtNum(x, p = 4) {
    if (x == null || !isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return Number(x).toExponential(p - 2);
    return String(Number(Number(x).toPrecision(p)));
  }
  const qk = (q) => 'q' + q.unit.split('_q').pop();

  function header() {
    const c = DATA.counts;
    const stat = (label, n) => Ht`<div class="m"><div class="muted">${label}</div><div class="big">${n}</div></div>`;
    $('w-stats').innerHTML = stat(tr('Videos'), c.videos) + stat(tr('Questions'), c.questions) + stat(tr('Answered'), c.answered) +
      stat(tr('Failed'), c.failed) + stat(tr('Unfinished'), c.stopped) + stat(tr('Scene answer determined'), c.determined);
    $('w-groups').innerHTML = Ht`<tr><th>Group</th><th class="n">Videos</th><th>Prior</th></tr>` +
      DATA.groups.map((g) => Ht`<tr><td>${tr(GROUPS[g.key][0])}</td><td class="n">${g.videos}</td>` +
        Ht`<td>${tr(GROUPS[g.key][1])}: ${g.priors.map(esc).join(' · ')}</td></tr>`).join('');
  }

  function answers(q) {
    if (q.status !== 'answered') return Ht`<td colspan="3" class="wstat" title="${esc(q.step || '')}">${tr(STATUS[q.status])}</td>`;
    const f = Ht`<td class="n">${fmtNum(q.final.value)} ${esc(q.final.unit)}</td>`;
    const s = q.scene.determined
      ? Ht`<td class="n">${fmtNum(q.scene.value)} ± ${fmtNum(q.scene.sigma, 2)} ${esc(q.scene.unit)}</td>`
      : Ht`<td class="n muted" title="${esc(q.scene.reason || '')}">${tr('undetermined')}</td>`;
    return f + s + Ht`<td class="n">${fmtNum(q.iou, 2)}</td>`;
  }
  const img = (src) => Ht`<a href="${esc(src)}"><img src="${esc(src)}" alt="" loading="lazy"></a>`;

  function block(v) {
    const priors = [...new Set(v.questions.map((q) => q.prior))];
    const prior = Ht`<ul class="prior">${priors.map((p) => Ht`<li>${esc(p)}</li>`).join('')}</ul>`;
    const table = Ht`<div class="scroll-x"><table class="wq"><tr><th></th><th>Question</th><th class="n">answer_final</th>` +
      Ht`<th class="n">answer_scene</th><th class="n">IoU</th></tr>` +
      v.questions.map((q) => Ht`<tr><td class="qid">${qk(q)}</td><td>${esc(q.q)}</td>${answers(q)}</tr>`).join('') + Ht`</table></div>`;
    const done = v.questions.filter((q) => q.status === 'answered');
    const strips = done.length
      ? done.map((q) => Ht`<div class="wstrip"><span class="qid">${qk(q)}</span><div class="wframes">${q.overlays.map(img).join('')}</div></div>`).join('')
      : Ht`<div class="wstrip"><span class="qid"></span><div class="wframes">${v.frames.map(img).join('')}</div></div>`;
    const plans = done.length
      ? Ht`<details class="sfres"><summary>Plan grid and shapes</summary>` +
        done.map((q) => Ht`<div class="wplan"><span class="qid">${qk(q)}</span><div>${img(q.plan_grid)}${img(q.shapes)}</div></div>`).join('') + Ht`</details>`
      : '';
    return Ht`<section class="wv" id="${esc(v.video)}"><h3>${esc(v.video)}</h3>${prior}${table}${strips}${plans}</section>`;
  }

  function draw() {
    header();
    $('w-list').innerHTML = DATA.videos.map(block).join('');
  }

  fetch('data/wild.json')
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => {
      DATA = d;
      draw();
      if (typeof I18N !== 'undefined') I18N.on(draw);
      const h = decodeURIComponent(location.hash.slice(1));
      if (h && document.getElementById(h)) document.getElementById(h).scrollIntoView();
    })
    .catch((e) => { $('w-status').textContent = tr('Could not load data/wild.json: {e}', { e: String(e) }); });
})();
