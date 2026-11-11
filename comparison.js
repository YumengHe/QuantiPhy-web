// index.html: the method summary of baseline.html (same computation, without the "Reported in original paper" column), from data/baseline.json,
// and the qualitative comparison from data/physmind/index.json.
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  // each method's colour is --method-<key> in site.css, the same on every page
  const dot = (key) => `<span class="mdot" style="background:var(--method-${key})"></span>`;
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  // overall = mean over the four categories of the per-category mean MRA (as in baseline.js)
  const catMean = (qs, key, c) => mean(qs.filter((q) => q.cat === c).map((q) => q.preds[key].mra));
  const macro = (qs, key) => mean(CATS.map((c) => catMean(qs, key, c)).filter((v) => isFinite(v)));

  // ours: data/comparison_ours.json (scores as given, not recomputed); the Questions row is shown only when
  // our question count equals the baselines' question set
  function table(d, ours) {
    const qs = d.questions;
    const head = '<tr><th>Method</th><th class="n">Overall</th>' + CATS.map((c) => `<th class="n">${c}</th>`).join('') + '</tr>';
    const rows = d.methods.map((m) => `<tr><td>${dot(m.key)}<a href="${esc(m.source)}" target="_blank" rel="noopener">${esc(m.label)}</a></td><td class="n"><b>${fmt(macro(qs, m.key))}</b></td>` +
      CATS.map((c) => `<td class="n">${fmt(catMean(qs, m.key, c))}</td>`).join('') + '</tr>').join('');
    const our = ours ? `<tr class="ours"><td>${dot('ours')}${esc(ours.label)} (${esc(ours.version)})</td><td class="n"><b>${fmt(ours.overall)}</b></td>` +
      CATS.map((c) => `<td class="n">${fmt(ours.cats[c])}</td>`).join('') + '</tr>' : '';
    const counts = ours && ours.n !== qs.length ? '' : '<tr><td class="muted">Questions</td><td class="n muted">' + qs.length + '</td>' +
      CATS.map((c) => `<td class="n muted">${qs.filter((q) => q.cat === c).length}</td>`).join('') + '</tr>';
    return head + rows + our + counts;
  }

  // qualitative: the frames where our v1 run drew a question overlay (data/runtime/v1), and PhysMind's posed meshes of the
  // same question objects drawn on the same frame the same way (data/physmind/index.json, tools/render_physmind.py)
  const WHY = { sam3_video_track_labels: 'SAM 3 found no objects', sam3d_objects: 'mesh reconstruction failed' };
  function card(v) {
    const img = (src, alt) => (src ? `<a href="${esc(src)}"><img src="${esc(src)}" alt="${esc(alt)}" loading="lazy"></a>` : '<div class="qc-none muted small">no overlay</div>');
    const head = `<div class="qc-head"><b>${esc(v.video)}</b><span class="muted small">PhysMind: ${v.posed_question_objects} / ${v.question_objects} question objects posed` +
      `${v.failed_stage ? ` · ${esc(WHY[v.failed_stage] || v.failed_stage)}` : ''}</span></div>`;
    const rows = v.frames.map((f) => {
      const pm = f.physmind ? img(f.physmind, `${v.video} PhysMind t = ${f.t} s`) : `<div class="qc-none muted small">${esc(WHY[v.failed_stage] || 'no reconstruction')}</div>`;
      const miss = f.physmind && f.missing.length ? ` · PhysMind has no pose here for ${f.missing.map(esc).join(', ')}` : '';
      return `<div class="qc-pair">${img(f.ours, `${v.video} ours t = ${f.t} s`)}${pm}</div>` +
        `<div class="qc-cap muted small">t = ${fmt(f.t, 2)} s${f.questions.length ? ` · ${f.questions.map((q) => esc(q.replace(v.video + '_', ''))).join(', ')}` : ''}${miss}</div>`;
    }).join('');
    return `<figure class="qc-card">${head}<div class="qc-pair qc-cols muted small"><span>${dot('ours')}Ours (v1)</span><span>${dot('physmind')}PhysMind</span></div>${rows}</figure>`;
  }

  if (typeof module !== 'undefined') { module.exports = { table, card }; return; }
  const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  Promise.all([get('data/baseline.json'), get('data/comparison_ours.json')])
    .then(([d, ours]) => { document.getElementById('quantitative-comparison-table').innerHTML = table(d, ours); });
  get('data/physmind/index.json')
    .then((d) => { document.getElementById('qualitative-comparison-grid').innerHTML = d.videos.map(card).join(''); });
})();
