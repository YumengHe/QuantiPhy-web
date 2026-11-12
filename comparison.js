// index.html: the method summary of baseline.html (same computation, without the "Reported in original paper" column), from data/baseline.json,
// and the qualitative comparison from data/qualitative/index.json.
(function () {
  'use strict';
  // UI text in English; I18N (i18n.js) translates it: Ht for templates, tr() for single strings
  const Ht = typeof I18N !== 'undefined' ? I18N.H : (s, ...v) => s.reduce((r, x, i) => r + v[i - 1] + x);
  const tr = typeof I18N !== 'undefined' ? I18N.t : (s, p) => (p ? s.replace(/\{(\w+)\}/g, (m, k) => (k in p ? p[k] : m)) : s);
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  // each method's colour is --method-<key> in site.css, the same on every page
  const dot = (key) => Ht`<span class="mdot" style="background:var(--method-${key})"></span>`;
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  // overall = mean over the four categories of the per-category mean MRA (as in baseline.js)
  const catMean = (qs, key, c) => mean(qs.filter((q) => q.cat === c).map((q) => q.preds[key].mra));
  const macro = (qs, key) => mean(CATS.map((c) => catMean(qs, key, c)).filter((v) => isFinite(v)));

  // ours: data/comparison_ours.json (tools/build_videos.py), answer_final of every baseline question, scored with the
  // baselines' rule; a baseline question missing from it is an error
  function oursScores(d, ours) {
    const miss = d.questions.filter((q) => !ours.questions[q.id]).map((q) => q.id);
    if (miss.length) throw new Error(Ht`comparison_ours.json lacks ${miss.join(', ')}`);
    const qs = d.questions.map((q) => ({ cat: q.cat, preds: { ours: { mra: ours.questions[q.id].final_mra } } }));
    return { overall: macro(qs, 'ours'), cats: Object.fromEntries(CATS.map((c) => [c, catMean(qs, 'ours', c)])) };
  }
  function table(d, ours) {
    const qs = d.questions;
    const head = Ht`<tr><th>Method</th><th class="n">Overall</th>` + CATS.map((c) => Ht`<th class="n">${c}</th>`).join('') + Ht`</tr>`;
    const rows = d.methods.map((m) => Ht`<tr><td>${dot(m.key)}<a href="${esc(m.source)}" target="_blank" rel="noopener">${esc(m.label)}</a></td><td class="n"><b>${fmt(macro(qs, m.key))}</b></td>` +
      CATS.map((c) => Ht`<td class="n">${fmt(catMean(qs, m.key, c))}</td>`).join('') + Ht`</tr>`).join('');
    const o = oursScores(d, ours);
    const our = Ht`<tr class="ours"><td>${dot('ours')}${esc(tr(ours.label))}</td><td class="n"><b>${fmt(o.overall)}</b></td>` +
      CATS.map((c) => Ht`<td class="n">${fmt(o.cats[c])}</td>`).join('') + Ht`</tr>`;
    const counts = Ht`<tr><td class="muted">Questions</td><td class="n muted">` + qs.length + Ht`</td>` +
      CATS.map((c) => Ht`<td class="n muted">${qs.filter((q) => q.cat === c).length}</td>`).join('') + Ht`</tr>`;
    return head + rows + our + counts;
  }

  // qualitative: data/qualitative/index.json (tools/build_qualitative.py): on each frame PhysMind or Thinking with Spatial Code
  // drew, the final scene program of one of the frame's questions, PhysMind's posed meshes of the question objects and
  // Spatial Code's encoder boxes; a method without a drawing on that frame leaves its cell empty
  const WHY = { sam3_video_track_labels: 'SAM 3 found no objects', sam3d_objects: 'mesh reconstruction failed' };
  const BASES = [['physmind', 'PhysMind'], ['spatialcode', 'Thinking with Spatial Code']];
  function card(v, ours) {
    const img = (src, alt) => (src ? Ht`<a href="${esc(src)}"><img src="${esc(src)}" alt="${esc(alt)}" loading="lazy"></a>` : Ht`<div></div>`);
    const pm = v.physmind;
    const head = Ht`<div class="qc-head"><b>${esc(v.video)}</b><span class="muted small">PhysMind: ${pm.posed_question_objects} / ${pm.question_objects} question objects posed` +
      Ht`${pm.failed_stage ? Ht` · ${esc(WHY[pm.failed_stage] ? tr(WHY[pm.failed_stage]) : pm.failed_stage)}` : ''}</span></div>`;
    const rows = v.frames.map((f) => {
      const miss = BASES.filter(([k]) => f[k] && f[`${k}_missing`].length)
        .map(([k, label]) => ' · ' + tr('{method} has no {what} here for {objects}', { method: label, what: tr(k === 'physmind' ? 'pose' : 'box'), objects: f[`${k}_missing`].map(esc).join(', ') })).join('');
      return Ht`<div class="qc-pair">${img(f.ours, Ht`${v.video} ours t = ${f.t} s`)}${BASES.map(([k, label]) => img(f[k], Ht`${v.video} ${label} t = ${f.t} s`)).join('')}</div>` +
        Ht`<div class="qc-cap muted small">t = ${fmt(f.t, 2)} s${f.questions.length ? Ht` · ${f.questions.map((q) => esc(q.replace(v.video + '_', ''))).join(', ')}` : ''}${miss}</div>`;
    }).join('');
    return Ht`<figure class="qc-card">${head}<div class="qc-pair qc-cols muted small"><span>${dot('ours')}${esc(tr(ours.label))}</span>` +
      BASES.map(([k, label]) => Ht`<span>${dot(k)}${label}</span>`).join('') + Ht`</div>${rows}</figure>`;
  }

  if (typeof module !== 'undefined') { module.exports = { table, card, oursScores }; return; }
  const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  Promise.all([get('data/baseline.json'), get('data/comparison_ours.json'), get('data/qualitative/index.json')])
    .then(([d, ours, qual]) => {
      const draw = () => {
        document.getElementById('quantitative-comparison-table').innerHTML = table(d, ours);
        document.getElementById('qualitative-comparison-grid').innerHTML = qual.videos.map((v) => card(v, ours)).join('');
      };
      draw();
      if (typeof I18N !== 'undefined') I18N.on(draw);
    });
})();
