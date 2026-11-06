// index.html: the method summary of baseline.html (same computation, without the "Reported in original paper" column), from data/baseline.json.
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  // overall = mean over the four categories of the per-category mean MRA (as in baseline.js)
  const catMean = (qs, key, c) => mean(qs.filter((q) => q.cat === c).map((q) => q.preds[key].mra));
  const macro = (qs, key) => mean(CATS.map((c) => catMean(qs, key, c)).filter((v) => isFinite(v)));

  // ours: data/comparison_ours.json (scores as given, not recomputed); the Questions row is shown only when
  // our question count equals the baselines' question set
  function table(d, ours) {
    const qs = d.questions;
    const head = '<tr><th>Method</th><th class="n">Overall</th>' + CATS.map((c) => `<th class="n">${c}</th>`).join('') + '</tr>';
    const rows = d.methods.map((m) => `<tr><td><a href="${esc(m.source)}">${esc(m.label)}</a></td><td class="n"><b>${fmt(macro(qs, m.key))}</b></td>` +
      CATS.map((c) => `<td class="n">${fmt(catMean(qs, m.key, c))}</td>`).join('') + '</tr>').join('');
    const our = ours ? `<tr><td><b>${esc(ours.label)} (${esc(ours.version)})</b></td><td class="n"><b>${fmt(ours.overall)}</b></td>` +
      CATS.map((c) => `<td class="n">${fmt(ours.cats[c])}</td>`).join('') + '</tr>' : '';
    const counts = ours && ours.n !== qs.length ? '' : '<tr><td class="muted">Questions</td><td class="n muted">' + qs.length + '</td>' +
      CATS.map((c) => `<td class="n muted">${qs.filter((q) => q.cat === c).length}</td>`).join('') + '</tr>';
    return head + rows + our + counts;
  }

  if (typeof module !== 'undefined') { module.exports = { table }; return; }
  const get = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  Promise.all([get('data/baseline.json'), get('data/comparison_ours.json')])
    .then(([d, ours]) => { document.getElementById('quantitative-comparison-table').innerHTML = table(d, ours); });
})();
