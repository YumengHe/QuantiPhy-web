// index.html: the method summary of baseline.html (same columns, same computation), from data/baseline.json.
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  // overall = mean over the four categories of the per-category mean MRA (as in baseline.js)
  const catMean = (qs, key, c) => mean(qs.filter((q) => q.cat === c).map((q) => q.preds[key].mra));
  const macro = (qs, key) => mean(CATS.map((c) => catMean(qs, key, c)).filter((v) => isFinite(v)));

  function table(d) {
    const qs = d.questions;
    const head = '<tr><th>Method</th><th class="n">Overall</th>' + CATS.map((c) => `<th class="n">${c}</th>`).join('') +
      '<th class="n">Reported</th></tr>';
    const rows = d.methods.map((m) => `<tr><td><a href="${esc(m.source)}">${esc(m.label)}</a></td><td class="n"><b>${fmt(macro(qs, m.key))}</b></td>` +
      CATS.map((c) => `<td class="n">${fmt(catMean(qs, m.key, c))}</td>`).join('') +
      `<td class="n muted">${fmt(m.paper)}</td></tr>`).join('');
    const counts = '<tr><td class="muted">Questions</td><td class="n muted">' + qs.length + '</td>' +
      CATS.map((c) => `<td class="n muted">${qs.filter((q) => q.cat === c).length}</td>`).join('') + '<td></td></tr>';
    return head + rows + counts;
  }

  if (typeof module !== 'undefined') { module.exports = { table }; return; }
  fetch('data/baseline.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => { document.getElementById('quantitative-comparison-table').innerHTML = table(d); });
})();
