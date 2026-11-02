// evaluation.html: one row per question (unit <video_id>_q<k>). Data: data/videos.json (built by tools/build_videos.py).
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const SRC_LABEL = { execution_failed: 'failed' };
  const SHAPE_LABEL = { mesh: 'mesh (bounding box)' };
  const PALETTE = ['#3b6fd8', '#d9622b', '#2e9b5f', '#b03fa8', '#c79a12', '#1b9aaa', '#d6455d', '#6d5bd0', '#7a8b2a', '#8a5a3c'];
  let DATA = null;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  // seconds as m:ss (under a minute: "N s")
  function fmtSec(x) {
    if (x == null || !isFinite(x)) return '';
    const r = Math.round(x);
    return r < 60 ? `${r} s` : `${Math.floor(r / 60)}:${String(r % 60).padStart(2, '0')}`;
  }
  const fmt = (x, d = 3) => (x == null || !isFinite(x) ? '—' : Number(x).toFixed(d));
  function fmtNum(x) {
    if (x == null || !isFinite(x)) return '—';
    const a = Math.abs(x);
    if (a !== 0 && (a < 0.01 || a >= 1e5)) return Number(x).toExponential(2);
    return String(Number(x.toPrecision(4)));
  }
  function mraClass(m) { return m >= 0.8 ? 'm4' : m >= 0.6 ? 'm3' : m >= 0.4 ? 'm2' : m >= 0.2 ? 'm1' : 'm0'; }
  function mraBadge(m) { return isFinite(m) ? `<span class="mra ${mraClass(m)}">${fmt(m, 2)}</span>` : '<span class="muted">—</span>'; }
  // the benchmark's overall score: mean over its four categories of the per-category mean (same as pipeline/metrics.evaluate)
  function macro(qs) {
    const vals = CATS.map((c) => mean(qs.filter((q) => q.cat === c).map((q) => q.mra))).filter((v) => isFinite(v));
    return mean(vals);
  }
  // questions of the selected set : DEV10 = docs/DEV10.md in its order, All = every question of the run
  function setQuestions() {
    if ($('v-set').value !== 'dev') return DATA.val.questions;
    const byId = Object.fromEntries(DATA.val.questions.map((q) => [q.id, q]));
    return DATA.val.dev.map((id) => byId[id]);
  }
  // ---------- summary of the selected set: DEV10 mean MRA, All the benchmark's overall (macro) MRA ----------
  function valSummary() {
    const all = setQuestions(), dev = $('v-set').value === 'dev';
    $('val-label').textContent = dev ? 'Mean MRA' : 'Overall MRA';
    $('val-head').textContent = dev ? fmt(mean(all.map((q) => q.mra)), 2) : fmt(macro(all), 4);
    const counts = DATA.val.runs.map((r) => [r.version, all.filter((q) => q.version === r.version).length]).filter(([, n]) => n);
    const vers = counts.length > 1 ? counts.map(([t, n]) => `${t} ${n}`) : counts.map(([t]) => t);
    $('val-n').textContent = [`${all.length} questions`, `${new Set(all.map((q) => q.video)).size} videos`, ...vers].join(' · ');
  }

  // ---------- one question ----------
  // the artifacts of the question, all from the run of its version (val.runs): with unit "video" the scene of its prior
  // group and the video's scene graph and tracks of that run, with unit "question" its own
  function artifacts(q, v) {
    const unit = DATA.val.runmap[q.version].unit, run = v.runs[q.version];
    if (unit === 'question') return { scene: q.scene, sgraph: q.sgraph, tracks: q.tracks };
    if (unit !== 'video') throw new Error(`unknown processing unit ${unit}`);
    if (!run) throw new Error(`${q.id}: ${v.id} has no ${q.version} run`);
    const group = q.group == null ? null : run.groups.find((g) => g.group === q.group);
    if (q.group != null && !group) throw new Error(`${q.id}: group ${q.group} not in ${v.id} (${q.version})`);
    return { scene: group ? group.scene : null, sgraph: run.sgraph, tracks: run.tracks };
  }
  function summaryRow(q, v) {
    const nums = `<span class="qv" data-k="GT">${fmtNum(q.answer)}</span>` +
      `<span class="qv pred" data-k="Pred">${q.failed && q.failed.length ? '' : fmtNum(q.pred)}</span>` +
      (q.failed && q.failed.length ? `<span class="qsrc qfailed">${q.failed.map(esc).join(' · ')}</span>`
        : `<span class="qsrc">${esc(SRC_LABEL[q.src] || q.src)}</span>`) +
      `<span class="qv qt"${q.time ? ' data-k="Time"' : ''}>${q.time ? fmtSec(q.time.total) : ''}</span>`;
    return `<summary><img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}"><span class="qmain"><span class="qid">${esc(q.id)}<span class="qver">${esc(q.version)}</span></span><span class="qtext">${esc(q.q)}</span></span>` +
      `<span class="qnums">${nums}</span><span class="qm">${mraBadge(q.mra)}</span></summary>`;
  }
  function questionCard(q, v) {
    const d = document.createElement('details');
    d.className = 'qrow';
    d.dataset.id = q.id;
    d.innerHTML = summaryRow(q, v);
    let built = false, viewer = null, video = null, view = null;
    const onToggle = () => {
      if (d.open && !built) {
        built = true;
        const a = artifacts(q, v);
        view = { ...v, scene: a.scene };
        const body = document.createElement('div');
        body.className = 'qbody';
        body.innerHTML = priorBlock(q) + depthBindings(q.depth_bindings) + mediaBlock(view, q) + overlayBlock(q, view) + trackBlock(a.tracks, view, q, a.sgraph) +
          sceneGraphTree(a.sgraph) + refmeasureBlock(q.refmeasure) + fitBlock(q, view) + roundsBlock(q.rounds) + timeBlock(q);
        d.appendChild(body);
        video = body.querySelector('video');
        tqMarks(body, view, q);
        body.addEventListener('click', (e) => {
          const b = e.target.closest('.tq-btn');
          if (b) jumpTo(body, view, b.dataset.t === '' ? null : Number(b.dataset.t));
        });
      }
      // the <video> only holds a src while its question is open, so collapsed rows keep no connections or buffers
      if (video) {
        // preload="auto" + a tiny seek decodes the first frame right away, so the overlay canvas can draw it
        if (d.open && !video.getAttribute('src')) {
          video.preload = 'auto'; video.src = v.video;
          video.addEventListener('loadedmetadata', () => { if (video.readyState < 2) video.currentTime = 0.001; }, { once: true });
        }
        if (!d.open) { video.pause(); video.removeAttribute('src'); video.load(); }
      }
      if (view && view.scene) {
        if (d.open && !viewer) viewer = startViewer(d.querySelector('.viewer'), d.querySelector('.timebar'), view, video);
        if (!d.open && viewer) { viewer.then((x) => x && x.dispose()); viewer = null; }
      }
    };
    d.addEventListener('toggle', onToggle);
    return d;
  }

  // the question's own prior: text facts and distance anchors, one per line
  function priorBlock(q) {
    const lines = [q.prior, q.depth].filter(Boolean).flatMap((s) => String(s).split('\n')).map((s) => s.trim()).filter(Boolean);
    return `<div class="qfull">${esc(q.q)}<span class="qver">${esc(q.version)}</span></div>` + (lines.length ? `<h4>prior</h4><ul class="prior">${lines.map((l) => `<li>${esc(l)}</li>`).join('')}</ul>` : '');
  }

  // the depth statements of the prior (rule C14): each bound one's node and what the scene used it for, each
  // unbound one with its reason; a statement the scene did not use is muted
  function depthBindings(db) {
    if (!db) return '';
    const used = db.bound.filter((b) => b.by).length;
    return `<div class="scroll-x"><table class="depthb"><tr><th><code>depth_bindings</code></th><th colspan="3">${db.given} given · ${db.bound.length} bound · ${used} used</th></tr>` +
      db.bound.map((b) => `<tr class="${b.by ? '' : 'muted'}"><td>${esc(b.name)}</td><td>${esc(b.node)}${b.end ? ` <span class="muted">${esc(b.end)}</span>` : ''}</td>` +
        `<td>${esc(b.as)}</td><td>${b.by ? `<code>${esc(b.by)}</code> · ` : ''}${esc(b.why)}</td></tr>`).join('') +
      db.unbound.map((b) => `<tr class="bad"><td>${esc(b.name)}</td><td>—</td><td>unbound</td><td>${esc(b.reason)}</td></tr>`).join('') +
      '</table></div>';
  }

  // ---------- question time t_q ----------
  const tqLabel = (q) => (q.tq.length ? q.tq.map((x) => fmt(x, 2)).join('–') + ' s' : q.tjump != null ? `≈${fmt(q.tjump, 2)} s` : '');
  // IoU < 0.3 or centre offset > 0.5 box sizes: the fitted object is not where the track is at t_q
  const alignBad = (a) => (a.iou != null && a.iou < 0.3) || (a.dc != null && a.dc > 0.5);
  const alignWarn = (a) => (a.iou != null && a.iou < 0.6) || (a.dc != null && a.dc > 0.25);
  function alignCell(a) {
    if (!a) return '<td class="n"><span class="muted">—</span></td>';
    const title = a.t != null ? `t = ${fmt(a.t, 3)} s${a.frame != null ? ` (frame ${a.frame})` : ''}` : '';
    return `<td class="n${alignBad(a) ? ' align-bad' : alignWarn(a) ? ' align-warn' : ''}" title="${esc(title)}">` +
      [a.iou != null ? 'IoU ' + fmt(a.iou, 2) : '', a.dc != null ? 'dc ' + fmt(a.dc, 2) : ''].filter(Boolean).join(' · ') + '</td>';
  }
  // scene time span of a viewer: the video duration
  function sceneT(v) { return v.duration; }
  function tqMarks(body, v, q) {
    const marks = body.querySelector('.tq-marks');
    if (!marks) return;
    const T = sceneT(v);
    q.tq.forEach((t) => {
      if (t > T + 1e-6) return;
      const m = document.createElement('span');
      m.className = 'tq-mark'; m.style.left = `${(100 * Math.min(Math.max(t / T, 0), 1)).toFixed(2)}%`;
      m.title = `${fmt(t, 2)} s`;
      m.onclick = () => jumpTo(body, v, t);
      marks.appendChild(m);
    });
  }
  function jumpTo(body, v, t) {
    const T = sceneT(v);
    if (t == null || !isFinite(t)) t = T / 2;
    const cam = body.querySelector('button[data-view="cam"]');
    if (cam && cam.getAttribute('aria-pressed') !== 'true') cam.click();
    const ov = body.querySelector('input.overlay');
    if (ov && !ov.disabled && !ov.checked) { ov.checked = true; ov.dispatchEvent(new Event('change')); }
    const video = body.querySelector('video.orig');
    if (video && !video.paused) video.pause();
    const slider = body.querySelector('.timebar input');
    if (slider) { slider.value = String(Math.round(1000 * Math.min(Math.max(t / T, 0), 1))); slider.dispatchEvent(new Event('input')); }
    else if (video && video.duration > 0) video.currentTime = Math.min(Math.max(t / T, 0), 1) * video.duration;
    const media = body.querySelector('.media');
    if (media) media.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ---------- media: original video and the 3D scene ----------
  function mediaBlock(v, q) {
    let vid;
    if (v.video) {
      const note = v.video_from !== 'val' ? `<p class="muted vnote">H.264 copy from the test set; the validation file is MPEG-4 Part 2. <a href="${esc(v.orig)}">Original file</a></p>` : '';
      vid = `<h4>Video</h4><video class="orig" controls muted loop playsinline preload="metadata"></video>${note}`;
    } else {
      vid = `<h4>Video</h4><div class="noscene">MPEG-4 Part 2 (<code>${esc(v.codec)}</code>) does not play in browsers. <a href="${esc(v.orig)}">Original file</a></div>`;
    }
    const tl = tqLabel(q);
    const jump = tl ? `<button type="button" class="tq-btn" data-t="${q.tjump == null ? '' : q.tjump}">t = ${esc(tl)}</button>` : '';
    let scene = '';
    if (v.scene) {
      scene = '<div><h4>3D scene</h4>' +
        '<div class="viewbar"><span class="seg" role="group" aria-label="View"><button type="button" data-view="free" aria-pressed="false">Free</button>' +
        '<button type="button" data-view="cam" aria-pressed="true">Camera</button></span>' +
        (v.video ? '<label class="off"><input type="checkbox" class="overlay" checked disabled> Overlay video</label>' : '') + jump + '</div>' +
        '<div class="viewer"><div class="msg">Loading three.js…</div></div>' +
        '<div class="timebar"><button type="button" class="play">Play</button><span class="tq-track"><input type="range" min="0" max="1000" value="0" aria-label="Time"><span class="tq-marks"></span></span><span class="tlabel">t = 0.00 s</span></div></div>';
    } else if (jump) {
      vid += `<div class="viewbar" style="margin-top:6px">${jump}</div>`;
    }
    return `<div class="media"><div>${vid}</div>${scene}</div>`;
  }

  // overlays rendered by the pipeline from the saved scene at the question's times
  // with a support.horizon record: the visible horizon row and the depth-model plane's vanishing row as two thin lines
  function overlayBlock(q, v) {
    const ims = q.overlays;
    if (!ims.length) return '';
    const hc = v.scene && v.scene.horizon && v.scene.horizon.check, H = v.size && v.size[1];
    const rows = hc && H && hc.row_horizon != null && hc.row_plane != null
      ? `<span class="hrow hrow-horizon" style="top:${(100 * hc.row_horizon) / H}%"></span><span class="hrow hrow-plane" style="top:${(100 * hc.row_plane) / H}%"></span>` : '';
    return '<h4>Overlays</h4><div class="runtime-frames">' + ims.map((im) => im.path
      ? `<figure class="runtime-frame"><a class="ovimg" href="${esc(im.path)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(im.path)}" alt="t = ${esc(im.t)} s">${rows}</a>` +
        `<figcaption>t = ${fmt(im.t, 2)} s</figcaption></figure>`
      : `<p class="fwarn">t = ${fmt(im.t, 2)} s: ${esc(im.unavailable)}</p>`).join('') + '</div>';
  }

  // tracked intervals per object on the video's time axis; the question's times as vertical lines; a scene-graph edge
  // that is an event (rule scenegraph.flight_end, zero length allowed) is a tick on both objects' rows
  function trackBlock(tracks, v, q, sg) {
    if (!tracks || !Object.keys(tracks).length) return q.identity || q.extents || q.presence ? '<h4>Tracks</h4>' + presenceBlock(q.presence) + identityBlock(q.identity) + extentsBlock(q.extents) : '';
    const T = v.duration;
    const order = v.scene ? v.scene.objects.map((o) => o.name) : [];
    const pct = (t) => `${(100 * Math.min(Math.max(t / T, 0), 1)).toFixed(2)}%`;
    const lines = q.tq.filter((t) => t <= T + 1e-6).map((t) => `<span class="trk-tq" style="left:${pct(t)}"></span>`).join('');
    const evs = sg ? [...sg.edges.map((e) => ({ ...e, unknown: false })), ...(sg.unknown || []).map((u) => ({ ...u, from: u.obj, unknown: true }))]
      .filter((e) => e.event && e.t0 != null) : [];
    const ticks = (name) => evs.filter((e) => e.from === name || e.to === name).map((e) =>
      `<span class="trk-ev${e.unknown ? ' unk' : ''}" style="left:${pct(e.t0)}" title="${esc(`${e.from} ${e.rel}${e.to ? ' ' + e.to : ''} ${span(e.t0, e.t1)} · ${e.event}`)}"></span>`).join('');
    const te = q.track_ends, fps = te ? te.fps : 0;
    // frames a track overlaps another instance of its class (rule track.visible_span), as runs under its bar
    const overlap = (name) => (te && te.tracks[name] ? runsOf(te.tracks[name].overlap_same_class) : []).map(([a, b]) =>
      `<span class="trk-ov" style="left:${pct(a / fps)};width:calc(${pct(b / fps)} - ${pct(a / fps)} + 2px)" title="${a === b ? a : `${a}–${b}`}"></span>`).join('');
    const rows = Object.entries(tracks).map(([name, ivs]) => {
      const i = order.indexOf(name), col = i >= 0 ? PALETTE[i % PALETTE.length] : 'var(--muted)';
      const segs = (ivs || []).map(([a, b]) => `<span class="trk-seg" style="left:${pct(a)};width:calc(${pct(b)} - ${pct(a)} + 2px);background:${col}" title="${fmt(a, 2)}–${fmt(b, 2)} s"></span>`).join('');
      return `<div class="trk"><span class="trk-n" title="${esc(name)}">${esc(name)}</span><span class="trk-bar">${segs}${overlap(name)}${lines}${ticks(name)}</span></div>`;
    }).join('');
    return `<h4>Tracks</h4><div class="trk-wrap">${rows}<div class="trk"><span class="trk-n"></span><span class="trk-axis"><span>0</span><span>${fmt(T, 2)} s</span></span></div></div>` +
      trackEndsTable(te) + presenceBlock(q.presence) + identityBlock(q.identity) + extentsBlock(q.extents);
  }
  // each track's first and last frame with why it starts and ends there, and its same-class overlap frames
  function trackEndsTable(te) {
    if (!te) return '';
    const end = (e) => `<td class="n">${e.frame}</td><td title="${esc(e.why)}"><code>${esc(e.reason)}</code></td>`;
    return '<div class="scroll-x"><table class="sgrel trkends"><tr><th><code>track.visible_span</code></th><th class="n">Start</th><th></th><th class="n">End</th><th></th>' +
      '<th><code>overlap_same_class</code></th></tr>' +
      Object.entries(te.tracks).map(([name, t]) => `<tr><td>${esc(name)}</td>${end(t.ends.start)}${end(t.ends.end)}` +
        `<td>${t.overlap_same_class.length ? frameRuns(t.overlap_same_class) : ''}</td></tr>`).join('') + '</table></div>';
  }
  // the inventory's presence verdict per object (rules decisions.no_conf, presence.split_vote): the grid votes, the
  // zoom re-check's votes (an unknown vote is a third state, not "not visible"), the rule, its reason, recorded errors
  function presenceBlock(p) {
    if (!p) return '';
    const hasWhy = p.some((x) => x.reason != null || x.errors), hasZoom = p.some((x) => x.zoom);
    const zoom = (z) => z.map((x) => `<span class="${x === 'unknown' ? 'zunk' : ''}">${esc(x)}</span>`).join(' · ');
    return `<div class="scroll-x"><table class="sgrel presence"><tr><th>Object</th><th>Present</th><th>Votes</th>${hasZoom ? '<th>Zoom</th>' : ''}<th>Rule</th>${hasWhy ? '<th></th>' : ''}</tr>` +
      p.map((x) => `<tr><td>${esc(x.name)}</td><td class="${x.present === 'absent' ? 'bad' : x.present == null ? 'muted' : ''}">${esc(x.present == null ? 'unknown' : x.present)}</td>` +
        `<td>${x.votes.map(esc).join(' · ')}</td>${hasZoom ? `<td>${x.zoom ? zoom(x.zoom) : ''}</td>` : ''}<td><code>${esc(x.rule)}</code></td>` +
        (hasWhy ? `<td>${esc(x.reason || '')}${(x.errors || []).map((e) => `<div class="bad"><code>${esc(e.where)}</code> ${esc(e.type)}: ${esc(e.msg)}</div>`).join('')}</td>` : '') + '</tr>').join('') +
      '</table></div>';
  }
  // frames and set-of-frames as compact ranges: [27, 28, 29, 30] -> 27–30
  function runsOf(fs) {
    const out = [];
    fs.forEach((f) => {
      const r = out[out.length - 1];
      if (r && f === r[1] + 1) r[1] = f; else out.push([f, f]);
    });
    return out;
  }
  function frameRuns(fs) {
    return runsOf(fs).map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(', ');
  }
  // per tracked object its frames and overlay-ring frames (extents2d, rule extent.overlay), and per dimension and
  // anchor the blur-free extent the scene used (rule extent.static): value ± σ, frames used, frames rejected with why
  function extentsBlock(e) {
    if (!e) return '';
    const anchor = (a) => `<td>${esc(a.dim)}</td><td>${esc(a.label)}</td>` +
      `<td class="n" title="${esc(a.how)}">${plusMinus(a.value, a.sigma)} px</td>` +
      `<td class="n" title="${esc(a.sigma_mask_how)}">${fmtNum(a.sigma_mask)} px</td><td class="n">${frameRuns(a.used)}</td>` +
      `<td>${a.rejected.map((r) => `<div class="bad">f${r.frame} · ${fmtNum(r.extent)} px · ${fmtNum(r.speed)} px/frame · ${esc(r.why)}</div>`).join('')}</td>`;
    const rows = e.rows.map((x) => {
      const n = Math.max(x.at.length, 1);
      const head = `<td rowspan="${n}">${esc(x.obj)}${x.kind ? ` <span class="muted">${esc(x.kind)}</span>` : ''}</td>` +
        `<td class="n" rowspan="${n}">${x.n != null ? `${x.n} <span class="muted">(${x.frames[0]}–${x.frames[1]})</span>` : ''}</td>` +
        `<td class="n${x.overlay ? ' bad' : ''}" rowspan="${n}">${x.overlay != null ? `${x.overlay}${x.ring != null ? ` · ring ${fmtNum(x.ring)} px` : ''}` : ''}</td>`;
      if (!x.at.length) return `<tr>${head}<td></td><td></td><td></td><td></td><td></td><td></td></tr>`;
      return x.at.map((a, i) => `<tr${i ? '' : ' class="xfirst"'}>${i ? '' : head}${anchor(a)}</tr>`).join('');
    }).join('');
    return `<div class="scroll-x"><table class="extents"><tr><th>Object</th><th class="n">Frames</th><th class="n">Overlay</th>` +
      '<th>Dim</th><th>At</th><th class="n"><code>extent.static</code></th><th class="n">σ mask</th><th class="n">Frames used</th><th>Rejected</th></tr>' +
      rows + '</table></div>';
  }
  // the track identity judge per object: every candidate in the order judged (static / verify / class_size /
  // presence / border), the accepted one carrying the accent bar; an untracked object spans the row with its reason
  function identityBlock(idt) {
    if (!idt) return '';
    const dimsText = (d) => (d ? ['candidate', 'ref'].filter((k) => d[k] !== 'class').map((k) => `${k === 'ref' ? '' : 'candidate '}${d[k].replace(/\+-/g, '±')}`).join('; ') : '');
    const sizeCell = (c) => c.size.map((x) => `<div class="${x.pass ? '' : 'bad'}">${fmtNum(x.ratio)} ${interval(x.bounds[0], x.bounds[1])} ${esc(x.ref)}` +
      `${dimsText(x.dims) ? ` <span class="muted">(${esc(dimsText(x.dims))})</span>` : ''}</div>`).join('');
    const rows = Object.entries(idt).map(([name, v]) => {
      const n = v.candidates.length + (v.untracked ? 1 : 0);
      const cand = v.candidates.map((c, i) => {
        const rej = c.outcome !== 'accepted', check = rej ? c.outcome.replace(/^rejected:\s*/, '') : '';
        // the failing check's own column is marked; the reason text only where no column carries it
        const why = c.why && !['static', 'verify', 'class_size'].includes(check) ? `<div>${esc(c.why)}</div>` : '';
        const bad = (k) => (check === k ? ' bad' : '');
        return `<tr class="${i === v.chosen ? 'chosen' : ''}${rej ? ' rej' : ''}">` + (i === 0 ? `<td rowspan="${n}" class="iobj">${esc(name)}</td>` : '') +
          `<td class="n">${i}</td><td>${esc(c.source)}${c.frame != null ? ` <span class="muted">f${c.frame}</span>` : ''}</td><td class="n">${c.n}</td>` +
          `<td class="n${bad('static')}">${c.px != null ? fmtNum(c.px) : ''}</td><td class="${bad('verify')}">${c.votes ? esc(c.votes.join(' ')) : ''}</td><td class="isize">${sizeCell(c)}</td>` +
          `<td class="n${bad('presence')}">${c.span ? `${c.span[0]}–${c.span[1]}` : ''}</td><td class="n${bad('border')}">${c.border != null ? fmt(c.border, 2) : ''}</td>` +
          `<td class="${rej ? 'bad' : 'ok'}">${esc(check || c.outcome)}${why}</td></tr>`;
      }).join('');
      const un = v.untracked ? `<tr class="untracked">${v.candidates.length ? '' : `<td class="iobj">${esc(name)}</td>`}<td colspan="9" class="bad">${esc(v.untracked)}</td></tr>` : '';
      return cand + un;
    }).join('');
    return '<div class="scroll-x"><table class="ident"><tr><th>Object</th><th class="n">#</th><th>Candidate</th><th class="n">Frames</th><th class="n">Moves (px)</th>' +
      `<th>Verify</th><th>Class size</th><th class="n">Present</th><th class="n">Border</th><th>Outcome</th></tr>${rows}</table></div>`;
  }

  // ---------- time axis of the 3D viewer ----------
  // Linked to the <video>: scene time is the same fraction of T as currentTime is of the video's duration
  // (the H.264 test copy of a validation video has the same frames but sometimes a different nominal fps).
  function linkTime(bar, video, T, setTime, signal) {
    const slider = bar.querySelector('input'), label = bar.querySelector('.tlabel'), play = bar.querySelector('.play');
    let raf = 0, t0 = 0, playing = false;
    const show = (t) => { setTime(t); label.textContent = `t = ${t.toFixed(2)} s`; };
    const linked = () => video && !video.error && video.duration > 0;
    const fromVideo = () => {
      const fr = Math.min(1, video.currentTime / video.duration);
      slider.value = String(Math.round(fr * 1000)); show(fr * T);
    };
    function tick(now) {
      if (linked()) {
        if (video.paused) return;
        fromVideo();
      } else {
        if (!playing) return;
        const t = (((now - t0) / 1000) % T);
        slider.value = String(Math.round((t / T) * 1000));
        show(t);
      }
      raf = requestAnimationFrame(tick);
    }
    slider.oninput = () => {
      const fr = slider.value / 1000;
      show(fr * T);
      if (linked()) video.currentTime = fr * video.duration;
    };
    play.onclick = () => {
      if (linked()) { if (video.paused) video.play(); else video.pause(); return; }
      playing = !playing; play.textContent = playing ? 'Pause' : 'Play';
      if (playing) { t0 = performance.now() - (slider.value / 1000) * T * 1000; raf = requestAnimationFrame(tick); } else cancelAnimationFrame(raf);
    };
    if (video) {
      const o = { signal };
      video.addEventListener('play', () => { play.textContent = 'Pause'; cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); }, o);
      video.addEventListener('pause', () => { play.textContent = 'Play'; cancelAnimationFrame(raf); if (linked()) fromVideo(); }, o);
      video.addEventListener('seeked', () => { if (linked() && video.paused) fromVideo(); }, o);
    }
    return {
      now: () => (slider.value / 1000) * T,
      redraw: () => show((slider.value / 1000) * T),
      stop() { playing = false; cancelAnimationFrame(raf); slider.oninput = null; play.onclick = null; play.textContent = 'Play'; },
    };
  }

  // ---------- scene graph (tools/build_videos.sgraph) drawn as a layered tree ----------
  // one box per object / support / hand, each resting on its first parent in time (parent below, child above, as in the
  // scene); later parents are extra edges (one that skips a level runs through a slot beside the parent's child, where
  // its label sits). Objects that are only ever airborne hang from an airborne root in the top row; an airborne span (or a relation without a target) of an object that also rests on
  // something is a short stub to the left of its box.
  let sgCtx = null;
  const sgTextW = (s, px, bold) => {
    if (!sgCtx) sgCtx = document.createElement('canvas').getContext('2d');
    sgCtx.font = `${bold ? '600 ' : ''}${px}px system-ui, sans-serif`;
    return sgCtx.measureText(s).width;
  };
  function sceneGraphTree(sg) {
    if (!sg) return '';
    const tree = sg.edges.length && sg.nodes.length ? sceneGraphSvg(sg) : '';
    const rel = sg.releases ? '<div class="scroll-x"><table class="sgrel"><tr><th>Object</th><th class="n">Release (s)</th><th>Rule</th><th></th></tr>' +
      sg.releases.map((r) => `<tr><td>${esc(r.obj)}</td><td class="n">${r.t[0] === r.t[1] ? fmt(r.t[0], 3) : `${fmt(r.t[0], 3)}–${fmt(r.t[1], 3)}`}</td>` +
        `<td><code>${esc(r.rule.rule)}</code> lo ${esc(r.rule.lo)} · hi ${esc(r.rule.hi)}</td>` +
        `<td>${esc(r.why || '')}</td></tr>`).join('') + '</table></div>' : '';
    const lau = sg.launches ? '<div class="scroll-x"><table class="sgrel"><tr><th>Object</th><th class="n">Launch (s)</th><th>Rule</th><th></th></tr>' +
      sg.launches.map((r) => `<tr><td>${esc(r.obj)}</td><td class="n">${fmt(r.t[0], 3)}–${fmt(r.t[1], 3)}</td>` +
        `<td><code>state.launch</code> off ${esc(r.support)} · frames ${r.last_contact[0]} → ${r.first_no_contact[0]}${r.inserted ? ' · inserted' : ''}</td>` +
        `<td>${esc(r.why)}${r.testable ? '' : `<div class="muted">untestable · ${esc(r.untestable_why)}</div>`}</td></tr>`).join('') + '</table></div>' : '';
    const unt = sg.untracked ? '<div class="scroll-x"><table class="sgrel"><tr><th>Untracked</th><th></th></tr>' +
      sg.untracked.map(([n, why]) => `<tr><td>${esc(n)}</td><td class="bad">${esc(why)}</td></tr>`).join('') + '</table></div>' : '';
    const rw = sg.rewritten ? '<div class="scroll-x"><table class="sgrel"><tr><th>Object</th><th>Edge</th><th>Was</th><th>Rule</th></tr>' +
      sg.rewritten.map((e) => `<tr><td>${esc(e.obj)}</td><td>${esc(e.rel)}${e.to ? ` ${esc(e.to)}` : ''} ${span(e.t0, e.t1)}</td>` +
        `<td>${esc(e.was.rel)}${e.was.to ? ` ${esc(e.was.to)}` : ''}</td><td><code>${esc(e.rule)}</code></td></tr>`).join('') + '</table></div>' : '';
    const tested = sg.edges.filter((e) => e.test);
    const edgeText = (e) => `${esc(e.rel)}${e.to ? ` ${esc(e.to)}` : ''} ${span(e.t0, e.t1)}`;
    const hasSrc = sg.edges.some((e) => e.source), hasEv = sg.edges.some((e) => e.event);
    const evCell = (e) => (hasEv ? `<td>${e.event ? `<code>${esc(e.event)}</code>` : ''}</td>` : '');
    const listed = sg.edges.filter((e) => e.test || e.source || e.event);
    const ts = listed.length ? `<div class="scroll-x"><table class="sgrel"><tr><th>Object</th><th>Edge</th>${hasSrc ? '<th>Source</th>' : ''}${hasEv ? '<th>Event</th>' : ''}<th>Test</th><th></th><th></th></tr>` +
      listed.map((e) => `<tr><td>${esc(e.from)}</td><td>${edgeText(e)}</td>${hasSrc ? `<td>${esc(e.source || '')}</td>` : ''}${evCell(e)}` +
        (e.test ? `<td><code>${esc(e.test.name)}</code></td>${testCells(e.test)}` : '<td></td><td></td><td></td>') + '</tr>' +
        (e.test && e.test.label ? e.test.label.map((x) => `<tr class="absorbed"><td></td><td></td>${hasSrc ? '<td></td>' : ''}${hasEv ? '<td></td>' : ''}<td><code>${esc(x.name)}</code></td>${testCells(x)}</tr>`).join('') : '') +
        (e.absorbed || []).map((x) => `<tr class="absorbed"><td></td><td>↳ ${edgeText(x)}</td>${hasSrc ? '<td></td>' : ''}${hasEv ? '<td></td>' : ''}<td><code>${esc(x.test.name)}</code></td>${testCells(x.test)}</tr>`).join('')).join('') +
      '</table></div>' : '';
    const cf = sg.conflicts ? '<div class="scroll-x"><table class="sgrel"><tr><th>Object</th><th><code>scenegraph.overlap</code></th><th>Edges</th><th>Source</th><th></th></tr>' +
      sg.conflicts.map((c) => `<tr><td>${esc(c.obj)}</td><td class="bad">${span(c.t0, c.t1)}</td><td>${c.edges.map(edgeText).join('<br>')}</td>` +
        `<td>${esc(c.source)}</td><td>${esc(c.why)}</td></tr>`).join('') + '</table></div>' : '';
    const st = sg.self_test ? `<p class="facts"><code>builder.self_test</code> tested ${sg.self_test.tested} · fail ${sg.self_test.fail} · merged ${sg.self_test.merged} · unknown ${sg.self_test.unknown}</p>` : '';
    const unk = sg.unknown ? '<div class="scroll-x"><table class="sgrel"><tr><th>Object</th><th>Unknown span</th><th>Removed edge</th><th>Reason</th></tr>' +
      sg.unknown.map((u) => `<tr><td>${esc(u.obj)}</td><td class="bad">${span(u.t0, u.t1)}</td><td>${esc(u.rel)}${u.to ? ` ${esc(u.to)}` : ''}${u.event ? ` <code>${esc(u.event)}</code>` : ''} <code>${esc(u.test.name)}</code></td>` +
        `<td>${esc(u.reason)}</td></tr>`).join('') + '</table></div>' : '';
    return tree || rel || lau || unt || rw || ts || unk || cf ? `<h4>Scene graph${sg.kept ? ` · round ${sg.kept}` : ''}</h4>${tree}${st}${ts}${cf}${unk}${rel}${lau}${rw}${unt}` : '';
  }
  // the geometric test of an agent state edit (rule graph_edit.retest): result and its reason
  function testCells(t) {
    const cls = t.result === 'pass' ? 'ok' : t.result === 'fail' ? 'bad' : 'muted';
    return `<td class="${cls}">${esc(t.result)}</td><td>${esc(t.why)}</td>`;
  }
  // the agent rounds (answer record "iterations"): each tool call with its result, the round's outcome; the kept round
  // (the state the answer used) carries the accent bar
  // with the loop's stop record: its reason on the heading, each tried call's outcome class, each rejected round's
  // newly failed checks
  function roundsBlock(r) {
    if (!r) return '';
    const par = (p) => Object.entries(p).map(([k, v]) => `${k} ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ');
    const key = (it, c) => `${it} ${c.name} ${JSON.stringify(Object.keys(c.params).sort().map((k) => [k, c.params[k]]))}`;
    const cls = new Map(((r.stop && r.stop.tried) || []).map((t) => [key(t.iter, t), t.class]));
    const list = (a) => (a.length ? a.map((k) => `<code>${esc(k)}</code>`).join(' ') : 'none');
    const rows = r.rounds.map((x) => {
      const calls = x.calls.map((c) => {
        const k = cls.get(key(x.iter, c));
        return `<div><code>${esc(c.name)}</code> <span class="muted">${esc(par(c.params))}</span>${k ? ` <span class="tclass">${esc(k)}</span>` : ''}</div>`;
      }).join('');
      const res = x.calls.map((c) => `<div class="${/^rejected\b/.test(c.result) ? 'bad' : ''}">${esc(c.result)}</div>` +
        (c.test ? `<div class="${c.test.result === 'pass' ? 'ok' : c.test.result === 'fail' ? 'bad' : 'muted'}"><code>${esc(c.test.name)}</code> ${esc(c.test.result)}</div>` : '')).join('') +
        x.rejected.map((w) => `<div class="bad">${esc(w)}</div>`).join('');
      const cand = (x.candidates || []).map((c) => `<div class="cand"><code>${esc(c.for)}</code> ← <code>${esc(c.name)}</code> <span class="muted">${esc(par(c.params))}</span></div>`).join('') +
        (x.dropped || []).map((c) => `<div class="cand drop"><code>${esc(c.name)}</code> <span class="muted">${esc(par(c.params))}</span>` +
          `<div class="bad">${c.rule ? `<code>${esc(c.rule)}</code> ` : ''}${esc(c.why)}</div></div>`).join('');
      return `<tr class="${x.iter === r.kept ? 'chosen' : ''}"><td class="n">${x.iter}</td><td>${cand}</td><td>${calls}</td><td>${res}</td>` +
        `<td class="${x.accepted ? 'ok' : ''}">${esc(x.swap ? x.why.replace(/;?\s*checks fixed \[[^\]]*\] -> broken \[[^\]]*\]\s*$/, '') : x.why)}` +
        (x.swap ? `<div class="swap">fixed ${list(x.swap[0])} → broken ${list(x.swap[1])}</div>` : '') +
        (x.rebuilt ? `<div>rebuilt ${x.rebuilt.steps.map((k) => `<code>${esc(k)}</code>`).join(' → ') || 'none'}` +
          (x.rebuilt.failed ? ` · <span class="bad">failed <code>${esc(x.rebuilt.failed)}</code></span>` : '') +
          (x.rebuilt.self_test ? ` · self-test ${Object.entries(x.rebuilt.self_test).map(([k, v]) => `${esc(k)} ${esc(v)}`).join(', ')}` : '') + '</div>' : '') +
        (x.newly_failed || []).map(([c, w]) => `<div class="bad"><code>${esc(c)}</code> ${esc(w)}</div>`).join('') + '</td></tr>';
    }).join('');
    const st = r.stop ? ` <span class="stop"><code>${esc(r.stop.reason)}</code> round ${esc(r.stop.round)}${r.stop.why ? ` · ${esc(r.stop.why)}` : ''}</span>` : '';
    return `<h4>Agent rounds${st}</h4><div class="scroll-x"><table class="rounds"><tr><th class="n">Round</th><th>Candidates</th><th>Call</th><th>Result</th><th>Outcome</th></tr>` +
      rows + '</table></div>';
  }
  function sceneGraphSvg(sg) {
    const AIR = '\u0000air', NF = 12.5, LF = 11, LL = 13, PADX = 10, GAPX = 16, M = 6, MAXW = window.innerWidth < 640 ? 96 : 176;
    const type = {}, order = [];
    const addNode = (id, t) => { if (!(id in type)) { type[id] = t || 'object'; order.push(id); } };
    sg.nodes.forEach((n) => addNode(n.id, n.type));
    // relations per object: one entry per (target, relation), time spans merged
    const rels = new Map();
    sg.edges.forEach((e) => {
      if (!e.from || e.to === e.from) return;
      addNode(e.from); if (e.to) addNode(e.to);
      if (!rels.has(e.from)) rels.set(e.from, []);
      const to = e.to || null, l = rels.get(e.from);
      let r = l.find((x) => x.to === to && x.rel === e.rel);
      if (!r) { r = { to, rel: e.rel, spans: [], first: Infinity }; l.push(r); }
      if (e.t0 != null || e.t1 != null) r.spans.push([e.t0 == null ? 0 : e.t0, e.t1]);
      r.first = Math.min(r.first, e.t0 == null ? 0 : e.t0);
    });
    const relText = (r) => {
      if (!r.spans.length) return r.rel === 'on' || r.rel === 'airborne' ? '' : r.rel;
      return `${r.rel} ${r.spans.sort((a, b) => a[0] - b[0]).map(([a, b]) => (a === b ? fmt(a, 2) : `${fmt(a, 2)}–${b == null ? '' : fmt(b, 2)}`)).join(', ')} s`;
    };
    // first parent in time; an object with nothing to rest on but an airborne span goes under the airborne root
    const parent = {}, treeRel = {}, extra = [], stubs = {};
    order.forEach((id) => {
      const l = (rels.get(id) || []).slice().sort((a, b) => a.first - b.first);
      const real = l.filter((r) => r.to);
      let main = real[0] || null;
      if (!main && l.some((r) => r.rel === 'airborne')) main = { ...l.find((r) => r.rel === 'airborne'), to: AIR };
      if (main) { parent[id] = main.to; treeRel[id] = [main]; }
      l.forEach((r) => {
        if (main && r.to === main.to && r.rel === main.rel) return;
        if (main && main.to === AIR && r.rel === 'airborne') return;
        if (!r.to) (stubs[id] = stubs[id] || []).push(r);
        else if (r.to === main.to) treeRel[id].push(r);
        else extra.push({ from: id, to: r.to, r });
      });
    });
    // break parent cycles
    order.forEach((id) => {
      const seen = new Set([id]);
      for (let p = parent[id]; p && p !== AIR; p = parent[p]) { if (seen.has(p)) { delete parent[id]; break; } seen.add(p); }
    });
    const ids = order.slice();
    if (ids.some((id) => parent[id] === AIR)) { ids.push(AIR); type[AIR] = 'air'; }
    const kids = {};
    ids.forEach((id) => { if (parent[id]) (kids[parent[id]] = kids[parent[id]] || []).push(id); });
    const depth = {};
    const dep = (id) => (id in depth ? depth[id] : (depth[id] = parent[id] ? dep(parent[id]) + 1 : 0));
    ids.forEach(dep);
    const anc = (id, d) => { let a = id; while (a && depth[a] > d) a = parent[a]; return a; };
    // node boxes: the name, wrapped by width
    const box = {};
    ids.forEach((id) => {
      const name = id === AIR ? 'airborne' : id, words = String(name).split(/\s+/), lines = [];
      let cur = '';
      words.forEach((w) => { const t = cur ? cur + ' ' + w : w; if (cur && sgTextW(t, NF, true) > MAXW) { lines.push(cur); cur = w; } else cur = t; });
      if (cur) lines.push(cur);
      const w = Math.max(44, Math.max(...lines.map((l) => sgTextW(l, NF, true))) + 2 * PADX), h = lines.length * 16 + 12;
      const tl = (treeRel[id] || []).map(relText).filter(Boolean);
      const st = (stubs[id] || []).map((r) => relText(r) || r.rel);
      const tlw = tl.length ? Math.max(...tl.map((t) => sgTextW(t, LF))) + 10 : 0;
      const stw = st.length ? Math.max(...st.map((t) => sgTextW(t, LF))) + 26 : 0;
      box[id] = { lines, w, h, tl, st, L: Math.max(w / 2 + stw, tlw / 2), R: Math.max(w / 2, tlw / 2) };
    });
    // an extra parent two or more levels up gets a label slot beside the child of that parent the object descends from
    extra.forEach((x, k) => {
      if (!(x.to in depth) || depth[x.from] - depth[x.to] < 2) return;
      const a = anc(x.from, depth[x.to] + 1);
      if (parent[a] !== x.to) return;
      const sid = '\u0001' + k, t = relText(x.r), w = t ? sgTextW(t, LF) + 10 : 8;
      box[sid] = { slot: true, t, w, h: 16, L: w / 2, R: w / 2 };
      depth[sid] = depth[x.to] + 1; parent[sid] = x.to; x.slot = sid;
      const ch = kids[x.to], i = ch.indexOf(a);
      let j = i + 1; while (j < ch.length && ch[j].startsWith('\u0001')) j++;
      ch.splice(j, 0, sid);
    });
    // layout: each subtree is a horizontal strip, children side by side, the parent centred over its real children
    const off = {}, ext = {};
    const lay = (id) => {
      const b = box[id], ch = kids[id] || [];
      if (!ch.length) { ext[id] = [-b.L, b.R]; return; }
      let cur = 0; const cx = {};
      ch.forEach((c) => { lay(c); cx[c] = cur - ext[c][0]; cur = cx[c] + ext[c][1] + GAPX; });
      const real = ch.filter((c) => !box[c].slot), mid = (cx[real[0]] + cx[real[real.length - 1]]) / 2;
      let lo = -b.L, hi = b.R;
      ch.forEach((c) => { off[c] = cx[c] - mid; lo = Math.min(lo, off[c] + ext[c][0]); hi = Math.max(hi, off[c] + ext[c][1]); });
      ext[id] = [lo, hi];
    };
    const roots = ids.filter((id) => !parent[id] && id !== AIR).concat(ids.includes(AIR) ? [AIR] : []);
    const X = {};
    let cur = M;
    roots.forEach((r) => { lay(r); X[r] = cur - ext[r][0]; cur = X[r] + ext[r][1] + GAPX * 1.5; });
    const place = (id) => (kids[id] || []).forEach((c) => { X[c] = X[id] + off[c]; place(c); });
    roots.forEach(place);
    const W = Math.ceil(cur - GAPX * 1.5 + M);
    // rows bottom-up, as things rest: a parent sits below its children; the airborne subtree hangs from the top row instead
    const all = Object.keys(box), inAir = {};
    all.forEach((id) => { let a = id; while (parent[a]) a = parent[a]; inAir[id] = a === AIR; });
    const dmax = (f) => Math.max(0, ...all.filter(f).map((id) => depth[id]));
    const ltop = Math.max(dmax((id) => !inAir[id]), dmax((id) => inAir[id])), lv = {};
    all.forEach((id) => { lv[id] = inAir[id] ? ltop - depth[id] : depth[id]; });
    // rowH: tallest box; gap[r] (between row r and the row above) fits the edge labels running through it
    const nr = ltop + 1, rowH = Array(nr).fill(0), gap = Array(nr).fill(26);
    all.forEach((id) => {
      const b = box[id], r = lv[id];
      rowH[r] = Math.max(rowH[r], b.h, b.st ? b.st.length * LL + 4 : 0);
      if (!parent[id]) return;
      const g = inAir[id] ? r : r - 1;
      gap[g] = Math.max(gap[g], b.tl && b.tl.length ? 42 + b.tl.length * LL : b.slot && b.t ? 24 : 26);
    });
    const rowTop = [];
    rowTop[nr - 1] = M;
    for (let r = nr - 2; r >= 0; r--) rowTop[r] = rowTop[r + 1] + rowH[r + 1] + gap[r];
    const H = Math.ceil(rowTop[0] + rowH[0] + M);
    const top = (id) => rowTop[lv[id]] + (rowH[lv[id]] - box[id].h) / 2, bot = (id) => top(id) + box[id].h;
    const curve = (x1, y1, x2, y2) => { const m = (y1 + y2) / 2; return `<path class="sge" d="M${x1},${y1} C${x1},${m} ${x2},${m} ${x2},${y2}"></path>`; };
    // org-chart connector: from the parent to a bus line next to it, across, into the child; rounded corners
    const busY = (p) => (inAir[p] ? rowTop[lv[p]] + rowH[lv[p]] + 12 : rowTop[lv[p]] - 12);
    const near = (p) => (inAir[p] ? bot(p) : top(p)), far = (c) => (inAir[c] ? top(c) : bot(c));
    const elbow = (x1, y1, by, x2, y2) => {
      const dx = x2 - x1, r = Math.min(6, Math.abs(dx) / 2), s = Math.sign(dx), v = Math.sign(y2 - by);
      if (Math.abs(dx) < 1) return `<path class="sge" d="M${x1},${y1} V${y2}"></path>`;
      return `<path class="sge" d="M${x1},${y1} V${by} H${x2 - s * r} Q${x2},${by} ${x2},${by + v * r} V${y2}"></path>`;
    };
    const label = (cx, yBottom, lines) => {
      const w = Math.max(...lines.map((t) => sgTextW(t, LF))) + 8, h = lines.length * LL + 3, y0 = yBottom - h;
      return `<rect class="sgl" x="${cx - w / 2}" y="${y0}" width="${w}" height="${h}" rx="3"></rect>` +
        lines.map((t, i) => `<text class="sgt" x="${cx}" y="${y0 + 11 + i * LL}" text-anchor="middle">${esc(t)}</text>`).join('');
    };
    let edges = '', labels = '', boxes = '';
    ids.forEach((id) => {
      const p = parent[id];
      if (!p) return;
      const by = busY(p), ye = far(id);
      edges += elbow(X[p], near(p), by, X[id], ye);
      if (box[id].tl.length) { const n = box[id].tl.length; labels += label(X[id], (by + ye) / 2 + (n * LL + 3) / 2, box[id].tl); }
    });
    extra.forEach((x) => {
      const c = x.from, t = relText(x.r);
      if (!(x.to in box)) return;
      if (x.slot) {
        const s = x.slot, sx = X[s], sy = rowTop[lv[s]] + rowH[lv[s]] / 2, cr = X[c] + box[c].w / 2, cm = (top(c) + bot(c)) / 2;
        const blocked = ids.some((o) => o !== c && lv[o] === lv[c] && X[o] > X[c] && X[o] < sx);
        edges += elbow(X[x.to], near(x.to), busY(x.to), sx, sy);
        const v = Math.sign(cm - sy);
        edges += blocked ? curve(sx, sy, X[c] + box[c].w / 2 - 8, far(c))
          : `<path class="sge" d="M${sx},${sy} V${cm - v * 6} Q${sx},${cm} ${sx - 6},${cm} H${cr}"></path>`;
        if (t) labels += label(sx, sy + 8, [t]);
        return;
      }
      const below = lv[x.to] < lv[c], x1 = X[c], y1 = below ? bot(c) : top(c), x2 = X[x.to], y2 = below ? top(x.to) : bot(x.to);
      edges += curve(x1, y1, x2, y2);
      if (t) labels += label((x1 + x2) / 2, (y1 + y2) / 2 + 8, [t]);
    });
    ids.forEach((id) => {
      const b = box[id], x = X[id] - b.w / 2, y = top(id), cls = { support: 'sup', holder: 'hold', air: 'air' }[type[id]] || 'obj';
      boxes += `<g class="sgn ${cls}"><rect x="${x}" y="${y}" width="${b.w}" height="${b.h}" rx="7"></rect>` +
        b.lines.map((l, i) => `<text x="${X[id]}" y="${y + 19 + i * 16}" text-anchor="middle">${esc(l)}</text>`).join('') + '</g>';
      const my = y + b.h / 2;
      b.st.forEach((t, i) => {
        const sy = my + (i - (b.st.length - 1) / 2) * LL;
        boxes += `<path class="sge stub" d="M${x},${sy} H${x - 14}"></path><circle class="sgs" cx="${x - 17}" cy="${sy}" r="3"></circle>` +
          `<text class="sgt" x="${x - 24}" y="${sy + 4}" text-anchor="end">${esc(t)}</text>`;
      });
    });
    return `<div class="sgwrap"><svg class="sg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="scene graph">${edges}${labels}${boxes}</svg></div>`;
  }

  // ---------- reconstruction: the question's own scene value, the fitted objects ----------
  // an object's dimensions in columns by source (scene_fit dim_source): observed, prior, class, bound; a bound
  // dimension is the interval its observations allow (dim_bounds), never a value; a class one is the class's
  // proportion of the height (rule dims.agent_lateral), not observed
  const DIM_SOURCES = ['observed', 'prior', 'class', 'bound'];
  const interval = (lo, hi) => `[${fmtNum(lo)}, ${hi == null ? '∞)' : fmtNum(hi) + ']'}`;
  function dimsBySource(o) {
    const keys = { sphere: ['D'], cylinder: ['D', 'H'], rope: ['L'] }[o.shape] || ['L', 'W', 'H'];
    const out = { observed: [], prior: [], class: [], bound: [] };
    keys.forEach((k) => {
      const src = o.dim_source[k], b = o.dim_bounds && o.dim_bounds[k];
      if (!(src in out)) throw new Error(`${o.name}: dim_source ${k} = ${src}`);
      out[src].push(`<span class="dim">${k} ${src === 'bound' && b ? interval(b[0], b[1]) : fmtNum(o.dims[k])}</span>`);
    });
    return out;
  }
  function supportText(o) {
    const sp = o.support;
    let t = sp.mode === 'support' ? `on ${sp.target || 'main support'}` : sp.mode === 'airborne' ? (sp.airborne_part && sp.target ? `on ${sp.target}, then airborne` : 'airborne') : sp.mode || '';
    if (o.hang) t += ' (top face is the support)';
    if (o.t_on > 0) t += `; at rest until ${fmt(o.t_on, 2)} s`;
    const MODE = { ballistic: 'bouncing', contact: 'contact (geometry first)', overhead: 'overhead structure', free3d: 'free 3D motion', own_plane: 'own plane' };
    if (o.mode && MODE[o.mode]) t += `; ${MODE[o.mode]}`;
    if (o.events && o.events.length) t += ` (${o.events.slice(0, 4).map((e) => `${e[1]} ${fmt(e[0], 2)} s`).join(', ')}${o.events.length > 4 ? '…' : ''})`;
    if (o.out_of_frame && o.out_of_frame.length) t += `; out of frame ${o.out_of_frame.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ')} s`;
    return t;
  }
  // focal-length votes of the fit (scene json "focal_votes")
  const F_SRC = { gravity: 'gravity parabola', lab_prior: 'lab prior', size_prior_depth: 'size prior + depth', typical_votes: 'typical size', typical_height: 'typical height', prior_kinematic: 'kinematic prior', person: 'person height', votes_first: 'votes first' };
  function focalVotes(s) {
    if (!s.focal_votes || !s.focal_votes.length) return '';
    const tw = s.focal_votes.reduce((a, x) => a + x.weight, 0);
    return '<details class="sfres"><summary>Focal votes</summary><div class="scroll-x"><table><tr><th>Source</th><th class="n">f (px)</th><th class="n">Weight</th></tr>' +
      s.focal_votes.map((x) => `<tr><td>${esc(F_SRC[x.source] || x.source)}</td><td class="n">${fmtNum(x.value)}</td><td class="n">${fmtNum(x.weight)}${tw ? ` <span class="muted">(${fmt((100 * x.weight) / tw, 0)}%)</span>` : ''}</td></tr>`).join('') +
      '</table></div></details>';
  }
  // per-hop gravity focal votes (scene json "focal", rule focal.gravity_hop); a hop without a vote is muted
  function focalTable(fo) {
    if (!fo || !fo.votes.length) return '';
    const iv = (a) => `${fmt(a[0], 2)}–${fmt(a[1], 2)}`;
    return '<div class="scroll-x"><table class="focal"><tr><th>Object</th><th>Airborne</th><th>Hop (s)</th><th class="n">f (px)</th>' +
      '<th class="n">n</th><th class="n">rms (px)</th><th>Law</th><th></th></tr>' +
      fo.votes.map((x) => `<tr class="${x.used ? '' : 'muted'}"><td>${esc(x.object)}</td><td>${iv(x.span)}</td><td>${iv(x.hop)}</td>` +
        `<td class="n">${x.f != null ? `${fmtNum(x.f)} ± ${fmtNum(x.sigma)}` : '—'}</td><td class="n">${x.n != null ? esc(x.n) : ''}</td>` +
        `<td class="n">${x.rms != null ? fmt(x.rms, 1) : ''}</td><td>${esc(x.law)}</td><td>${x.used ? 'used' : esc(x.why)}</td></tr>`).join('') +
      '</table></div>';
  }
  // camera.f before the fusion replaced it (rule C1a camera.fused)
  function wasText(fz) {
    if (!fz) return '';
    return `, was ${fz.was.f != null ? fmtNum(fz.was.f) + ' ' : ''}${esc(fz.was.source)}`;
  }
  // objects whose agency is unknown cast no gravity focal vote (rule agency.unknown)
  function agencyUnknown(cam) {
    if (!cam || !cam.agency_unknown) return '';
    return cam.agency_unknown.map((x) => `<p class="facts muted"><code>agency.unknown</code> ${esc(x.object)} · ${esc(x.why)}</p>`).join('');
  }
  const res3 = (p) => (p === true ? 'pass' : p === false ? 'fail' : 'untestable');
  const cls3 = (p) => (p === true ? 'ok' : p === false ? 'bad' : 'muted');
  // the focal length each prior or pendulum implies against the scene camera (rule C3a focal.consistent)
  function consistentTable(cam) {
    if (!cam || !cam.consistent.length) return '';
    const ind = cam.consistent.some((c) => c.independent != null);
    const indCell = (c) => (!ind ? '' : `<td class="${c.independent ? '' : 'bad'}">${c.independent ? 'yes' : 'no'}</td>`);
    const why = (c) => c.reason.replace(/^untestable:\s*/, '').replace(/^.*?: not independent:\s*/, '');
    return '<div class="scroll-x"><table class="fcons"><tr><th><code>focal.consistent</code></th>' +
      (ind ? '<th><code>focal.independent</code></th>' : '') + '<th class="n">Implied f (px)</th>' +
      '<th class="n">Scene f (px)</th><th class="n">|Δ| vs 3σ</th><th></th><th></th></tr>' +
      cam.consistent.map((c) => `<tr class="${cls3(c.pass)}"><td>${esc(c.source)}</td>${indCell(c)}` +
        `<td class="n">${c.f_obj != null ? plusMinus(c.f_obj, c.sigma_obj) : '—'}` +
        (c.extent ? `<div class="muted">${plusMinus(c.extent.value, c.extent.sigma)} px · f${frameRuns(c.extent.used)}</div>` : '') + '</td>' +
        `<td class="n">${c.f != null ? plusMinus(c.f, c.sigma) : '—'}</td>` +
        `<td class="n">${c.diff != null && c.bound != null ? `${fmtNum(c.diff)} ${c.pass ? '≤' : '>'} ${fmtNum(c.bound)}` : '—'}</td>` +
        `<td>${res3(c.pass)}</td><td>${c.pass === null ? esc(why(c)) : esc(c.how)}` +
        (c.class_size ? `<div class="${c.class_size.pass ? 'ok' : 'bad'}"><code>focal.class_size</code> ${esc(c.class_size.reason)}</div>` : '') +
        '</td></tr>').join('') + '</table></div>';
  }
  // the scene camera fused from stage A and every pendulum's joint f (rule camera.one); an unused vote is muted
  function cameraOneTable(co) {
    if (!co) return '';
    return '<div class="scroll-x"><table class="focal camone"><tr><th><code>camera.one</code></th><th>Object</th><th class="n">f (px)</th><th></th></tr>' +
      co.votes.map((x) => `<tr class="${x.used ? '' : 'muted'}"><td>${esc(x.source)}</td><td>${esc(x.object)}</td>` +
        `<td class="n">${plusMinus(x.f, x.sigma)}</td><td>${x.used ? 'used' : esc(x.why)}</td></tr>`).join('') +
      `<tr class="camf"><td colspan="2" title="${esc(co.how)}">camera.f</td><td class="n">${plusMinus(co.f, co.sigma)}</td><td></td></tr></table></div>`;
  }
  // the scene's focal votes fused by independent source (rule focal.fusion); an outlier or disagreeing source is
  // marked, the last row is the fused f (camera.f is on the line above), then every measurement the fusion failed (check focal)
  function fusionTable(fu, q) {
    if (!fu) return '';
    const vcls = (v) => (v === 'used' ? '' : v === 'excluded' || v === 'no sigma' ? 'muted' : 'bad');
    const why = fu.reason.replace(/^untestable:\s*/, '');
    const failed = ((q && q.measurements) || []).flatMap((m) => m.failed.filter((c) => c.name === 'focal').map((c) => [m.name, c]));
    return '<div class="scroll-x"><table class="focal fusion"><tr><th><code>focal.fusion</code></th><th>Object</th><th class="n">f (px)</th>' +
      '<th class="n">Votes</th><th>Camera</th><th></th><th></th></tr>' +
      fu.sources.map((x) => `<tr class="${vcls(x.verdict)}"><td>${esc(x.kind)}</td><td>${esc(x.object)}</td>` +
        `<td class="n" title="${esc(x.how)}">${plusMinus(x.f, x.sigma)}${x.bounded ? `<div class="muted">${fmtNum(x.bounded[0])}–${fmtNum(x.bounded[1])}</div>` : ''}</td>` +
        `<td class="n">${x.n}</td><td>${x.camera ? 'yes' : 'no'}</td><td>${esc(x.verdict)}</td><td>${x.why ? esc(x.why) : ''}</td></tr>`).join('') +
      `<tr class="camf ${cls3(fu.pass)}"><td colspan="2">fused · ${fu.n} source${fu.n === 1 ? '' : 's'}</td><td class="n">${fu.f != null ? plusMinus(fu.f, fu.sigma) : '—'}</td>` +
      `<td></td><td></td><td>${res3(fu.pass)}</td><td>${fu.pass === true ? '' : esc(why)}</td></tr>` +
      failed.map(([n, c]) => `<tr class="bad"><td colspan="2"><code>${esc(n)}</code> <code>focal</code></td><td colspan="5">${esc(c.reason)}</td></tr>`).join('') +
      '</table></div>';
  }
  // the support plane's acceptance with its sub-tests (rule D9 support.plane); a recorded sub-test has no verdict
  function supportPlaneTable(sp) {
    if (!sp) return '';
    if (!sp.tests.length) return `<p class="facts"><code>support.plane</code> <span class="muted">${esc(sp.reason)}</span></p>`;
    const v = (t) => (t.pass === null && !t.level && !/^untestable/.test(t.reason) ? 'recorded' : res3(t.pass));
    return '<div class="scroll-x"><table class="checks splane">' +
      `<tr class="${cls3(sp.pass)}"><th><code>support.plane</code></th><th>${res3(sp.pass)}</th><th></th></tr>` +
      sp.tests.map((t) => `<tr class="${cls3(t.pass)}"><td>${esc(t.name)}</td><td>${v(t)}</td><td>${esc(t.reason.replace(/^untestable:\s*/, ''))}${levelLine(t.level)}</td></tr>`).join('') +
      '</table></div>';
  }
  const deg = (x) => (x === 'inf' ? '∞' : `${fmtNum(x)}°`);
  // support.plane level (rule support.main): the angle against 3σ, σ in quadrature of the depth plane's (per-frame
  // spread, alignment) and the scene plane's own
  function levelLine(l) {
    if (!l || l.angle == null) return '';
    const p = l.parts;
    return `<div class="sub">${deg(l.angle)} vs 3 × ${deg(l.sigma)}` +
      (p ? ` = depth plane ${deg(p.depth_plane)} (frames ${deg(p.depth_plane_frames)}, alignment ${deg(p.depth_alignment)}) ⊕ scene plane ${deg(p.scene_plane)}` : '') + '</div>';
  }
  // the plane through the main support's own evidence (rule support.main) and its refit when stage A's plane was not
  // level (rule support.level_refit)
  function mainPlaneLines(mp, main) {
    if (!mp) return '';
    const ex = mp.excluded ? Object.entries(mp.excluded).map(([o, sups]) => `${esc(o)} on ${sups.map(esc).join(', ')}`).join('; ') : '';
    let h = `<p class="facts"><code>support.main</code> ${esc(main || '')} · <span title="${esc(mp.how)}">${mp.n_mask} mask points · ${mp.n_contacts} contacts</span>` +
      ` · roll ${deg(mp.roll)}${mp.roll_source ? ` <span class="muted">(${esc(mp.roll_source)})</span>` : ''} · pitch ${deg(mp.pitch)} · height ${fmtNum(mp.height)} m` +
      ` · residual ${fmtNum(mp.residual)} m · spread ${deg(mp.spread)}${ex ? ` · fitted free: ${ex}` : ''}</p>`;
    const r = mp.refit;
    if (r) {
      h += `<p class="facts"><code>support.level_refit</code> <span class="bad" title="${esc(r.why)}">${deg(r.level.angle)} &gt; 3 × ${deg(r.level.sigma)}</span>` +
        ` · roll ${deg(r.roll_deg[0])} → ${deg(r.roll_deg[1])} · pitch ${deg(r.pitch_deg[0])} → ${deg(r.pitch_deg[1])}` +
        ` · height ${fmtNum(r.height[0])} → ${fmtNum(r.height[1])} m · rms ${fmt(r.rms[0], 1)} → ${fmt(r.rms[1], 1)} px</p>`;
    }
    return h;
  }
  // the scale object's contacts as depth equations (rule scale.contact), re-tested at the fused plane when the plane's
  // distance was fused with them (rule plane.fusion)
  function scaleContactBlock(sc) {
    if (!sc) return '';
    const hasReach = sc.events.some((e) => e.reach);
    const reachCell = (e) => (!hasReach ? '' : e.reach
      ? `<td class="n" title="${esc(`${e.reach.point} · H ${fmtNum(e.reach.H)} m · ${e.reach.H_how}`)}">${fmtNum(e.reach.sigma)}</td>` : '<td></td>');
    if (!sc.events.length) return `<p class="facts"><code>scale.contact</code> ${esc(sc.object)} <span class="muted">${esc(sc.reason.replace(/^untestable:\s*/, 'untestable · '))}</span></p>`;
    const table = (evs, p) => '<div class="scroll-x"><table class="sgrel contact"><tr>' +
      `<th><code>scale.contact</code> ${esc(sc.object)}</th><th class="n">t (s)</th><th>Other</th><th class="n">Z scale (m)</th><th class="n">Z other (m)</th>${hasReach ? '<th class="n">Reach σ (m)</th>' : ''}<th class="n">z</th><th>${res3(p)}</th></tr>` +
      evs.map((e) => `<tr class="${cls3(e.pass)}"><td title="${esc(e.source)}">${esc(e.kind)}</td><td class="n">${fmt(e.t, 3)}</td><td>${esc(e.other)}</td>` +
        `<td class="n">${plusMinus(e.z_scale, e.s_scale)}</td><td class="n" title="${esc((e.n != null ? `${e.n} frames, nearest ${fmt(e.gap, 3)} s` : '') + (e.s_plane != null ? ` · plane σ ${fmtNum(e.s_plane)} m` : '') + (e.s_line != null ? ` · line σ ${fmtNum(e.s_line)} m · orientation σ ${fmtNum(e.s_angles)} m · extrapolation ${fmt(100 * e.extrap, 1)} %` : ''))}">${plusMinus(e.z_other, e.s_other)}</td>` +
        `${reachCell(e)}<td class="n">${fmt(e.z, 1)}</td><td title="${esc(e.why)}">${res3(e.pass)}</td></tr>`).join('') + '</table></div>';
    return table(sc.events, sc.pass) + (sc.after ? table(sc.after.events, sc.after.pass).replace('<code>scale.contact</code>', '<code>scale.contact</code> at the fused plane') : '');
  }
  // the camera height fused by inverse variance from the depth model's plane and every scale.contact depth equation
  // (rule plane.fusion): one row per source, then the fused height and its largest pull against 3
  function planeFusionTable(pf) {
    if (!pf) return '';
    const rows = pf.sources.map((x) => `<tr><td title="${esc(x.how)}"><code>${esc(x.source)}</code>${x.other ? ` ${esc(x.other)}` : ''}</td>` +
      `<td class="n">${x.t ? x.t.map((t) => fmt(t, 3)).join(', ') : ''}</td><td class="n">${plusMinus(x.h, x.sigma)}</td><td></td></tr>`).join('');
    const pull = pf.pass == null ? '' : `${fmt(pf.z, 2)} ${pf.pass ? '≤' : '>'} 3`;
    return '<div class="scroll-x"><table class="sgrel contact pfuse"><tr><th><code>plane.fusion</code></th><th class="n">t (s)</th><th class="n">h (m)</th><th class="n">Pull</th></tr>' +
      rows + `<tr class="fused ${cls3(pf.pass)}"><td title="${esc(pf.reason)}">fused${pf.k !== 1 ? ` · k ${fmtNum(pf.k)}` : ''}</td><td></td>` +
      `<td class="n">${plusMinus(pf.h, pf.sigma)}</td><td class="n" title="${esc(pf.reason)}">${pull || res3(pf.pass)}</td></tr></table></div>` +
      (pf.rescaled ? `<p class="facts"><code>${esc(pf.rescaled.rule)}</code> k ${fmtNum(pf.rescaled.k)} · ${pf.rescaled.n_dense} dense rows · ${pf.rescaled.n_plane_anchors} plane anchors · ${pf.rescaled.n_depth_contacts} ground contacts · ${pf.rescaled.n_extents} extents</p>` : '');
  }
  // the plane's distance sigma at each object's foot row: the row's depth beyond the support mask's depths and the
  // pitch lever there (rule plane.fusion), the odd-plane share (rule depth_rows.odd)
  function planeSigmaTable(ps) {
    if (!ps || !ps.extrapolation.length) return '';
    const odd = new Map(ps.odd);
    return '<div class="scroll-x"><table class="sgrel contact"><tr><th>Extrapolation</th><th class="n">Row (px)</th><th class="n">Row depth (m)</th><th class="n">Mask depth (m)</th>' +
      '<th class="n">Lever (m)</th><th class="n">Pitch σ</th><th class="n">σ rel</th><th class="n">Plane σ rel</th></tr>' +
      ps.extrapolation.map((x) => `<tr><td title="${esc(x.how)}">${esc(x.obj)}</td><td class="n">${fmtNum(x.row)}</td>` +
        `<td class="n${x.beyond ? ' bad' : ''}">${fmtNum(x.z_row)}</td><td class="n">${fmtNum(x.mask_depth[0])}–${fmtNum(x.mask_depth[1])}</td>` +
        `<td class="n">${fmtNum(x.lever)}</td><td class="n">${fmt(x.phi, 2)}°</td><td class="n">${fmt(100 * x.rel, 1)} %</td>` +
        `<td class="n">${fmt(100 * x.plane_rel, 1)} %${odd.has(x.obj) ? ` <span class="muted">odd ${fmt(100 * odd.get(x.obj), 1)} %</span>` : ''}</td></tr>`).join('') + '</table></div>';
  }
  // an articulated agent's dims: lateral ones from its class, never its box (rule dims.agent_lateral); its height
  // observed only when the ray meets the support plane within the threshold angle (rule dims.agent_height)
  function agentDimsTable(s) {
    const os = s.objects.filter((o) => o.agent_lateral);
    if (!os.length) return '';
    return '<div class="scroll-x"><table class="objs fullext"><tr><th><code>dims.agent_lateral</code></th><th>Lateral</th><th class="n">Box width</th><th></th>' +
      '<th><code>dims.agent_height</code></th><th class="n">Pitch</th><th>H</th><th></th></tr>' +
      os.map((o) => {
        const a = o.agent_lateral, b = o.agent_height;
        return `<tr><td>${esc(o.name)}</td><td>${Object.entries(a.sources).map(([k, v]) => `<span class="dim">${k} ${esc(v)} ${fmtNum(o.dims[k])}</span>`).join('')}</td>` +
          `<td class="n">${fmtNum(a.box_px)} px = ${fmtNum(a.box_m)} m${a.fits_box ? '' : ' <span class="bad">narrower than the body</span>'}</td>` +
          `<td class="${cls3(a.pass)}" title="${esc(a.reason)}">${res3(a.pass)}</td>` +
          `<td class="n">ray ${fmt(b.ray, 1)}° ${b.pass === false ? '>' : '≤'} ${fmt(b.threshold, 1)}°</td><td class="n">${fmt(b.pitch, 1)}°</td>` +
          `<td>${esc(b.H_source)} ${fmtNum(o.dims.H)}</td><td class="${cls3(b.pass)}" title="${esc(b.reason)}">${res3(b.pass)}</td></tr>`;
      }).join('') + '</table></div>';
  }
  // each physics chain's phases over the clip; a driven span (rule law.impulse_span) has no law, a straight segment
  function chainTable(s, T) {
    const os = s.objects.filter((o) => o.chain && o.chain.length);
    if (!os.length) return '';
    const W = 600, H = 34, B = 14;
    const bar = (ph) => {
      const end = Math.max(T || 0, ph[ph.length - 1][0] + 1e-3), X = (t) => W * Math.min(Math.max(t, 0), end) / end;
      return `<svg class="chainbar" viewBox="0 0 ${W} ${H}" role="img" aria-label="physics chain">` + ph.map(([t, k, kick], i) => {
        const x0 = X(t), x1 = i + 1 < ph.length ? X(ph[i + 1][0]) : W, w = Math.max(x1 - x0, 1.5);
        return `<rect class="k-${k}" x="${x0}" y="0" width="${w}" height="${H - B}"><title>${k}${kick ? ' (kick)' : ''} ${fmt(t, 3)} s</title></rect>` +
          (w > 8 * k.length ? `<text x="${x0 + w / 2}" y="${(H - B) / 2 + 4}" text-anchor="middle">${k}</text>` : '') +
          (i ? `<text class="t" x="${x0}" y="${H - 2}" text-anchor="${k === 'driven' ? 'end' : 'start'}">${fmt(t, 3)}</text>` : '');
      }).join('') + '</svg>';
    };
    return '<div class="scroll-x"><table class="objs chain"><tr><th>Physics chain</th><th>Phases</th></tr>' +
      os.map((o) => `<tr><td>${esc(o.name)}</td><td>${bar(o.chain)}</td></tr>`).join('') +
      '</table></div>';
  }
  // each object's silhouette depth against its support-plane depth before the fit (rule scale.contact_rescale): a
  // failing object's dense depth rows leave the fit
  function depthRowsTable(rows, qid) {
    if (!rows) return '';
    return '<div class="scroll-x"><table class="sgrel contact"><tr><th><code>direct.depth_agree</code></th><th class="n">Silhouette (m)</th><th class="n">Plane (m)</th>' +
      '<th class="n">|Δ| (m)</th><th class="n">Rows</th><th></th></tr>' +
      rows.map((d, i) => `<tr class="${cls3(d.pass)}" id="${esc(depthRowId(qid, i))}"><td>${esc(d.obj)}</td><td class="n">${fmtNum(d.z_mask)}</td><td class="n">${fmtNum(d.z_plane)}</td>` +
        `<td class="n">${d.pass == null ? '' : `${fmtNum(Math.abs(d.z_mask - d.z_plane))} ${d.pass ? '≤' : '>'} ${fmtNum(d.bound)}`}</td>` +
        `<td class="n">${d.dropped ? `${d.n_rows} dropped` : d.n_rows}</td><td title="${esc(d.reason)}">${res3(d.pass)}${d.odd ? ` · <code>depth_rows.odd</code> ${esc(d.odd)}` : ''}</td></tr>` +
        (d.ground ? `<tr class="${cls3(d.ground.pass)}"><td>↳ ground beside the feet</td><td class="n">${fmtNum(d.ground.z)}</td><td></td>` +
          `<td class="n">${d.ground.pass == null ? '' : `${fmtNum(Math.abs(d.z_mask - d.ground.z))} ${d.ground.pass ? '≤' : '>'} ${fmtNum(d.ground.bound)}`}</td><td class="n">${d.ground.n}</td>` +
          `<td title="${esc(d.ground.reason)}">${res3(d.ground.pass)}${d.sigma_added != null ? ` · plane σ +${fmt(100 * d.sigma_added, 1)} %` : ''}</td></tr>` : '')).join('') + '</table></div>';
  }
  const depthRowId = (qid, i) => `depth-rows-${qid}-${i}`;
  // each articulated agent's upright, unmerged extent (rule size.full_extent): frames used, frames excluded by kind,
  // the median box height over the used frames against all clear frames, h z against its reference
  function fullExtentTable(s) {
    const os = s.objects.filter((o) => o['size.full_extent']);
    if (!os.length) return '';
    const kinds = ['overlap', 'shrink', 'border', 'no_depth'].concat(os.some((o) => o['size.full_extent'].excluded.height) ? ['height'] : []);
    return '<div class="scroll-x"><table class="objs fullext"><tr><th><code>size.full_extent</code></th><th class="n">Used</th>' +
      `<th>Overlap</th><th>Shrink</th><th>Border</th><th>No depth</th>${kinds.length > 4 ? '<th>Height</th>' : ''}<th class="n">Extent (px)</th><th class="n">h z (px m)</th><th>Depth</th><th></th></tr>` +
      os.map((o) => fullExtentRow(o.name, o['size.full_extent'], kinds)).join('') + '</table></div>';
  }
  function fullExtentRow(name, e, kinds) {
    const x = e.excluded;
    return `<tr class="${cls3(e.pass)}"><td>${esc(name)}</td><td class="n">${e.used.length}</td>` +
      kinds.map((k) => `<td>${x[k] && x[k].length ? frameRuns(x[k]) : ''}</td>`).join('') +
      `<td class="n">${fmtNum(e.extent_px)} <span class="muted">/ ${fmtNum(e.extent_px_all)}</span></td>` +
      `<td class="n">${fmtNum(e.extent_norm)} <span class="muted">/ ${fmtNum(e.ref_norm)}</span></td><td>${esc(e.depth)}</td>` +
      `<td class="${cls3(e.pass)}" title="${esc(e.reason)}">${res3(e.pass)}</td></tr>`;
  }
  // each object's fit as a whole (rule fit.object): s² against its bound, χ² / n per row kind; a failed fit is marked
  function fitObjectTable(s) {
    const os = s.objects.filter((o) => o.fit_object);
    if (!os.length) return '';
    return '<div class="scroll-x"><table class="objs fitobj"><tr><th><code>fit.object</code></th><th class="n">s²</th><th class="n">Bound</th><th class="n">n − p</th>' +
      '<th class="n">Fits</th><th>χ² / n</th><th></th></tr>' +
      os.map((o) => {
        const f = o.fit_object;
        return `<tr class="${cls3(f.pass)}"><td>${esc(o.name)}</td><td class="n">${fmtNum(f.s2)}</td><td class="n">${f.bound != null ? `${f.pass ? '≤' : '>'} ${fmtNum(f.bound)}` : ''}</td>` +
          `<td class="n">${f.dof}</td><td class="n">${f.n_fits}</td><td>${f.rows.map(([k, c, n]) => `<span class="dim">${esc(k)} ${fmtNum(c)} / ${n}</span>`).join('')}</td>` +
          `<td class="${cls3(f.pass)}" title="${esc(f.reason)}">${f.pass === false ? 'failed' : res3(f.pass)}</td></tr>`;
      }).join('') + '</table></div>' + fitObjectFacts(os);
  }
  // what fit.object records beside its verdict (rule fit.object_noise), none of it a gate: each stated row's tension,
  // the box sides' noise and bias (rms before → after the bias), the box-noise s² against its bound, the frames where
  // the chain flies while the graph puts the object on a support
  function fitObjectFacts(os) {
    const fs = os.filter((o) => 'box_test' in o.fit_object);
    if (!fs.length) return '';
    const px = (a) => a.map((x) => fmtNum(x)).join(' / ');
    return '<div class="scroll-x"><table class="objs fitfacts"><tr><th><code>fit.object_noise</code></th><th>Stated z</th><th>Sides</th><th class="n">Box noise (px)</th>' +
      '<th class="n">Box bias (px)</th><th class="n">Box rms (px)</th><th class="n">Box-noise s²</th><th><code>state_mismatch</code></th></tr>' +
      fs.map((o) => {
        const f = o.fit_object, bt = f.box_test, sm = f.mismatch;
        return `<tr><td>${esc(o.name)}</td>` +
          `<td>${(f.stated || []).map((x) => `<span class="dim" title="${esc(`residual ${x.residual} · σ stated ${x.s_stated} · σ pred ${x.s_pred}`)}">${esc(x.kind)} ${fmtNum(x.z)}</span>`).join('')}</td>` +
          `<td>${f.sides ? esc(f.sides.join(' / ')) : ''}</td><td class="n">${f.box_noise ? px(f.box_noise) : ''}</td><td class="n">${f.box_bias ? px(f.box_bias) : ''}</td>` +
          `<td class="n">${f.box_rms ? f.box_rms.map(([a, b]) => `${fmtNum(a)} → ${fmtNum(b)}`).join(' / ') : ''}</td>` +
          `<td class="n${bt ? ' ' + cls3(bt.pass) : ''}" title="${esc(bt ? bt.why : '')}">${!bt ? '' : bt.pass === null ? `untestable · dof ${bt.dof}` : `${fmtNum(bt.s2)} ${bt.pass ? '≤' : '>'} ${fmtNum(bt.bound)}`}</td>` +
          `<td title="${esc(sm ? sm.reason : '')}">${sm.pass === null ? '<span class="muted">untestable</span>' : sm.pass ? '0 frames' : `${sm.frames} frames ${span(sm.t[0], sm.t[1])}`}</td></tr>`;
      }).join('') + '</table></div>';
  }
  // the support plane's pitch source and its horizon check (rule support.horizon)
  function horizonLines(h) {
    if (!h) return '';
    const deg = (x) => `${fmt(x, 2)}°`;
    let out = '';
    if (h.plane) {
      out += `<p class="facts">support plane ${deg(h.plane.tilt)} from up` + (h.plane.pitch_source ? ` · pitch from ${esc(h.plane.pitch_source)}` : '') +
        (h.plane.depth_tilt != null ? ` · depth model ${deg(h.plane.depth_tilt)}` : '') + '</p>';
    }
    const c = h.check;
    if (!c) return out;
    const res = c.pass === true ? 'pass' : c.pass === false ? 'fail' : 'untestable';
    if (c.phi_plane == null) {
      return out + `<p class="facts"><code>support.horizon</code> <span class="muted">${esc(c.reason)}</span></p>`;
    }
    return out + `<p class="facts"><code>support.horizon</code> <span class="${c.pass ? 'ok' : c.pass === false ? 'bad' : 'muted'}">${res}</span>` +
      ` · plane ${deg(c.phi_plane)} vs horizon ${deg(c.phi_horizon)} ± ${fmt(c.sigma, 2)}°, z ${fmt(c.z, 1)}` +
      ` · row ${fmtNum(c.row)} ± ${fmt(c.sigma_row, 1)} px (plane ${fmtNum(c.row_plane)}), spread ${fmtNum(c.spread)} px, ${esc(c.n_found)}/${esc(c.n_frames)} frames` +
      (c.far ? ` · far: band ${fmtNum(c.far.band)} m vs support p95 ${fmtNum(c.far.support_p95)} m` : '') +
      (c.repaired ? ' · refit' : '') + '</p>';
  }
  // the scene's roll against the roll of the image's near-vertical segments (rule camera.roll), the pose refit on fail
  function rollLines(r) {
    if (!r) return '';
    const deg = (x) => `${fmt(x, 2)}°`;
    if (r.pass === null) {
      const why = r.reason.replace(/^camera\.roll untestable:\s*/, '');
      return `<p class="facts"><code>camera.roll</code> <span class="muted">untestable · ${esc(why)}</span></p>`;
    }
    const im = r.image || {};
    const rf = r.refit;
    return `<p class="facts"><code>camera.roll</code> <span class="${r.pass ? 'ok' : 'bad'}">${r.pass ? 'pass' : 'fail'}</span>` +
      ` · <span title="${esc(r.how || '')}">fit ${deg(r.fit)}</span> vs image ${deg(r.image_roll)} ± ${fmt(r.sigma, 2)}°` +
      ` (${esc(im.n_lines)} segments, ${esc(im.n_frames)} frames), z ${fmt(r.z, 1)} · roll from ${esc(r.source)}` +
      (rf ? ` · refit roll ${deg(rf.roll_deg[0])} → ${deg(rf.roll_deg[1])}, pitch ${deg(rf.pitch_deg[0])} → ${deg(rf.pitch_deg[1])},` +
        ` dist ${fmtNum(rf.dist[0])} → ${fmtNum(rf.dist[1])}${rf.rms ? `, rms ${fmt(rf.rms[0], 1)} → ${fmt(rf.rms[1], 1)} px` : ''}` : '') + '</p>';
  }
  function fitBlock(q, v) {
    const f = q.fit, s = v.scene;
    let h = '<h4>Reconstruction</h4>';
    if (q.binding_error) h += `<p class="fwarn">${esc(q.binding_error)}</p>`;
    if (q.error) return h + `<p class="fwarn">${esc(q.error)}</p>`;
    if (q.measurements) {
      h += measurementBlock(q);
    } else if ((f || q.method || q.status) && q.checks) {
      // decided by named checks: an unresolved question shows the scene value its record kept, never as the answer
      const unres = q.status === 'unresolved';
      const val = unres ? (q.scene_value != null ? `<span class="notans">${fmtNum(q.scene_value)}</span>` : '—') : fmtNum(f && f.pred);
      h += '<div class="scroll-x"><table class="fit"><tr><th>Object</th><th>Method</th><th class="n">Scene value</th>' +
        '<th class="n">t_q alignment</th><th>Status</th></tr>' +
        `<tr><td>${f ? esc(f.obj) : '—'}</td><td><code>${esc(q.method)}</code></td><td class="n">${val}</td>` +
        `${alignCell(q.align)}<td><code>${esc(q.status)}</code></td></tr></table></div>` +
        '<div class="scroll-x"><table class="checks"><tr><th>Check</th><th></th><th>Reason</th></tr>' +
        q.checks.map((c) => `<tr class="${c.pass ? 'ok' : 'bad'}"><td><code>${esc(c.name)}</code></td>` +
          `<td>${c.pass ? 'pass' : 'fail'}</td><td>${esc(c.reason)}</td></tr>`).join('') + '</table></div>';
    } else if (f || q.method || q.status) {
      h += '<div class="scroll-x"><table class="fit"><tr><th>Object</th><th>Method</th><th class="n">Scene value</th>' +
        '<th class="n">t_q alignment</th><th>Status</th></tr>' +
        `<tr><td>${f ? esc(f.obj) : '—'}</td><td><code>${esc(q.method)}</code></td>` +
        `<td class="n">${fmtNum(f && f.pred)}</td>${alignCell(q.align)}<td><code>${esc(q.status)}</code></td></tr></table></div>`;
    }
    if (s) {
      const hasSup = s.objects.some((o) => o.support), hasVis = s.objects.some((o) => o.visible);
      const hasLaw = s.objects.some((o) => o.law);
      const dims = s.objects.map(dimsBySource), cols = DIM_SOURCES.filter((k) => dims.some((d) => d[k].length));
      const rows = s.objects.map((o, i) => {
        return `<tr><td><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span></td><td>${esc(o.name)}</td><td>${SHAPE_LABEL[o.shape] || esc(o.shape)}</td>` +
          cols.map((k) => `<td class="dims dims-${k}">${dims[i][k].join('')}</td>`).join('') + (hasSup ? `<td>${esc(supportText(o))}</td>` : '') +
          (hasLaw ? `<td>${lawText(o.law)}</td>` : '') +
          (hasVis ? `<td>${o.visible ? o.visible.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ') + ' s' : '<span class="muted">—</span>'}</td>` : '') +
          `<td class="n">${fmt(o.rms, 1)}</td></tr>`;
      }).join('');
      h += `<div class="scroll-x"><table class="objs"><tr><th></th><th>Object</th><th>Shape</th>${cols.map((k) => `<th class="dims-${k}">${k[0].toUpperCase() + k.slice(1)} (m)</th>`).join('')}` +
        (hasSup ? '<th>Support</th>' : '') + (hasLaw ? '<th>Motion law</th>' : '') + (hasVis ? '<th>Visible</th>' : '') + `<th class="n">Residual (px)</th></tr>${rows}</table></div>` +
        agentDimsTable(s) + fullExtentTable(s) + fitObjectTable(s) + segfitTable(s) + chainTable(s, v.duration) + eventsTable(s) + impulseTable(s) + inclineBlock(s) + pendulumTable(s) +
        (s.camera ? `<p class="facts"><span title="${esc(s.camera.how)}">f ${plusMinus(s.camera.f, s.camera.sigma)} px${s.camera.source ? ` (${esc(s.camera.source)}${wasText(s.camera.fused)})` : ''}</span>`
          : `<p class="facts">f = ${fmtNum(s.f)} px${s.f_source ? ` · ${esc(s.f_source)}` : ''}`) +
        (s.f_first_clamp ? ` · first estimate ${fmtNum(s.f_first_clamp.from)} → ${fmtNum(s.f_first_clamp.to)} px (band ${fmtNum(s.f_first_clamp.band[0])}–${fmtNum(s.f_first_clamp.band[1])})` : '') +
        (s.focal && s.focal.f_votes != null ? ` · votes ${fmtNum(s.focal.f_votes)} ± ${fmtNum(s.focal.sigma)} px (${esc(s.focal.n_votes)})` : '') +
        (s.main_support ? ` · main support ${esc(s.main_support)}` : '') + '</p>' + cameraOneTable(s.camera_one) + fusionTable(s.fusion, q) + consistentTable(s.camera) + focalTable(s.focal) + agencyUnknown(s.camera) + supportPlaneTable(s.support_plane) + mainPlaneLines(s.main_plane, s.main_support) + horizonLines(s.horizon) + rollLines(s.camera_roll) +
        (s.scale ? `<p class="facts">scale ${fmtNum(s.scale.s)} · ${esc(s.scale.source)}${s.scale.object ? ` (${esc(s.scale.object)})` : ''} · ${esc(s.scale.prior)} · ${esc(s.scale.extent || s.scale.how)}</p>` +
          (s.scale.checks ? checksTable(s.scale.checks) : '') : '') + planeFusionTable(s.plane_fusion) + scaleContactBlock(s.scale_contact) + depthRowsTable(s.depth_rows, q.id) + planeSigmaTable(s.plane_sigma) +
        gravityBlock(s.gravity) +
        focalVotes(s);
    }
    if (q.measurement || q.provenance || q.quality) {
      h += `<details class="sfres"><summary>Diagnostics</summary><pre>${esc(JSON.stringify({
        scene_id: q.scene_id, row: q.row, method: q.method, source: q.src, status: q.status, failed: q.failed,
        execution_status: q.execution_status,
        measurement: q.measurement, measurement_id: q.measurement_id, measurement_record: q.measurement_record,
        quality: q.quality, provenance: q.provenance }, null, 2))}</pre></details>`;
    }
    return h;
  }
  // the motion-law test of every piece (rule segfit.point): tested point, its noise σ, bound, rms, run of failing frames
  function segfitTable(s) {
    const os = s.objects.filter((o) => o.segfit && o.segfit.length);
    if (!os.length) return '';
    const px = (a) => a.map((x) => fmtNum(x)).join(' / ');
    return '<div class="scroll-x"><table class="objs segfit"><tr><th>Object</th><th>Piece</th><th>Law</th><th>Point</th><th class="n">σ (px)</th>' +
      '<th class="n">Bound (px)</th><th class="n">rms (px)</th><th class="n">Run</th><th></th></tr>' +
      os.map((o) => o.segfit.map((p, i) => `<tr><td>${i ? '' : esc(o.name)}</td><td>${span(p.t0, p.t1)}</td><td><code>${esc(p.model)}</code></td>` +
        `<td>${esc(p.point)}</td><td class="n">${px(p.sigma)}</td><td class="n">${px(p.bound)}</td><td class="n">${p.rms == null ? '—' : fmtNum(p.rms)}</td>` +
        `<td class="n">${p.run} / ${p.n_test}</td><td class="${p.pass === true ? 'ok' : p.pass === false ? 'bad' : 'muted'}">${p.pass === true ? 'pass' : p.pass === false ? 'fail' : 'untestable'}</td></tr>`).join('')).join('') +
      '</table></div>';
  }
  // the boundaries of each physics chain, each a scene-graph event (rule events.graph)
  function eventsTable(s) {
    const os = s.objects.filter((o) => o.events_graph && o.events_graph.some((e) => e.t != null));
    if (!os.length) return '';
    return '<div class="scroll-x"><table class="objs events"><tr><th>Object</th><th class="n">Phase</th><th>Graph event</th><th class="n">t (s)</th>' +
      '<th class="n">Window (s)</th><th>Precision</th><th class="n">t fit (s)</th><th>Kick</th></tr>' +
      os.map((o) => o.events_graph.map((e, i) => `<tr><td>${i ? '' : esc(o.name)}</td><td class="n">${e.phase}</td><td>${esc(e.event)}</td>` +
        `<td class="n">${e.t == null ? '—' : Array.isArray(e.t) ? `${fmt(e.t[0], 3)}–${fmt(e.t[1], 3)}` : fmt(e.t, 3)}</td><td class="n">${e.window ? `${fmt(e.window[0], 3)}–${fmt(e.window[1], 3)}` : '—'}</td>` +
        `<td>${esc(e.precision || '')}</td><td class="n">${fmt(e.t_fit, 3)}</td><td>${e.kick ? 'yes' : ''}</td></tr>`).join('')).join('') +
      '</table></div>';
  }
  // every impulse decision of each physics chain (rule law.impulse): the agent that delivers it, or the track's
  // velocity jump against its noise, or why the state law needs none
  function impulseTable(s) {
    const os = s.objects.filter((o) => o.impulse);
    if (!os.length) return '';
    const cls = (v) => (v === 'kick' || v === 'agent.unobserved' ? 'ok' : v.startsWith('rejected') ? 'bad' : '');
    return '<div class="scroll-x"><table class="objs impulse"><tr><th>Object</th><th class="n">t (s)</th><th>Event</th><th>Agent</th>' +
      '<th class="n">Jump (px/s)</th><th><code>law.impulse</code></th><th></th></tr>' +
      os.map((o) => o.impulse.map((x, i) => `<tr><td>${i ? '' : esc(o.name)}</td><td class="n">${fmt(x.t, 3)}</td><td>${esc(x.event)}</td>` +
        `<td>${x.agent ? `${esc(x.agent)}${x.source ? ` <span class="muted">${esc(x.source)}</span>` : ''}` : '<span class="muted">none</span>'}</td>` +
        `<td class="n"${x.jump_how ? ` title="${esc(x.jump_how)}"` : ''}>${x.jump != null ? plusMinus(x.jump, x.sigma) : ''}</td>` +
        `<td class="${cls(x.verdict)}">${esc(x.verdict)}</td><td>${esc(x.why)}</td></tr>`).join('') +
        (o.impulse_span ? `<tr class="absorbed"><td></td><td class="n">${fmt(o.impulse_span.onset, 3)}</td><td><code>law.impulse_span</code></td>` +
          `<td colspan="2">driven ${o.impulse_span.ramp_end == null ? 'to the next phase' : `${fmt(o.impulse_span.onset, 3)}–${fmt(o.impulse_span.ramp_end, 3)} s`} · ${o.impulse_span.n_driven} driven frames · first frame ${fmt(o.impulse_span.first_frame, 3)} s</td>` +
          `<td>${o.impulse_span.applied ? 'applied' : '<span class="muted">not applied</span>'}${o.impulse_span.free_piece ? ' · free piece' : ''}</td><td>${esc(o.impulse_span.how)}</td></tr>` : '')).join('') +
      '</table></div>';
  }
  // the measured slope of each inclined support and the rolling law it gives (rule support.incline), with the
  // refined onset of each start from rest (rule kinematic.onset)
  function inclineBlock(s) {
    let h = '';
    (s.incline || []).forEach((x) => {
      h += `<p class="facts"><code>support.incline</code> ${esc(x.support)} · <span title="${esc(x.why)}">θ ${plusMinus(x.theta, x.sigma)}°</span>` +
        `${x.sigma_f == null && x.sigma_f_how ? ` <span class="muted">(${esc(x.sigma_f_how)})</span>` : x.sigma_f != null ? ` · σ_f ${fmtNum(x.sigma_f)}°` : ''}` +
        `${x.length != null ? ` · ${fmtNum(x.length)} m` : ''} · ${esc(x.method)} · <span class="${x.inclined ? '' : 'muted'}">${x.inclined ? 'inclined' : 'not inclined'}</span></p>` +
        `<p class="facts">up: ${esc(x.up.source)} · <span class="${x.up.level ? '' : 'bad'}" title="${esc(x.up.why)}">${x.up.level ? 'level' : 'not level'} · ${plusMinus(x.up.angle, x.up.sigma)}°</span></p>`;
    });
    const ls = s.objects.filter((o) => o.incline_law || o.onset);
    if (ls.length) {
      h += '<div class="scroll-x"><table class="objs"><tr><th>Object</th><th class="n">Phase</th><th class="n">t_on (s)</th><th>Support</th>' +
        '<th class="n">a (m/s²)</th><th class="n">a geometry (m/s²)</th><th class="n">κ</th><th class="n">θ (°)</th><th>Law</th></tr>' +
        ls.map((o) => {
          const ph = new Set([...(o.onset || []).map((x) => x.phase), ...(o.incline_law || []).map((x) => x.phase)]);
          return [...ph].sort((a, b) => a - b).map((p, i) => {
            const on = (o.onset || []).find((x) => x.phase === p), l = (o.incline_law || []).find((x) => x.phase === p);
            const t = l ? plusMinus(l.t_on, l.sigma_t_on) : on ? plusMinus(on.t, on.sigma) : '';
            return `<tr><td>${i ? '' : esc(o.name)}</td><td class="n">${p}</td><td class="n"${on ? ` title="${esc(on.how)}"` : ''}>${t}</td>` +
              (l ? `<td>${esc(l.support)}</td><td class="n">${plusMinus(l.accel, l.sigma_accel)} <span class="muted">${esc(l.accel_source)}</span></td>` +
                `<td class="n">${fmtNum(l.accel_geometry)}</td><td class="n" title="${esc(l.shape)}">${fmtNum(l.kappa)}</td><td class="n">${fmtNum(l.theta)}</td><td>${esc(l.law)}</td>`
                : '<td></td><td></td><td></td><td></td><td></td><td></td>') + '</tr>';
          }).join('');
        }).join('') + '</table></div>';
    }
    return h;
  }
  // pendulum objects (rules scale.converged / pendulum.period): fitted L, swing period, f solved with the stated size
  function pendulumTable(s) {
    const ps = s.objects.filter((o) => o.pendulum);
    if (!ps.length) return '';
    return '<div class="scroll-x"><table class="objs"><tr><th>Pendulum</th><th class="n">L (m)</th><th class="n">Period (s)</th>' +
      '<th class="n">f start → f (px)</th><th class="n">f bounds (px)</th><th class="n">D / D stated (m)</th><th>Converged</th></tr>' +
      ps.map((o) => {
        const p = o.pendulum, ff = p.f_free;
        return `<tr><td>${esc(o.name)}</td><td class="n">${fmtNum(p.L)}</td><td class="n">${plusMinus(p.period, p.period_sigma)} <span class="muted">(${p.periods}, ${esc(p.period_from)})</span></td>` +
          (ff ? `<td class="n">${fmtNum(ff.f_start)} → ${fmtNum(ff.f)}</td><td class="n${ff.at_bound ? ' bad' : ''}">${fmtNum(ff.bounds[0])}–${fmtNum(ff.bounds[1])}${ff.at_bound ? ` · ${esc(ff.at_bound)}` : ''}</td>` +
            `<td class="n">${fmtNum(ff.D)} / ${fmtNum(ff.D_prior)} <span class="muted">(${fmt(100 * ff.rel, 1)}%)</span></td><td class="${ff.converged ? 'ok' : 'bad'}">${ff.converged ? 'yes' : 'no'}</td>`
            : `<td class="n">${fmtNum(p.f)}</td><td></td><td></td><td></td>`) + '</tr>';
      }).join('') + '</table></div>';
  }
  function checksTable(cs, extra = '') {
    return '<div class="scroll-x"><table class="checks"><tr><th>Check</th><th></th><th>Reason</th></tr>' +
      cs.map((c) => `<tr class="${c.pass ? 'ok' : 'bad'}"><td><code>${esc(c.name)}</code></td><td>${c.pass ? 'pass' : 'fail'}</td><td>${esc(c.reason)}</td></tr>`).join('') +
      extra + '</table></div>';
  }
  // the unit's reference measurements (step 5.1): object, quantity, graduation unit, length in px, named checks
  function refmeasureBlock(rm) {
    if (!rm) return '';
    return '<h4>Reference measurement</h4>' + rm.map((r) =>
      `<p class="facts">${esc(r.obj)} · ${esc(r.quantity)}${r.unit ? ` · ${esc(r.unit)}` : ''}${r.px != null ? ` · ${fmtNum(r.px)} px` : ''}` +
      `${r.frame != null ? ` · frame ${r.frame}` : ''}${r.method ? ` · ${esc(r.method)}` : ''}</p>` + checksTable(r.checks)).join('');
  }
  // the scene's gravity (rule B2): g and its source, every ballistic piece / physics probe it was judged on, its checks
  function gravityBlock(g) {
    if (!g) return '';
    const ev = g.evidence.map((e) => e.kind === 'ballistic_piece'
      ? `<tr class="${e.use ? '' : 'inadm'}"><td>${esc(e.obj)}</td><td>${span(e.t[0], e.t[1])}</td><td class="n">${e.n}</td>` +
        `<td class="n">${plusMinus(e.g, e.sigma)}</td><td class="${e.law.pass ? 'ok' : 'bad'}">${e.law.pass ? 'pass' : 'fail'} · run ${e.law.run} / ${e.law.n_test}</td>` +
        `<td class="${e.admissible.pass ? 'ok' : 'bad'}">${esc(e.admissible.reason)}</td>` +
        `<td>${e.use ? 'used' : esc(e.why && e.why !== e.admissible.reason ? e.why : 'not used')}</td></tr>`
      : `<tr><td>${esc(e.obj)}</td><td>physics probe</td><td class="n"></td><td class="n">${fmtNum(e.g)}</td><td>ρ ${fmtNum(e.rho)}</td><td></td>` +
        `<td class="${e.agree ? 'ok' : 'bad'}">${e.agree ? 'agrees with Earth' : 'disagrees with Earth'}</td></tr>`).join('');
    const et = g.earth_test ? `<tr class="et"><td><code>earth_test</code></td><td>${g.earth_test.pass ? 'pass' : 'fail'}</td><td>${esc(g.earth_test.reason)}</td></tr>` : '';
    return `<p class="facts">g = ${plusMinus(g.g, g.sigma)} m/s² · ${esc(g.source)} · ${esc(g.rule)} · up: ${esc(g.up_source)}</p>` +
      (ev ? '<div class="scroll-x"><table class="gev"><tr><th>Object</th><th>Piece</th><th class="n">n</th><th class="n">g (m/s²)</th><th>Law residual</th><th>Admissible</th><th></th></tr>' + ev + '</table></div>' : '') +
      checksTable(g.checks, et);
  }
  // ---------- v5: every measurement of the asked quantity, the checks agree / constraints ----------
  const span = (a, b) => (a != null && a === b ? `${fmt(a, 2)} s` : `${a == null ? '…' : fmt(a, 2)}–${b == null ? '…' : fmt(b, 2)} s`);
  const span3 = ([a, b]) => `${fmt(a, 3)}–${fmt(b, 3)} s`;
  const plusMinus = (v, sg) => `${fmtNum(v)}${sg != null ? ` ± ${fmtNum(sg)}` : ''}`;
  function sigmaSub(m) {
    const j = (a) => a.map((x) => fmtNum(x)).join(' / ');
    const terms = m.terms ? Object.entries(m.terms).filter(([, v]) => v != null).map(([k, v]) => `${k} ${fmtNum(v)}`) : [];
    const parts = [...terms, m.scale && `scale ${j(m.scale)}`, m.dof && `dof ${m.dof.join(' / ')}`, m.q && `q ${j(m.q)}`].filter(Boolean);
    return parts.length ? `<span class="msub">${esc(parts.join(' · '))}</span>` : '';
  }
  function measurementRow(m, q) {
    const chosen = q.chosen;
    const val = m.value == null ? '<span class="muted">—</span>'
      : `<span>${plusMinus(m.value, m.sigma)} ${esc(m.unit)}</span>${sigmaSub(m)}`;
    const inp = [!m.wstate && m.window && `window ${span(m.window[0], m.window[1])}`, m.n != null && `n ${m.n}`,
      !m.chain && m.piece && `piece ${span(m.piece[0], m.piece[1])}`, m.state, m.law && `ballistic g ${fmtNum(m.law.g)} (${m.law.source})`].filter(Boolean).join(' · ');
    const failed = m.failed.map((c) => `<div><code>${esc(c.which || c.name)}</code> ${esc(c.reason)}</div>`).join('');
    const prof = m.profile ? m.profile.map((p) => `<div class="${p.pass ? 'ok' : 'bad'}"><code>at_bound.profile</code> ${esc(p.param)} ` +
      `${fmtNum(p.bound)} → ${fmtNum(p.values[0])} · ${fmtNum(p.at)} → ${fmtNum(p.values[1])} · |Δ| ${fmtNum(p.delta)} ${p.pass ? '<' : '≥'} σ ${fmtNum(p.sigma_answer)}</div>`).join('') : '';
    return `<tr class="${m.name === chosen ? 'chosen' : ''}${m.admissible ? '' : ' inadm'}"><td><code>${esc(m.name)}</code></td>` +
      `<td><code>${esc(m.method)}</code></td><td class="n mval">${val}</td>` +
      `<td class="${m.admissible ? 'ok' : 'bad'}">${m.admissible ? 'admissible' : esc(m.why)}</td>` +
      `<td class="bad">${failed}</td><td><span class="muted">${esc(inp)}</span>${sigmaLines(m)}${directLines(m, q)}${impulseLine(m.window_impulse)}${onsetLine(m.kin_onset)}${prof}</td></tr>`;
  }
  // the window this measurement reads inside the piece after a kick's ramp (rule window.impulse)
  function impulseLine(w) {
    if (!w) return '';
    return `<div class="${cls3(w.pass)}" title="${esc(w.reason)}"><code>window.impulse</code> ${res3(w.pass)} · window ${span3(w.window)} · piece ${span3(w.piece)}</div>`;
  }
  // a law-driven start from rest bounds the speed (rule kinematic.onset): |v| <= ∫|a| dt + 3σ
  function onsetLine(k) {
    if (!k) return '';
    return `<div class="${cls3(k.pass)}"><code>kinematic.onset</code> ${res3(k.pass)} · |v| ${fmtNum(k.v)} ${k.pass === false ? '>' : '≤'} ` +
      `${fmtNum(k.integral)} + 3 × ${fmtNum(k.sigma)} = ${fmtNum(k.bound)} · t_on ${plusMinus(k.t_on, k.sigma_t_on)} s</div>`;
  }
  // how the measurement's sigma was found: the chain piece holding t with its fixed events (rule fit.sigma_chain)
  function sigmaLines(m) {
    let h = '';
    const c = m.chain;
    if (c && c.chain === false) {
      h += `<div>piece ${fmt(c.piece[0], 3)}–${fmt(c.piece[1], 3)} s · motion piece, no physics chain</div>`;
    } else if (c) {
      const s3 = (x) => fmt(x, 3);
      const ev = c.events.map(([t, k]) => `${esc(k)} ${s3(t)} s`).join(' · ');
      const more = [c.window_short.length && `<code>window_short</code> ${esc(JSON.stringify(c.window_short))}`,
        c.events_near_t.length && `events near t ${esc(JSON.stringify(c.events_near_t))}`].filter(Boolean);
      h += `<div><code>fit.sigma_chain</code> piece ${s3(c.piece[0])}–${s3(c.piece[1])} s · ${ev}${more.length ? ' · ' + more.join(' · ') : ''}</div>`;
    }
    if (m.sigma_how) h += `<div class="muted">${esc(m.sigma_how)}</div>`;
    if (m.sigma_flags) h += Object.entries(m.sigma_flags).map(([k, v]) => `<div class="bad"><code>${esc(k)}</code> ${esc(JSON.stringify(v))}</div>`).join('');
    return h;
  }
  // the direct path's state span (rule window.state) and its support-plane depth with the silhouette cross-check (C12)
  function directLines(m, q) {
    let h = '';
    const w = m.wstate;
    if (w) {
      const s3 = (a, b) => `${a == null ? '…' : fmt(a, 3)}–${b == null ? '…' : fmt(b, 3)} s`;
      h += `<div><code>window.state</code> ${esc(w.state || '—')} · span ${s3(w.t0, w.t1)} · clipped by ${esc(w.start || '—')} / ${esc(w.end || '—')}` +
        (w.used ? ` · window ${s3(w.used[0], w.used[1])}` : '') + (w.check ? ` · ${w.law ? esc(w.law) : 'no law'} · <code>${esc(w.check)}</code>` : '') + '</div>';
      if (m.centred) {
        h += `<div><code>window.centred</code> from ${s3(m.centred.from[0], m.centred.from[1])} · extremum ${esc(m.centred.extremum)} · stop ${esc(m.centred.stop)}</div>`;
      }
      if (w.driven) {
        h += `<div><code>window.driven</code> ${esc(w.driven.state)} · <span title="${esc(w.driven.how)}">noise ${w.driven.noise.map((x) => fmtNum(x)).join(' / ')} px</span></div>`;
      }
      if (/^window\.short/.test(m.why)) h += `<div class="bad"><code>window.short</code> ${esc(m.why.replace(/^window\.short:\s*/, ''))}</div>`;
    }
    const d = m.support_depth;
    if (d) {
      h += d.applies ? `<div><code>support_depth</code> ${fmtNum(d.z)} m · ${d.n} frames · plane σ ${fmtNum(d.sigma_plane)} m · row σ ${fmtNum(d.sigma_row)} m</div>`
        : `<div class="muted"><code>support_depth</code> ${esc(d.reason)}</div>`;
    }
    const a = m.depth_agree;
    if (a && a.pass == null) {
      h += `<div class="muted"><code>direct.depth_agree</code> ${esc(a.reason)}</div>`;
    } else if (a) {
      const rows = (q.scene && q.scene.depth_rows) || [], i = rows.findIndex((r) => r.reason === a.reason);
      const see = a.see ? ` · ${a.dropped ? `${a.n_rows} rows dropped · ` : ''}<a href="#${esc(depthRowId(q.id, i))}"><code>${esc(a.see)}</code></a>` : '';
      h += `<div class="${a.pass ? 'ok' : 'bad'}" title="${esc(a.reason)}"><code>direct.depth_agree</code> ${a.pass ? 'pass' : 'fail'} · silhouette ${fmtNum(a.z_mask)} m vs plane ${fmtNum(a.z_plane)} m · |Δ| ${fmtNum(a.diff)} ${a.pass ? '≤' : '>'} ${fmtNum(a.bound)}${see}</div>`;
    }
    const e = m['size.full_extent'];
    if (e) {
      h += `<div class="${cls3(e.pass)}" title="${esc(e.reason)}"><code>size.full_extent</code> ${res3(e.pass)} · ${e.used.length} frames · ${fmtNum(e.extent_px)} px · h z ${fmtNum(e.extent_norm)} px m</div>`;
    }
    return h;
  }
  // the kick's ramp and the piece after it (rule window.impulse): frame-pair 3D speeds after the onset, the ramp's end
  // at their first maximum, the piece from there, and the direct measurement's minimal window, whose verdict this is
  function impulseBlock(w) {
    if (!w) return '';
    const W = 600, H = 150, L = 44, R = 12, T = 12, B = 26;
    const ts = w.speeds.map((p) => p[0]), vs = w.speeds.map((p) => p[1]);
    const t0 = Math.min(w.onset, ...ts, w.window[0]), t1 = Math.max(...ts, w.window[1]);
    const pad = (t1 - t0) * 0.04, x0 = t0 - pad, x1 = t1 + pad, vmax = Math.max(...vs) * 1.15;
    const X = (t) => L + (W - L - R) * (Math.min(Math.max(t, x0), x1) - x0) / (x1 - x0), Y = (v) => T + (H - T - B) * (1 - v / vmax);
    const band = (a, b, cls) => `<rect class="${cls}" x="${X(a)}" y="${T}" width="${Math.max(X(b) - X(a), 1)}" height="${H - T - B}"></rect>`;
    const vline = (t, cls) => `<line class="${cls}" x1="${X(t)}" x2="${X(t)}" y1="${T}" y2="${H - B}"></line>`;
    const xt = [w.onset, w.ramp_end].map((t) => `<text x="${X(t)}" y="${H - 8}" text-anchor="middle">${fmt(t, 3)}</text>`).join('');
    const yt = [0, vmax / 1.15].map((v) => `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, 2)}</text>`).join('');
    const svg = `<svg class="impchart" viewBox="0 0 ${W} ${H}" role="img" aria-label="frame-pair speeds after the onset">` +
      band(w.piece[0], w.piece[1], 'piece') + band(w.window[0], w.window[1], `win ${cls3(w.pass)}`) + vline(w.onset, 'onset') + vline(w.ramp_end, 'ramp') +
      `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"></line>` +
      `<polyline points="${w.speeds.map(([t, v]) => `${X(t)},${Y(v)}`).join(' ')}"></polyline>` +
      w.speeds.map(([t, v]) => `<circle cx="${X(t)}" cy="${Y(v)}" r="4"><title>${fmt(t, 3)} s · ${fmt(v, 3)} m/s</title></circle>`).join('') + xt + yt + '</svg>';
    return `<div class="impulse-win"><p class="facts"><code>window.impulse</code> ${esc(w.verdict)}${w.agent ? ` by ${esc(w.agent)}` : ''} · onset ${fmt(w.onset, 3)} s · ` +
      `<span class="ramp">ramp end ${fmt(w.ramp_end, 3)} s</span> · <span class="piece">piece ${span3(w.piece)}</span> · first frame ${fmt(w.first_frame, 3)} s · ` +
      `<span class="win ${cls3(w.pass)}" title="${esc(w.reason)}">direct window ${span3(w.window)} ${res3(w.pass)}</span></p>${svg}</div>`;
  }
  function measurementBlock(q) {
    let h = '<div class="scroll-x"><table class="meas"><tr><th>Measurement</th><th>Method</th><th class="n">Value</th><th>Admissible</th><th>Failed</th><th>Inputs</th></tr>' +
      q.measurements.map((m) => measurementRow(m, q)).join('') + '</table></div>' + impulseBlock(q.window_impulse);
    const other = q.measurements.filter((m) => m.name !== q.chosen && m.checks.length);
    if (other.length) {
      h += '<div class="scroll-x"><table class="checks mchecks"><tr><th>Check</th><th></th><th>Reason</th></tr>' +
        other.map((m) => m.checks.map((c, i) => `<tr class="${c.pass ? 'ok' : 'bad'}${i ? '' : ' mfirst'}"><td><code>${esc(m.name)}</code> <code>${esc(c.name)}</code></td>` +
          `<td>${c.pass ? 'pass' : 'fail'}</td><td>${esc(c.reason)}</td></tr>`).join('')).join('') + '</table></div>';
    }
    const a = q.agree;
    if (a.values.length) {
      const chi = a.chi2 != null ? `χ² ${fmtNum(a.chi2)} ${a.chi2 > a.bound ? '>' : '≤'} ${fmtNum(a.bound)} (dof ${a.dof})` : '<span class="muted">—</span>';
      h += '<div class="scroll-x"><table class="checks agree"><tr><th>Check</th><th></th><th class="n">χ²</th><th>Values</th></tr>' +
        `<tr class="${a.pass ? 'ok' : 'bad'}"><td><code>agree</code></td><td>${a.pass ? 'pass' : 'fail'}</td><td class="n">${chi}</td>` +
        `<td>${a.values.map(([k, v, sg]) => `<code>${esc(k)}</code> ${plusMinus(v, sg)}`).join(' · ')}</td></tr></table></div>`;
    }
    const tested = q.constraints.filter((c) => c.pass !== null), untested = q.constraints.filter((c) => c.pass === null && !c.inadmissible);
    const inadm = q.constraints.filter((c) => c.inadmissible);
    if (tested.length) {
      const anyNo = tested.some((c) => !c.testable);
      h += '<div class="scroll-x"><table class="checks cons"><tr><th>Constraint</th><th></th><th class="n">Observed</th><th class="n">Predicted</th>' +
        `<th class="n">|Δ|</th><th class="n">Bound</th><th>Stated</th><th class="n">σ stated</th><th class="n">σ pred</th>${anyNo ? '<th>Testable</th>' : ''}</tr>` +
        tested.map((c) => `<tr class="${c.pass ? 'ok' : 'bad'}"><td><code>${esc(c.id)}</code></td><td>${c.pass ? 'pass' : 'fail'}</td>` +
          `<td class="n">${fmtNum(c.observed)}</td><td class="n">${fmtNum(c.predicted)}</td><td class="n">${fmtNum(c.diff)}</td>` +
          `<td class="n">${fmtNum(c.bound)}</td><td>${c.stated && c.stated.text ? esc(c.stated.text) : ''}</td><td class="n">${fmtNum(c.sigma_stated)}</td><td class="n">${fmtNum(c.sigma_pred)}</td>` +
          `${anyNo ? `<td>${c.testable ? 'yes' : 'no'}</td>` : ''}</tr>`).join('') + '</table></div>';
    }
    if (untested.length) {
      h += '<div class="scroll-x"><table class="checks cons"><tr><th>Constraint</th><th></th><th class="n">Observed</th><th class="n">Predicted</th>' +
        '<th>Stated</th><th class="n">σ pred</th><th>Reason</th></tr>' +
        untested.map((c) => `<tr class="muted"><td><code>${esc(c.id)}</code></td><td>untestable</td>` +
          `<td class="n">${fmtNum(c.observed)}</td><td class="n">${fmtNum(c.predicted)}</td><td>${c.stated && c.stated.text ? esc(c.stated.text) : ''}</td>` +
          `<td class="n" title="${esc(c.untestable.source)}">${fmtNum(c.untestable.sigma_pred)}</td><td>${esc(c.untestable.reason)}</td></tr>`).join('') + '</table></div>';
    }
    if (inadm.length) {
      h += '<div class="scroll-x"><table class="checks cons"><tr><th>Constraint</th><th></th><th class="n">Observed</th><th class="n">Predicted</th>' +
        '<th>Stated</th><th>Rule</th><th>Reason</th></tr>' +
        inadm.map((c) => `<tr class="inadm"><td><code>${esc(c.id)}</code></td><td>inadmissible</td>` +
          `<td class="n">${fmtNum(c.observed)}</td><td class="n">${fmtNum(c.predicted)}</td><td>${c.stated && c.stated.text ? esc(c.stated.text) : ''}</td>` +
          `<td><code>${esc(c.inadmissible.rule)}</code></td><td>${esc(c.inadmissible.reason.replace(/^fit\.object:\s*/, ''))}</td></tr>`).join('') + '</table></div>';
    }
    h += `<div class="scroll-x"><table class="checks"><tr><th>Check</th><th></th><th>Reason</th></tr>` +
      q.checks.map((c) => `<tr class="${c.pass ? 'ok' : 'bad'}"><td><code>${esc(c.which || c.name)}</code></td>` +
        `<td>${c.pass ? 'pass' : 'fail'}</td><td>${esc(c.reason)}${c.name === 'plausible' ? dimsCheck(q.dims_check) + partsUntestable(q.plausible_untestable) : ''}</td></tr>`).join('') + '</table></div>';
    return h;
  }
  // plausible parts that could not judge an object (critic.agency_unknown)
  function partsUntestable(rows) {
    if (!rows) return '';
    return rows.map(([part, rs]) => rs.map((r) => `<div class="muted"><code>${esc(part)}</code> untestable · ${esc(r)}</div>`).join('')).join('');
  }
  // plausible.dims: each object's largest dimension (an interval when a dimension is bound) against its class range
  function dimsCheck(rows) {
    if (!rows || !rows.length) return '';
    return '<table class="dimchk"><tr><th>Object</th><th class="n">Largest dimension (m)</th><th class="n">Class range (m)</th></tr>' +
      rows.map((r) => `<tr class="${r.pass ? '' : r.pass === null ? 'muted' : 'bad'}"><td>${esc(r.obj)}</td>` +
        `<td class="n">${r.hi === r.lo ? fmtNum(r.lo) : interval(r.lo, r.hi)}</td>` +
        `<td class="n">${r.range ? interval(r.range[0], r.range[1]) : '<span class="muted">—</span>'}</td></tr>`).join('') + '</table>';
  }
  // a scene object's motion law (v5 motion_law): the law, its rotation relation and pieces, or its state phases
  function lawText(l) {
    if (!l) return '<span class="muted">—</span>';
    const parts = [esc(l.model)];
    if (l.rotation) parts.push(`${esc(l.rotation.rel)} ${esc(l.rotation.target || 'a fixed point')}`, l.rotation.pieces.map(esc).join(' → '));
    if (l.phases) parts.push(l.phases.map(([t, st]) => `${esc(st)} ${fmt(t, 2)}`).join(', ') + ' s');
    if (l.unknown) parts.push(`<span class="bad">unknown ${l.unknown.map(([a, b]) => span(a, b)).join(', ')}</span>`);
    if (l.agency) parts.push(`<span class="muted">agency untestable · ${esc(l.agency.why)}</span>`);
    if (l.release_fit) {
      const r = l.release_fit;
      parts.push(`<span title="${esc(r.why)}"><code>release.fit</code> ${r.t == null ? 'none' : `${fmt(r.t, 3)} s`} in ${fmt(r.lo, 3)}–${fmt(r.hi, 3)} s, flight to ${fmt(r.t_stop, 3)} s</span>`);
    }
    return parts.join(' · ');
  }
  // wall time of the question's own run, per pipeline step (answer record "timing"); reused = served from a sibling
  // question's vision memo (SAM 2 propagation, floor masks), cached = this question's file from an earlier run kept
  const STEP_LABEL = { objects: 'Objects', scenegraph_grid: 'Scene graph (grid)', track: 'Tracking', measure: 'Measure',
    scenegraph_pf: 'Scene graph (per frame)', observe: 'Observe', ground: 'Ground', scene: 'Physics-guided Fitting', agent: 'Agent loop', answer: 'Answer' };
  function reuseText(x) {
    if (x.cached) return 'cached';
    if (!x.reused) return '';
    const sh = x.shared || {};
    const hits = sh.sam2_hits != null ? ` (SAM 2 ${sh.sam2_hits}/${sh.sam2_hits + (sh.sam2_misses || 0)})`
      : sh.ground_floor === 'shared' ? ' (floor)' : sh.support_masks === 'shared' ? ' (support masks)' : '';
    return (x.reused === true ? 'reused' : 'partly reused') + hits;
  }
  function timeBlock(q) {
    if (!q.time) return '';
    const marks = q.time.steps.map(reuseText), any = marks.some(Boolean);
    const rows = q.time.steps.map((x, i) => `<tr${marks[i] ? ' class="reused"' : ''}><td>${esc(STEP_LABEL[x.step] || x.step)}</td><td class="n">${fmt(x.s, 1)}</td>` +
      (any ? `<td>${esc(marks[i])}</td>` : '') + '</tr>').join('');
    return '<h4>Time</h4><div class="scroll-x"><table class="tsteps"><tr><th>Step</th><th class="n">s</th>' + (any ? '<th></th>' : '') + `</tr>${rows}` +
      `<tr class="ttotal"><td>Total</td><td class="n">${fmt(q.time.total, 1)}</td>${any ? '<td></td>' : ''}</tr></table></div>`;
  }
  // ---------- three.js scene viewer ----------
  let threeP = null;
  function loadThree() {
    if (!threeP) threeP = Promise.all([import('three'), import('three/addons/controls/OrbitControls.js')]);
    return threeP;
  }

  async function startViewer(box, bar, v, video) {
    let THREE, OrbitControls;
    try { [THREE, { OrbitControls }] = await loadThree(); } catch (e) {
      box.querySelector('.msg').textContent = 'three.js failed to load (needs cdn.jsdelivr.net)';
      return null;
    }
    const s = v.scene;
    const T = v.duration;
    const toW = (p) => new THREE.Vector3(p[0], -p[1], -p[2]); // camera frame (X right, Y down, Z fwd) -> three.js (Y up)
    const add = (a, b, k = 1) => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const norm = (a) => Math.hypot(a[0], a[1], a[2]);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    // FitResult.pos: the object rests at p0 until t_on, then moves with v0, a
    // a sampled path (ballistic bounces, motion pieces) is interpolated; otherwise one parabola from t_on
    const pos = (o, t) => {
      const P = o.path;
      if (P && P.length > 1) {
        const dt = P[1][0] - P[0][0], k = Math.min(Math.max((t - P[0][0]) / dt, 0), P.length - 1);
        const i = Math.min(Math.floor(k), P.length - 2), w = k - i;
        return [1, 2, 3].map((c) => P[i][c] * (1 - w) + P[i + 1][c] * w);
      }
      const u = Math.max(t - o.t_on, 0); return add(add(o.p0, o.v0, u), o.a, 0.5 * u * u);
    };
    // tracked in the video at time t (no track info: always)
    const halfF = 0.5 / v.fps + 1e-6;
    const visAt = (o, t) => !o.visible || o.visible.some((iv) => t >= iv[0] - halfF && t <= iv[1] + halfF);

    const pl = s.plane && s.plane.normal ? s.plane : null;
    const n = pl ? pl.normal : [0, -1, 0];
    const u1 = pl && pl.u1 ? pl.u1 : [1, 0, 0];
    const u2 = pl && pl.u2 ? pl.u2 : cross(n, u1);

    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    box.innerHTML = '';
    box.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(3, 10, 4); scene.add(sun);

    // objects
    const objs = s.objects.filter((o) => o.p0 && o.v0 && o.a);
    const pts = [[0, 0, 0]];
    objs.forEach((o) => { for (let k = 0; k <= 10; k++) pts.push(pos(o, (T * k) / 10)); });
    const objPts = pts.slice(1);
    const centre = objPts.length ? [0, 1, 2].map((i) => mean(objPts.map((p) => p[i]))) : [0, 0, 5];
    const radius =Math.max(1, ...pts.map((p) => norm(add(p, centre, -1))));

    // ground grid on the fitted plane (origin = dist * n), covering the object footprint
    if (pl) {
      const O = [n[0] * pl.dist, n[1] * pl.dist, n[2] * pl.dist];
      const c1 = dot(add(centre, O, -1), u1), c2 = dot(add(centre, O, -1), u2);
      const R = radius * 1.3, step = Math.pow(10, Math.floor(Math.log10(R / 5)));
      const nStep = Math.ceil(R / step);
      const g = [];
      for (let i = -nStep; i <= nStep; i++) {
        const a = add(add(O, u1, c1 + i * step), u2, c2 - nStep * step), b = add(add(O, u1, c1 + i * step), u2, c2 + nStep * step);
        const c = add(add(O, u2, c2 + i * step), u1, c1 - nStep * step), d = add(add(O, u2, c2 + i * step), u1, c1 + nStep * step);
        g.push(toW(a), toW(b), toW(c), toW(d));
      }
      const gl = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(g), new THREE.LineBasicMaterial({ color: 0x8a919a, transparent: true, opacity: 0.45 }));
      scene.add(gl);
    }

    // camera frustum at the origin
    const [iw, ih] = v.size;
    const f = s.f;
    const zf = Math.max(0.3, radius * 0.25);
    const corners = [[-iw / 2, -ih / 2], [iw / 2, -ih / 2], [iw / 2, ih / 2], [-iw / 2, ih / 2]].map(([x, y]) => [x / f * zf, y / f * zf, zf]);
    const fr = [];
    corners.forEach((c, i) => { fr.push(toW([0, 0, 0]), toW(c), toW(c), toW(corners[(i + 1) % 4])); });
    const frustum = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(fr), new THREE.LineBasicMaterial({ color: getComputedStyle(document.body).color }));
    scene.add(frustum);

    const meshes = [];
    objs.forEach((o, i) => {
      const col = new THREE.Color(PALETTE[i % PALETTE.length]);
      const d = o.dims;
      let geo, axes;
      // object frame: heading h (yaw, else velocity at mid-time projected on the plane, else u1), lateral e2 = n x h, up n
      let h;
      if (o.yaw != null) h = add([0, 0, 0].map((_, k) => Math.cos(o.yaw) * u1[k]), u2, Math.sin(o.yaw));
      else {
        const vm = add(o.v0, o.a, T / 2);
        h = add(vm, n, -dot(vm, n));
        h = norm(h) > 1e-6 ? h.map((x) => x / norm(h)) : u1;
      }
      const e2 = cross(n, h);
      if (o.shape === 'sphere') { geo = new THREE.SphereGeometry(d.D / 2, 24, 16); axes = [h, e2, n]; }
      else if (o.shape === 'cylinder') { geo = new THREE.CylinderGeometry(d.D / 2, d.D / 2, d.H, 24); axes = [h, n, e2.map((x) => -x)]; }
      else { geo = new THREE.BoxGeometry(d.L, d.W, d.H); axes = [h, e2, n]; }
      const mat = new THREE.MeshStandardMaterial({ color: col, transparent: true, opacity: o.shape === 'mesh' ? 0.35 : 0.8, roughness: 0.7 });
      const m = new THREE.Mesh(geo, mat);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: col }));
      const grp = new THREE.Group(); grp.add(m); grp.add(edges);
      grp.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(toW(axes[0]), toW(axes[1]), toW(axes[2])));
      scene.add(grp);
      const tr = [];
      for (let k = 0; k <= 40; k++) tr.push(toW(pos(o, (T * k) / 40)));
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(tr), new THREE.LineDashedMaterial({ color: col, dashSize: radius * 0.03, gapSize: radius * 0.02 }));
      line.computeLineDistances();
      scene.add(line);
      meshes.push({ o, grp, mat, edges });
    });
    const counter = document.createElement('div');
    counter.className = 'vcount';
    counter.title = 'visible: objects with a track box in this frame; in view: objects whose fitted position projects into the camera image, in front of the camera';
    box.appendChild(counter);
    let tNow = 0;

    const cam = new THREE.PerspectiveCamera(45, 1, Math.max(0.01, radius / 500), radius * 50);
    const tgt = toW(centre);
    cam.position.copy(tgt).add(new THREE.Vector3(radius * 1.2, radius * 1.0, radius * 1.6));
    const controls = new OrbitControls(cam, renderer.domElement);
    controls.target.copy(tgt);
    controls.update();

    // camera view = the original video camera: at the origin looking down +Z (three.js -Z), principal point at the image centre
    // (pipeline/fit3d.py: u = W/2 + f X/Z), vertical fov from f and the image height
    const camC = new THREE.PerspectiveCamera((2 * Math.atan(ih / 2 / f) * 180) / Math.PI, iw / ih, 0.01, radius * 50);
    camC.position.set(0, 0, 0); camC.up.set(0, 1, 0); camC.lookAt(0, 0, -1);
    let active = cam, vtex = null;
    const vbar = box.previousElementSibling && box.previousElementSibling.classList.contains('viewbar') ? box.previousElementSibling : null;
    const btns = vbar ? [...vbar.querySelectorAll('.seg button')] : [];
    const ovBox = vbar && vbar.querySelector('.overlay'), ovLabel = ovBox && ovBox.parentElement;
    function setOverlay(on) {
      if (on && video && !video.error) {
        if (!vtex) { vtex = new THREE.VideoTexture(video); vtex.colorSpace = THREE.SRGBColorSpace; }
        scene.background = vtex;
      } else scene.background = null;
      setTime(tNow);
    }
    const baseOpacity = (o) => (scene.background ? 0.45 : (o.shape === 'mesh' ? 0.35 : 0.8));
    function setView(mode) {
      const camMode = mode === 'cam';
      active = camMode ? camC : cam;
      controls.enabled = !camMode; frustum.visible = !camMode;
      box.classList.toggle('cam', camMode);
      btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === mode)));
      if (ovBox) {
        ovBox.disabled = !camMode || !video || !!video.error;
        ovLabel.classList.toggle('off', ovBox.disabled);
      }
      setOverlay(camMode && ovBox && ovBox.checked);
      resize();
    }
    btns.forEach((b) => { b.onclick = () => setView(b.dataset.view); });
    if (ovBox) ovBox.onchange = () => setOverlay(ovBox.checked);

    // an object is drawn only while the video tracks it: hidden in the camera view, a faint outline in the free view
    function setTime(t) {
      tNow = t;
      let nVis = 0, nIn = 0;
      meshes.forEach(({ o, grp, mat, edges }) => {
        const p = pos(o, t), vis = visAt(o, t);
        grp.position.copy(toW(p));
        grp.visible = vis || active !== camC;
        mat.opacity = vis ? baseOpacity(o) : 0.06;
        edges.material.transparent = true; edges.material.opacity = vis ? 1 : 0.25;
        nVis += vis;
        if (p[2] > 0 && Math.abs((f * p[0]) / p[2]) <= iw / 2 && Math.abs((f * p[1]) / p[2]) <= ih / 2) nIn++;
      });
      counter.textContent = `visible ${nVis} / in view ${nIn}`;
      counter.classList.toggle('bad', nVis !== nIn);
      render();
    }
    function resize() {
      const W = box.clientWidth, H = box.clientHeight;
      if (!W || !H) return;
      if (active === camC) {
        // letterbox to iw:ih inside the box, like the <video> on the left (object-fit: contain, same box size)
        const k = Math.min(W / iw, H / ih), w = Math.round(iw * k), hh = Math.round(ih * k);
        renderer.setSize(w, hh, false);
        renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = hh + 'px';
      } else {
        renderer.setSize(W, H, false);
        renderer.domElement.style.width = ''; renderer.domElement.style.height = '';
        cam.aspect = W / H; cam.updateProjectionMatrix();
      }
      render();
    }
    function render() {
      // VideoTexture only refreshes on its own frame callback; force an upload so a seek while paused shows the right frame
      if (scene.background && vtex && video.readyState >= 2) vtex.needsUpdate = true;
      renderer.render(scene, active);
    }
    controls.addEventListener('change', render);
    const ac = new AbortController();
    const tl = linkTime(bar, video, T, setTime, ac.signal);
    if (video) {
      const o = { signal: ac.signal };
      ['loadeddata', 'canplay', 'seeked'].forEach((ev) => video.addEventListener(ev, () => { if (scene.background) { if (vtex) vtex.needsUpdate = true; render(); } }, o));
      video.addEventListener('error', () => { if (ovBox) { ovBox.checked = false; setView(active === camC ? 'cam' : 'free'); } }, o);
    }
    const ro = new ResizeObserver(resize); ro.observe(box);
    resize(); tl.redraw(); if (vbar) { if (ovBox && video && !video.error) ovBox.checked = true; setView("cam"); }
    return {
      dispose() {
        ac.abort(); tl.stop();
        ro.disconnect(); controls.dispose();
        scene.background = null; if (vtex) vtex.dispose();
        btns.forEach((b) => { b.onclick = null; b.setAttribute("aria-pressed", String(b.dataset.view === "cam")); });
        if (ovBox) { ovBox.onchange = null; ovBox.checked = true; ovBox.disabled = true; ovLabel.classList.add("off"); }
        box.classList.remove('cam');
        scene.traverse((x) => { if (x.geometry) x.geometry.dispose(); if (x.material) x.material.dispose(); });
        renderer.dispose(); renderer.forceContextLoss();
        box.innerHTML = '<div class="msg">Closed</div>';
      },
    };
  }

  // ---------- filtering ----------
  function renderList(items) {
    const list = $('val-list'); list.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'qhead';
    head.innerHTML = '<span></span><span>Question</span><span class="qnums"><span class="qv">GT</span><span class="qv">Pred</span><span>Source</span><span class="qv">Time</span></span><span class="qm">MRA</span>';
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
    const src = $('v-src').value, vsrc = $('v-vsrc').value, mf = $('v-mra').value, sort = $('v-sort').value;
    const needle = $('v-q').value.trim().toLowerCase(), vids = DATA.val.vmap;
    const okM = (m) => !mf || (mf === 'lt0.2' ? m < 0.2 : mf === 'lt0.5' ? m < 0.5 : m >= 0.8);
    // setQuestions() is already in set order (DEV10 order, or video then k)
    const items = setQuestions().filter((q) => (!src || q.src === src) &&
      (!vsrc || vids[q.video].source === vsrc) && okM(q.mra) && matchText(q, needle)).map((q) => ({ q, v: vids[q.video] }));
    const nanLast = (f) => (a, b) => (isNaN(a.q.mra) ? 1 : 0) - (isNaN(b.q.mra) ? 1 : 0) || f(a, b);
    if (sort === 'mra-asc') items.sort(nanLast((a, b) => a.q.mra - b.q.mra));
    else if (sort === 'mra-desc') items.sort(nanLast((a, b) => b.q.mra - a.q.mra));
    $('v-status').textContent = `${items.length} questions` + (items.length ? ` · mean MRA ${fmt(mean(items.map((x) => x.q.mra)), 2)}` : '');
    renderList(items);
  }
  const renderAll = () => { valSummary(); renderVal(); };
  function debounce(fn, ms) { let t = 0; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  fetch('data/videos.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then((d) => {
    DATA = d;
    d.val.vmap = Object.fromEntries(d.val.videos.map((v) => [v.id, v]));
    d.val.runmap = Object.fromEntries(d.val.runs.map((r) => [r.version, r]));
    fillSourceSelect($('v-vsrc'), d.val.videos);
    $('v-set').addEventListener('change', renderAll);
    ['v-src', 'v-vsrc', 'v-mra', 'v-sort'].forEach((id) => $(id).addEventListener('change', renderVal));
    $('v-q').addEventListener('input', debounce(renderVal, 200));
    renderAll();
    // deep link: #val/<video_id>_q<k> opens that question (switching to All when it is not a DEV10 question)
    function openFromHash() {
      const m = decodeURIComponent(location.hash.slice(1)).match(/^val\/(.+)$/);
      if (!m || !DATA.val.questions.some((q) => q.id === m[1])) return;
      if (!DATA.val.dev.includes(m[1])) $('v-set').value = '';
      ['v-src', 'v-vsrc', 'v-mra', 'v-q'].forEach((id) => { $(id).value = ''; });
      renderAll();
      const card = [...$('val-list').querySelectorAll('details.qrow')].find((x) => x.dataset.id === m[1]);
      card.open = true; card.scrollIntoView();
    }
    openFromHash();
    addEventListener('hashchange', openFromHash);
  }).catch((e) => {
    $('val-list').innerHTML = `<p class="noscene">data/videos.json: ${esc(e.message)}</p>`;
  });
})();
