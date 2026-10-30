// videos.html: one row per question (unit <video_id>_q<k>). Data: data/videos.json (built by tools/build_videos.py).
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const SRC_LABEL = { scene: '场景重放', measurement: '独立测量', prior: '显式先验', unresolved: '未解决',
    execution_failed: '执行失败' };
  const SHAPE_LABEL = { box: '长方体', sphere: '球', cylinder: '圆柱', prism: '三棱柱', mesh: '网格（画包围盒）' };
  const PALETTE = ['#3b6fd8', '#d9622b', '#2e9b5f', '#b03fa8', '#c79a12', '#1b9aaa', '#d6455d', '#6d5bd0', '#7a8b2a', '#8a5a3c'];
  let DATA = null;

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
  function mraBadge(m) { return isFinite(m) ? `<span class="mra ${mraClass(m)}">${fmt(m, 2)}</span>` : '<span class="muted">—</span>'; }
  // ---------- summary: the listed questions (docs/DEV10.md) ----------
  function valSummary(val) {
    const all = val.questions;
    $('val-mean').textContent = fmt(mean(all.map((q) => q.mra)), 2);
    $('val-n').textContent = `${all.length} 题`;
    const row = (label, qs, strong) => {
      const s = (x) => (strong ? `<b>${x}</b>` : x);
      return `<tr><td>${s(label)}</td><td class="n">${s(qs.length)}</td><td class="n">${s(fmt(mean(qs.map((q) => q.mra)), 2))}</td></tr>`;
    };
    let h = '<tr><th>类别</th><th class="n">题数</th><th class="n">平均 MRA</th></tr>';
    CATS.forEach((c) => { h += row(c, all.filter((q) => q.cat === c)); });
    h += row('全部', all, true);
    $('val-sum').innerHTML = h;
  }

  // ---------- one question ----------
  // the artifacts of the question's processing unit (val.unit): with unit "video" the scene of its prior group and the
  // video's scene graph and tracks, with unit "question" its own
  function artifacts(q, v) {
    if (DATA.val.unit === 'question') return { scene: q.scene, sgraph: q.sgraph, tracks: q.tracks };
    if (DATA.val.unit !== 'video') throw new Error(`unknown processing unit ${DATA.val.unit}`);
    const group = q.group == null ? null : v.groups.find((g) => g.group === q.group);
    if (q.group != null && !group) throw new Error(`${q.id}: group ${q.group} not in ${v.id}`);
    return { scene: group ? group.scene : null, sgraph: v.sgraph, tracks: v.tracks };
  }
  function summaryRow(q, v) {
    const nums = `<span class="qcat"><span class="tag">${esc(q.cat)}</span></span>` +
      `<span class="qv" data-k="真值">${fmtNum(q.answer)}</span>` +
      `<span class="qv pred" data-k="预测">${fmtNum(q.pred)}</span>` +
      `<span class="qsrc">${esc(SRC_LABEL[q.src] || q.src)}</span>`;
    return `<summary><img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}"><span class="qmain"><span class="qid">${esc(q.id)}</span><span class="qtext">${esc(q.q)}</span></span>` +
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
        body.innerHTML = priorBlock(q) + mediaBlock(view, q) + overlayBlock(q) + trackBlock(a.tracks, view, q) +
          sceneGraphTree(a.sgraph) + fitBlock(q, view);
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
    return `<div class="qfull">${esc(q.q)}</div>` + (lines.length ? `<h4>prior</h4><ul class="prior">${lines.map((l) => `<li>${esc(l)}</li>`).join('')}</ul>` : '');
  }

  // ---------- question time t_q ----------
  const tqLabel = (q) => (q.tq.length ? q.tq.map((x) => fmt(x, 2)).join('–') + ' s' : q.tjump != null ? `≈${fmt(q.tjump, 2)} s` : '');
  // IoU < 0.3 or centre offset > 0.5 box sizes: the fitted object is not where the track is at t_q
  const alignBad = (a) => (a.iou != null && a.iou < 0.3) || (a.dc != null && a.dc > 0.5);
  const alignWarn = (a) => (a.iou != null && a.iou < 0.6) || (a.dc != null && a.dc > 0.25);
  function alignCell(a) {
    if (!a) return '<td class="n"><span class="muted">—</span></td>';
    const title = [a.t != null ? `t = ${fmt(a.t, 3)} s${a.frame != null ? `（第 ${a.frame} 帧）` : ''}` : '',
      a.px != null ? `中心偏差 ${fmtNum(a.px)} px` : ''].filter(Boolean).join('\n');
    return `<td class="n${alignBad(a) ? ' align-bad' : alignWarn(a) ? ' align-warn' : ''}" title="${esc(title)}">` +
      [a.iou != null ? 'IoU ' + fmt(a.iou, 2) : '', a.dc != null ? 'dc ' + fmt(a.dc, 2) : (a.px != null ? fmtNum(a.px) + ' px' : '')].filter(Boolean).join(' · ') + '</td>';
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
      const note = v.video_from !== 'val' ? `<p class="muted vnote">播放的是测试集里同一视频的 H.264 版本（验证集原文件是 MPEG-4 Part 2，浏览器放不了）。<a href="${esc(v.orig)}">下载原文件</a></p>` : '';
      vid = `<h4>原视频</h4><video class="orig" controls muted loop playsinline preload="metadata"></video>${note}`;
    } else {
      vid = `<h4>原视频</h4><div class="noscene">MPEG-4 Part 2（<code>${esc(v.codec)}</code>），浏览器放不了。<a href="${esc(v.orig)}">下载原文件</a></div>`;
    }
    const tl = tqLabel(q);
    const jump = tl ? `<button type="button" class="tq-btn" data-t="${q.tjump == null ? '' : q.tjump}">t = ${esc(tl)}</button>` : '';
    let scene = '';
    if (v.scene) {
      scene = '<div><h4>3D 场景</h4>' +
        '<div class="viewbar"><span class="seg" role="group" aria-label="视角"><button type="button" data-view="free" aria-pressed="false">自由视角</button>' +
        '<button type="button" data-view="cam" aria-pressed="true">相机视角</button></span>' +
        (v.video ? '<label class="off"><input type="checkbox" class="overlay" checked disabled> 叠加原视频</label>' : '') + jump + '</div>' +
        '<div class="viewer"><div class="msg">加载 three.js …</div></div>' +
        '<div class="timebar"><button type="button" class="play">播放</button><span class="tq-track"><input type="range" min="0" max="1000" value="0" aria-label="时间"><span class="tq-marks"></span></span><span class="tlabel">t = 0.00 s</span></div></div>';
    } else if (jump) {
      vid += `<div class="viewbar" style="margin-top:6px">${jump}</div>`;
    }
    return `<div class="media"><div>${vid}</div>${scene}</div>`;
  }

  // overlays rendered by the pipeline from the saved scene at the question's times
  function overlayBlock(q) {
    const ims = q.overlays;
    if (!ims.length) return '';
    return '<h4>叠加图</h4><div class="runtime-frames">' + ims.map((im) => im.path
      ? `<figure class="runtime-frame"><a href="${esc(im.path)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(im.path)}" alt="t = ${esc(im.t)} s"></a>` +
        `<figcaption>t = ${fmt(im.t, 2)} s</figcaption></figure>`
      : `<p class="fwarn">t = ${fmt(im.t, 2)} s：${esc(im.unavailable)}</p>`).join('') + '</div>';
  }

  // tracked intervals per object on the video's time axis; the question's times as vertical lines
  function trackBlock(tracks, v, q) {
    if (!tracks || !Object.keys(tracks).length) return '';
    const T = v.duration;
    const order = v.scene ? v.scene.objects.map((o) => o.name) : [];
    const pct = (t) => `${(100 * Math.min(Math.max(t / T, 0), 1)).toFixed(2)}%`;
    const lines = q.tq.filter((t) => t <= T + 1e-6).map((t) => `<span class="trk-tq" style="left:${pct(t)}"></span>`).join('');
    const rows = Object.entries(tracks).map(([name, ivs]) => {
      const i = order.indexOf(name), col = i >= 0 ? PALETTE[i % PALETTE.length] : 'var(--muted)';
      const segs = (ivs || []).map(([a, b]) => `<span class="trk-seg" style="left:${pct(a)};width:calc(${pct(b)} - ${pct(a)} + 2px);background:${col}" title="${fmt(a, 2)}–${fmt(b, 2)} s"></span>`).join('');
      return `<div class="trk"><span class="trk-n" title="${esc(name)}">${esc(name)}</span><span class="trk-bar">${segs}${lines}</span></div>`;
    }).join('');
    return `<h4>跟踪</h4><div class="trk-wrap">${rows}<div class="trk"><span class="trk-n"></span><span class="trk-axis"><span>0</span><span>${fmt(T, 2)} s</span></span></div></div>`;
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
      playing = !playing; play.textContent = playing ? '暂停' : '播放';
      if (playing) { t0 = performance.now() - (slider.value / 1000) * T * 1000; raf = requestAnimationFrame(tick); } else cancelAnimationFrame(raf);
    };
    if (video) {
      const o = { signal };
      video.addEventListener('play', () => { play.textContent = '暂停'; cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); }, o);
      video.addEventListener('pause', () => { play.textContent = '播放'; cancelAnimationFrame(raf); if (linked()) fromVideo(); }, o);
      video.addEventListener('seeked', () => { if (linked() && video.paused) fromVideo(); }, o);
    }
    return {
      now: () => (slider.value / 1000) * T,
      redraw: () => show((slider.value / 1000) * T),
      stop() { playing = false; cancelAnimationFrame(raf); slider.oninput = null; play.onclick = null; play.textContent = '播放'; },
    };
  }

  // ---------- scene graph (tools/build_videos.sgraph) drawn as a layered tree ----------
  // one box per object / support / hand, each under its first parent in time; later parents are extra edges (one that
  // skips a level runs through a slot beside the parent's child, where its label sits). Objects that are only ever
  // airborne hang under a 空中 root; an airborne span (or a relation without a target) of an object that also rests on
  // something is a short stub to the left of its box.
  let sgCtx = null;
  const sgTextW = (s, px, bold) => {
    if (!sgCtx) sgCtx = document.createElement('canvas').getContext('2d');
    sgCtx.font = `${bold ? '600 ' : ''}${px}px system-ui, sans-serif`;
    return sgCtx.measureText(s).width;
  };
  function sceneGraphTree(sg) {
    if (!sg || !sg.nodes || !sg.nodes.length) return '';
    const AIR = '\u0000air', NF = 12.5, LF = 11, LL = 13, PADX = 10, GAPX = 16, M = 6, MAXW = window.innerWidth < 640 ? 96 : 176;
    const type = {}, order = [];
    const addNode = (id, t) => { if (!(id in type)) { type[id] = t || 'object'; order.push(id); } };
    sg.nodes.forEach((n) => addNode(n.id, n.type));
    // relations per object: one entry per (target, relation), time spans merged
    const rels = new Map();
    (sg.edges || []).forEach((e) => {
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
      return `${r.rel} ${r.spans.sort((a, b) => a[0] - b[0]).map(([a, b]) => `${fmt(a, 2)}–${b == null ? '' : fmt(b, 2)}`).join(', ')} s`;
    };
    // first parent in time; an object with nothing to rest on but an airborne span goes under the 空中 root
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
      const name = id === AIR ? '空中' : id, words = String(name).split(/\s+/), lines = [];
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
    // rows: height of the tallest box; the gap above a row fits its edge labels
    const all = Object.keys(box), nd = Math.max(...all.map((id) => depth[id])) + 1;
    const rowH = Array(nd).fill(0), gap = Array(nd).fill(0);
    all.forEach((id) => {
      const b = box[id], d = depth[id];
      rowH[d] = Math.max(rowH[d], b.h, b.st ? b.st.length * LL + 4 : 0);
      if (d) gap[d] = Math.max(gap[d], b.tl && b.tl.length ? 42 + b.tl.length * LL : b.slot && b.t ? 24 : 26);
    });
    const rowTop = [];
    rowH.forEach((h, d) => { rowTop[d] = d ? rowTop[d - 1] + rowH[d - 1] + gap[d] : M; });
    const H = Math.ceil(rowTop[nd - 1] + rowH[nd - 1] + M);
    const top = (id) => rowTop[depth[id]] + (rowH[depth[id]] - box[id].h) / 2, bot = (id) => top(id) + box[id].h;
    const curve = (x1, y1, x2, y2) => { const m = (y1 + y2) / 2; return `<path class="sge" d="M${x1},${y1} C${x1},${m} ${x2},${m} ${x2},${y2}"></path>`; };
    // org-chart connector: down from the parent to the row's bus line, across, down into the child; rounded corners
    const busY = (d) => rowTop[d] + rowH[d] + 12;
    const elbow = (x1, y1, by, x2, y2) => {
      const dx = x2 - x1, r = Math.min(6, Math.abs(dx) / 2), s = Math.sign(dx);
      if (Math.abs(dx) < 1) return `<path class="sge" d="M${x1},${y1} V${y2}"></path>`;
      return `<path class="sge" d="M${x1},${y1} V${by} H${x2 - s * r} Q${x2},${by} ${x2},${by + r} V${y2}"></path>`;
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
      const by = busY(depth[p]);
      edges += elbow(X[p], bot(p), by, X[id], top(id));
      if (box[id].tl.length) { const n = box[id].tl.length; labels += label(X[id], (by + top(id)) / 2 + (n * LL + 3) / 2, box[id].tl); }
    });
    extra.forEach((x) => {
      const c = x.from, t = relText(x.r);
      if (!(x.to in box)) return;
      if (x.slot) {
        const s = x.slot, sx = X[s], sy = rowTop[depth[s]] + rowH[depth[s]] / 2, cr = X[c] + box[c].w / 2, cm = (top(c) + bot(c)) / 2;
        const blocked = ids.some((o) => o !== c && depth[o] === depth[c] && X[o] > X[c] && X[o] < sx);
        edges += elbow(X[x.to], bot(x.to), busY(depth[x.to]), sx, sy);
        edges += blocked ? curve(sx, sy, X[c] + box[c].w / 2 - 8, top(c))
          : `<path class="sge" d="M${sx},${sy} V${cm - 6} Q${sx},${cm} ${sx - 6},${cm} H${cr}"></path>`;
        if (t) labels += label(sx, sy + 8, [t]);
        return;
      }
      const up = depth[x.to] < depth[c], x1 = X[c], y1 = up ? top(c) : bot(c), x2 = X[x.to], y2 = up ? bot(x.to) : top(x.to);
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
    return `<h4>scene graph</h4><div class="sgwrap"><svg class="sg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="scene graph">${edges}${labels}${boxes}</svg></div>`;
  }

  // ---------- reconstruction: the question's own scene value, the fitted objects ----------
  // one dimension with its source (scene_fit dim_source): observed = plain, prior = tagged, bound = grey (not seen)
  function dimCell(o, key, prefix = '') {
    const d = o.dims || {}, src = (o.dim_source || {})[key];
    const v = prefix + fmtNum(d[key]);
    if (src === 'bound') return `<span class="dim-bound" title="框和深度都没看到这一维，取防塌缩下限">${v}<sup>看不见</sup></span>`;
    if (src === 'prior') return `<span>${v}<sup class="dim-prior">先验</sup></span>`;
    return v;
  }
  function dimsText(o) {
    if (o.shape === 'sphere') return [dimCell(o, 'D', '直径 '), '', ''];
    if (o.shape === 'cylinder') return [dimCell(o, 'D', '直径 '), '', dimCell(o, 'H')];
    if (o.shape === 'rope') return [dimCell(o, 'L', '绳长 '), '', ''];
    return [dimCell(o, 'L'), dimCell(o, 'W'), dimCell(o, 'H')];
  }
  function supportText(o) {
    const sp = o.support || {};
    let t = sp.mode === 'support' ? `on ${sp.target || '主支撑面'}` : sp.mode === 'airborne' ? (sp.airborne_part && sp.target ? `先 on ${sp.target}，后 airborne` : 'airborne') : sp.mode || '';
    if (o.hang) t += '（顶面即支撑面）';
    if (o.t_on > 0) t += `；${fmt(o.t_on, 2)} s 前静止`;
    const MODE = { ballistic: '弹跳模型', contact: '接触（几何证据优先）', overhead: '顶部结构', free3d: '自由 3D 运动', own_plane: '自有平面' };
    if (o.mode && MODE[o.mode]) t += `；${MODE[o.mode]}`;
    if (o.events && o.events.length) t += `（${o.events.slice(0, 4).map((e) => `${e[1] === 'bounce' ? '弹' : e[1] === 'stop' ? '停' : e[1]} ${fmt(e[0], 2)} s`).join('、')}${o.events.length > 4 ? '…' : ''}）`;
    if (o.out_of_frame && o.out_of_frame.length) t += `；出画 ${o.out_of_frame.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ')} s`;
    return t;
  }
  // conf-lowering signals of one scene answer (quality fields written by scene_fit)
  function confSignals(q) {
    if (!q) return [];
    const out = [];
    if (q.own_plane) out.push('自有平面');
    if (q.support_uncertain) out.push('支撑不确定');
    if (q.support_consistency != null && q.support_consistency < 1) out.push(`支撑一致 ${fmt(q.support_consistency, 2)}`);
    if (q.dim_source === 'bound') out.push('看不见（bound）');
    if (q.collapsed) out.push('塌缩');
    if (q.structure) out.push('结构');
    if (q.t_cover === false) out.push('t_q 不在可见区间');
    return out;
  }
  // focal-length votes of the fit (scene json "focal_votes")
  const F_SRC = { gravity: '重力抛物线', lab_prior: '实验室先验', size_prior_depth: '尺寸先验 + 深度', typical_votes: '典型尺寸', typical_height: '典型高度', prior_kinematic: '运动学先验', person: '人身高', votes_first: '投票优先' };
  function focalVotes(s) {
    if (!s.focal_votes || !s.focal_votes.length) return '';
    const tw = s.focal_votes.reduce((a, x) => a + (x.weight || 0), 0);
    return '<details class="sfres"><summary>焦距投票</summary><div class="scroll-x"><table><tr><th>来源</th><th class="n">f (px)</th><th class="n">权重</th></tr>' +
      s.focal_votes.map((x) => `<tr><td>${esc(F_SRC[x.source] || x.source)}</td><td class="n">${fmtNum(x.value)}</td><td class="n">${fmtNum(x.weight)}${tw ? ` <span class="muted">(${fmt((100 * x.weight) / tw, 0)}%)</span>` : ''}</td></tr>`).join('') +
      '</table></div></details>';
  }
  function fitBlock(q, v) {
    const f = q.fit, s = v.scene;
    let h = '<h4>重建</h4>';
    if (q.binding_error) h += `<p class="fwarn">${esc(q.binding_error)}</p>`;
    if (f || q.method || q.status) {
      const sig = confSignals(f && f.quality);
      h += '<div class="scroll-x"><table class="fit"><tr><th>物体</th><th>方法</th><th class="n">场景值</th>' +
        '<th class="n">置信度</th><th class="n">t_q 对齐</th><th>状态</th></tr>' +
        `<tr><td>${f ? esc(f.obj) : '—'}</td><td><code>${esc(q.method)}</code>${sig.length ? '<br>' + sig.map((x) => `<span class="sigtag">${esc(x)}</span>`).join(' ') : ''}</td>` +
        `<td class="n">${fmtNum(f && f.pred)}</td>` +
        `<td class="n">${fmt(q.conf, 3)}</td>${alignCell(q.align)}<td><code>${esc(q.status)}</code></td></tr></table></div>`;
    }
    if (s) {
      const hasSup = s.objects.some((o) => o.support), hasVis = s.objects.some((o) => o.visible);
      const rows = s.objects.map((o, i) => {
        const [L, W, H] = dimsText(o);
        return `<tr><td><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span></td><td>${esc(o.name)}</td><td>${SHAPE_LABEL[o.shape] || esc(o.shape)}</td>` +
          `<td class="n">${L}</td><td class="n">${W}</td><td class="n">${H}</td>` + (hasSup ? `<td>${esc(supportText(o))}</td>` : '') +
          (hasVis ? `<td>${o.visible ? o.visible.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ') + ' s' : '<span class="muted">—</span>'}</td>` : '') +
          `<td class="n">${fmt(o.rms, 1)}</td></tr>`;
      }).join('');
      h += `<div class="scroll-x"><table class="objs"><tr><th></th><th>物体</th><th>形状</th><th class="n">L (m)</th><th class="n">W (m)</th><th class="n">H (m)</th>` +
        (hasSup ? '<th>支撑</th>' : '') + (hasVis ? '<th>可见</th>' : '') + `<th class="n">残差 (px)</th></tr>${rows}</table></div>` +
        `<p class="facts">f = ${fmtNum(s.f)} px${s.f_source ? ` · ${esc(s.f_source)}` : ''}` +
        (s.main_support ? ` · 主支撑面 ${esc(s.main_support)}` : '') + '</p>' + focalVotes(s);
    }
    if (q.measurement || q.provenance || q.quality) {
      h += `<details class="sfres"><summary>诊断</summary><pre>${esc(JSON.stringify({
        scene_id: q.scene_id, row: q.row, method: q.method, source: q.src, status: q.status,
        execution_status: q.execution_status,
        measurement: q.measurement, measurement_id: q.measurement_id, measurement_record: q.measurement_record,
        quality: q.quality, provenance: q.provenance }, null, 2))}</pre></details>`;
    }
    return h;
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
      box.querySelector('.msg').textContent = 'three.js 加载失败（需要能访问 cdn.jsdelivr.net）';
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
      const u = Math.max(t - (o.t_on || 0), 0); return add(add(o.p0, o.v0, u), o.a, 0.5 * u * u);
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
      const d = o.dims || {};
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
    counter.title = '视频可见 = 当前帧跟踪有框的物体数；场景视野内 = 拟合位置投影在原相机画面内（且在相机前方）的物体数';
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
      counter.textContent = `视频可见 ${nVis} / 场景视野内 ${nIn}`;
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
        box.innerHTML = '<div class="msg">已收起</div>';
      },
    };
  }

  // ---------- filtering ----------
  function renderList(items) {
    const list = $('val-list'); list.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'qhead';
    head.innerHTML = '<span></span><span>题目</span><span class="qnums"><span>类别</span><span class="qv">真值</span><span class="qv">预测</span><span>来源</span></span><span class="qm">MRA</span>';
    if (items.length) list.appendChild(head);
    const frag = document.createDocumentFragment();
    items.forEach(({ q, v }) => frag.appendChild(questionCard(q, v)));
    list.appendChild(frag);
  }
  function renderVal() {
    const cat = $('v-cat').value, sort = $('v-sort').value, vids = DATA.val.vmap;
    const items = DATA.val.questions.filter((q) => !cat || q.cat === cat).map((q) => ({ q, v: vids[q.video] }));
    const nanLast = (f) => (a, b) => (isNaN(a.q.mra) ? 1 : 0) - (isNaN(b.q.mra) ? 1 : 0) || f(a, b);
    if (sort === 'mra-asc') items.sort(nanLast((a, b) => a.q.mra - b.q.mra));
    else if (sort === 'mra-desc') items.sort(nanLast((a, b) => b.q.mra - a.q.mra));
    renderList(items);
  }

  fetch('data/videos.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then((d) => {
    DATA = d;
    d.val.vmap = Object.fromEntries(d.val.videos.map((v) => [v.id, v]));
    $('gen').textContent = `${d.val.run} · ${d.generated}`;
    valSummary(d.val);
    ['v-cat', 'v-sort'].forEach((id) => $(id).addEventListener('change', renderVal));
    renderVal();
    // deep link: #val/<video_id>_q<k> opens that question
    function openFromHash() {
      const m = decodeURIComponent(location.hash.slice(1)).match(/^val\/(.+)$/);
      if (!m || !DATA.val.questions.some((q) => q.id === m[1])) return;
      $('v-cat').value = '';
      renderVal();
      const card = [...$('val-list').querySelectorAll('details.qrow')].find((x) => x.dataset.id === m[1]);
      card.open = true; card.scrollIntoView();
    }
    openFromHash();
    addEventListener('hashchange', openFromHash);
  }).catch((e) => {
    $('val-list').innerHTML = `<p class="noscene">data/videos.json：${esc(e.message)}</p>`;
  });
})();
