// videos.html: per-video question browser. Data: data/videos.json (built by tools/build_videos.py).
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const SRC_LABEL = { geo: '几何', vlm: 'Qwen', s2d: '2D 场景值', fit: '拟合值' };
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
  // macro = mean over the four categories of the per-category mean (same as pipeline/metrics.evaluate)
  function macro(qs) {
    const per = {};
    CATS.forEach((c) => { per[c] = mean(qs.filter((q) => q.cat === c).map((q) => q.mra)); });
    const vals = CATS.map((c) => per[c]).filter((v) => isFinite(v));
    return { per, macro: mean(vals), micro: mean(qs.map((q) => q.mra)), n: qs.length };
  }

  // ---------- tabs ----------
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    $('pane-val').hidden = b.dataset.tab !== 'val';
    $('pane-test').hidden = b.dataset.tab !== 'test';
    try { localStorage.setItem('qp-videos-tab', b.dataset.tab); } catch (e) { /* storage unavailable */ }
  }));

  // ---------- summary tables ----------
  function valSummary(val) {
    const all = val.videos.flatMap((v) => v.questions);
    const m = macro(all);
    $('val-macro').textContent = fmt(m.macro, 4);
    $('val-src-file').innerHTML = `${m.n} 题、${val.videos.length} 个视频，来自 <code>${esc(val.file)}</code>；micro（逐题平均）${fmt(m.micro, 3)}。3D 视图为场景拟合 v9b（c_v9b），2D 图层为 scene2d v8（s2d_v8）`;
    // per answer source: how many questions took their final prediction from it and their MRA
    const SRCS = [['geo', '几何'], ['fit', '拟合'], ['s2d', '2D 场景'], ['vlm', 'Qwen']].filter(([k]) => all.some((q) => q.src === k));
    let h = '<tr><th>类别</th><th class="n">题数</th><th class="n">MRA</th>' + SRCS.map(([, l]) => `<th class="n">${l} 题数</th><th class="n">${l} MRA</th>`).join('') + '</tr>';
    const row = (label, qs, strong) => {
      const s = (x) => (strong ? `<b>${x}</b>` : x);
      return `<tr><td>${s(label)}</td><td class="n">${qs.length}</td><td class="n">${s(fmt(mean(qs.map((q) => q.mra))))}</td>` +
        SRCS.map(([k]) => { const g = qs.filter((q) => q.src === k); return `<td class="n">${g.length}</td><td class="n">${g.length ? fmt(mean(g.map((q) => q.mra))) : '—'}</td>`; }).join('') + '</tr>';
    };
    CATS.forEach((c) => { h += row(c, all.filter((q) => q.cat === c)); });
    h += `<tr><td><b>macro</b></td><td class="n">${all.length}</td><td class="n"><b>${fmt(m.macro, 4)}</b></td><td colspan="${2 * SRCS.length}" class="muted">四类 MRA 的平均</td></tr>`;
    $('val-sum').innerHTML = h;
  }
  function testSummary(test) {
    const all = test.videos.flatMap((v) => v.questions);
    const v3 = test.videos.filter((v) => v.dim === 3);
    const fit = v3.filter(hasScene).length;
    $('test-src-file').innerHTML = `${all.length} 题、${test.videos.length} 个视频，预测来自 submission_v7（<code>${esc(test.file)}</code>，按行号和 <code>load_split('test')</code> 对齐）；v8 运行中（commit e4b00b4）。测试集没有答案，不显示准确度。3D 视频 ${v3.length} 个，其中 ${fit} 个有场景拟合结果。`;
    let h = '<tr><th>类别</th>' + CATS.map((c) => `<th class="n">${c}</th>`).join('') + '<th class="n">合计</th></tr>';
    h += '<tr><td>题数</td>' + CATS.map((c) => `<td class="n">${all.filter((q) => q.cat === c).length}</td>`).join('') + `<td class="n">${all.length}</td></tr>`;
    h += '<tr><td>视频数</td>' + CATS.map((c) => `<td class="n">${test.videos.filter((v) => v.cats.includes(c)).length}</td>`).join('') + `<td class="n">${test.videos.length}</td></tr>`;
    $('test-sum').innerHTML = h;
  }

  // ---------- list rendering ----------
  function fillSourceSelect(sel, videos) {
    [...new Set(videos.map((v) => v.source))].sort().forEach((s) => { const o = document.createElement('option'); o.textContent = s; sel.appendChild(o); });
  }
  function matchText(v, q, needle) { return !needle || v.id.toLowerCase().includes(needle) || q.q.toLowerCase().includes(needle); }

  const hasScene = (v) => !!(v.scene || v.has_scene);
  function videoCard(v, qs, split) {
    const d = document.createElement('details');
    d.className = 'vid';
    d.dataset.id = v.id;
    const m = split === 'val' ? mean(qs.map((q) => q.mra)) : NaN;
    const sceneTag = v.dim === 3 ? (hasScene(v) ? '<span class="tag">场景拟合</span>' : '<span class="tag" style="opacity:.6">无场景拟合</span>')
      : (v.scene2d || v.has_scene2d ? '<span class="tag">图层场景</span>' : '');
    const thumb = v.thumb ? `<img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}">` : '<span class="thumb"></span>';
    d.innerHTML = `<summary>${thumb}<div class="vmeta"><div class="vid-id">${esc(v.id)}</div>` +
      `<div class="sub">${esc(v.source)} · ${esc(v.video_type)} · ${v.fps} fps${v.duration ? ' · ' + fmt(v.duration, 1) + ' s' : ''} · ${qs.length} 题 ` +
      v.cats.map((c) => `<span class="tag">${c}</span>`).join('') + sceneTag + '</div></div>' +
      '<div class="vright"><span class="vright-row">' +
      (v.eval && v.eval.layer && v.eval.layer !== '—' ? `<span class="evtag" title="评估员判断的主因层：${esc(v.eval.layer)}\n根因：${esc(v.eval.cause)}">${esc(shortLayer(v.eval.layer))}</span>` : '') +
      (split === 'val' ? `<span title="该视频（当前筛选下）的平均 MRA">${mraBadge(m)}</span>` : '') + '</span>' +
      (v.run_at ? `<span class="runat" title="最近一次运行这个视频的时间（洛杉矶时间）">${esc(v.run_at)}</span>` : '') + '</div></summary>';
    let built = false, viewer = null, video = null;
    const onToggle = () => {
      // test-set scenes live in one file per video (data/scenes/test/<id>.json) and are fetched when the card opens
      if (d.open && v.scene_ref && !v._loaded) {
        fetch(v.scene_ref).then((r) => (r.ok ? r.json() : {})).catch(() => ({})).then((j) => {
          v.scene = j.scene || null; v.scene2d = j.scene2d || null; v._loaded = true;
          if (d.open) onToggle();
        });
        return;
      }
      if (d.open && !built) {
        built = true;
        const body = document.createElement('div');
        body.className = 'vbody';
        body.innerHTML = mediaBlock(v, split) + evalBlock(v) + questionTable(qs, split, v.dim === 2 && !!v.scene2d) +
          (v.dim === 3 ? sceneTables(v, split) : v.scene2d ? layerTables(v) : '');
        d.appendChild(body);
        video = body.querySelector('video');
        tqMarks(body, v, qs);
        body.addEventListener('click', (e) => {
          const b = e.target.closest('.tq-btn');
          if (b) jumpTo(body, v, b.dataset.t === '' ? null : Number(b.dataset.t));
        });
      }
      // the <video> only holds a src while its card is open, so collapsed cards do not keep connections or buffers
      if (video) {
        // preload="auto" + a tiny seek decodes the first frame right away, so the layer / overlay canvases
        // can draw it before the user presses play (with metadata-only preload the canvas stays black)
        if (d.open && !video.getAttribute('src')) {
          video.preload = 'auto'; video.src = v.video;
          video.addEventListener('loadedmetadata', () => { if (video.readyState < 2) video.currentTime = 0.001; }, { once: true });
        }
        if (!d.open) { video.pause(); video.removeAttribute('src'); video.load(); }
      }
      if (v.dim === 3 && v.scene) {
        if (d.open && !viewer) viewer = startViewer(d.querySelector('.viewer'), d.querySelector('.timebar'), v, video);
        if (!d.open && viewer) { viewer.then((x) => x && x.dispose()); viewer = null; }
      }
      if (v.dim === 2 && v.scene2d) {
        if (d.open && !viewer) viewer = Promise.resolve(startLayers(d.querySelector('.viewer'), d.querySelector('.timebar'), v, video));
        if (!d.open && viewer) { viewer.then((x) => x && x.dispose()); viewer = null; }
      }
    };
    d.addEventListener('toggle', onToggle);
    return d;
  }

  // list tag: the main layer without parentheticals and trailing clauses ("尺度（…）+ 拟合（…）；q1 …" -> "尺度 + 拟合")
  function shortLayer(l) { return String(l).replace(/（[^）]*）|\([^)]*\)/g, '').split(/[；;]/)[0].replace(/\s*\+\s*/g, ' + ').trim(); }
  // evaluator verdict: the per-video report (outputs/eval/<video>.md -> data/eval/<video>.html) wins, the overview row is the fallback
  function evalBlock(v) {
    const e = v.eval;
    if (!e) return '';
    const prog = e.progress && e.progress !== '—' ? `<span class="muted">修复进行中：${esc(e.progress)}</span>` : '';
    const rep = e.report ? `<div class="ev-rep"><b>逐视频报告${e.report.title ? `（${esc(e.report.title)}）` : ''}：</b>${esc(e.report.excerpt)}${e.report.excerpt.length >= 400 ? '…' : ''} <a href="${esc(e.report.file)}" target="_blank" rel="noopener">全文</a></div>` : '';
    return `<div class="evbox"><h4>评估</h4><div><b>主因层：</b>${esc(e.layer || '—')}　${prog}</div>` +
      (e.cause && e.cause !== '—' ? `<div><b>根因：</b>${esc(e.cause)}</div>` : '') + rep +
      `<div class="muted" style="font-size:12px">` + (e.from_report ? `主因层和根因取自逐视频报告；概览 <code>${esc(e.source || '')}</code> 兜底` : `来源 <code>${esc(e.source || '')}</code>`) +
      `（评估员的视频平均 MRA ${esc(e.mra || '')}）</div></div>`;
  }

  // ---------- question time t_q: jump button, alignment check, ticks on the time axis ----------
  const hasAlign = (qs) => qs.some((q) => q.align);
  function tqHead(qs) {
    return '<th title="把原视频和 3D / 2D 视图跳到题目问的时刻 t_q（没写时刻的题跳到被问物体可见区间的中点），并切到相机视角、叠加原视频">问到的帧</th>' +
      (hasAlign(qs) ? '<th class="n" title="t_q 时刻拟合的物体和跟踪框的对齐：IoU / dc（中心偏差 ÷ 框尺寸）；红：IoU &lt; 0.3 或 dc &gt; 0.5，橙：IoU &lt; 0.6 或 dc &gt; 0.25">t_q 对齐</th>' : '');
  }
  // IoU < 0.3 or centre offset > 0.5 box sizes: the fitted object is not where the track is at t_q
  const alignBad = (a) => (a.iou != null && a.iou < 0.3) || (a.dc != null && a.dc > 0.5);
  const alignWarn = (a) => (a.iou != null && a.iou < 0.6) || (a.dc != null && a.dc > 0.25);
  const alignTitle = (a) => [a.t != null ? `检查帧 t = ${fmt(a.t, 3)} s${a.frame != null ? `（第 ${a.frame} 帧）` : ''}` : '',
    a.iou != null ? `IoU = 拟合框与跟踪框的交并比 ${fmt(a.iou, 3)}` : '', a.dc != null ? `dc = 中心偏差 ÷ 框尺寸 ${fmt(a.dc, 3)}` : '',
    a.px != null ? `中心偏差 ${fmtNum(a.px)} px` : '', '红：IoU < 0.3 或 dc > 0.5；橙：IoU < 0.6 或 dc > 0.25'].filter(Boolean).join('\n');
  function tqCells(qs, q) {
    const t = q.tjump;
    const lab = q.tq && q.tq.length ? q.tq.map((x) => fmt(x, 2)).join('–') + ' s' : (t != null ? `≈${fmt(t, 2)} s` : '中点');
    let h = `<td><button type="button" class="tq-btn" data-t="${t == null ? '' : t}" title="${q.tq && q.tq.length ? '题目问的时刻' : '题目没写时刻：跳到被问物体可见区间的中点'}">${lab}</button></td>`;
    if (hasAlign(qs)) {
      const a = q.align;
      h += !a ? '<td class="n"><span class="muted">—</span></td>'
        : `<td class="n${alignBad(a) ? ' align-bad' : alignWarn(a) ? ' align-warn' : ''}" title="${esc(alignTitle(a))}">` +
          [a.iou != null ? 'IoU ' + fmt(a.iou, 2) : '', a.dc != null ? 'dc ' + fmt(a.dc, 2) : (a.px != null ? fmtNum(a.px) + ' px' : '')].filter(Boolean).join(' · ') + '</td>';
    }
    return h;
  }
  // scene time span of a card's viewer (the 3D view uses the video duration, the 2D view n_frames / fps)
  function sceneT(v) {
    const s = v.scene2d;
    if (v.dim === 2 && s && s.n_frames && s.fps) return s.n_frames / s.fps;
    return v.duration || 2;
  }
  function tqMarks(body, v, qs) {
    const marks = body.querySelector('.tq-marks');
    if (!marks) return;
    const T = sceneT(v);
    qs.forEach((q, i) => (q.tq || []).forEach((t) => {
      if (t > T + 1e-6) return;
      const m = document.createElement('span');
      m.className = 'tq-mark'; m.style.left = `${(100 * Math.min(Math.max(t / T, 0), 1)).toFixed(2)}%`;
      m.title = `第 ${i + 1} 题 @ ${fmt(t, 2)} s：${q.q}`;
      m.onclick = () => jumpTo(body, v, t);
      marks.appendChild(m);
    }));
  }
  function jumpTo(body, v, t) {
    const T = sceneT(v);
    if (t == null || !isFinite(t)) t = T / 2;
    const cam = body.querySelector('button[data-view="cam"]');
    if (cam && cam.getAttribute('aria-pressed') !== 'true') cam.click();
    const ov = body.querySelector('input.overlay');
    if (ov && !ov.disabled && !ov.checked) { ov.checked = true; ov.dispatchEvent(new Event('change')); }
    const bgv = body.querySelector('button[data-bg="video"]');
    if (bgv && !bgv.disabled && bgv.getAttribute('aria-pressed') !== 'true') bgv.click();
    const video = body.querySelector('video.orig');
    if (video && !video.paused) video.pause();
    const slider = body.querySelector('.timebar input');
    if (slider) { slider.value = String(Math.round(1000 * Math.min(Math.max(t / T, 0), 1))); slider.dispatchEvent(new Event('input')); }
    const media = body.querySelector('.media');
    if (media) media.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function questionTable(qs, split, s2d) {
    const kn = (q) => (q.prior ? `已知：${esc(q.prior)}` : '') + (q.depth ? `\ndepth_info：${esc(q.depth)}` : '');
    // when every question of the video has the same prior / depth_info, show it once above the table
    const shared = qs.every((q) => q.prior === qs[0].prior && q.depth === qs[0].depth);
    const known = (q) => (shared ? '' : kn(q));
    let h = '<h4>问题</h4>' + (shared && kn(qs[0]) ? `<div class="known-shared">${kn(qs[0])}</div>` : '') +
      '<div class="scroll-x"><table class="qtable"><tr><th>类别</th><th>' + (shared ? '问题' : '问题与已知条件') + '</th>';
    if (split === 'val') {
      const hasFit = qs.some((q) => q.fit != null);
      const confT = (c) => (c != null ? `置信度 ${fmt(c, 2)}` : '');
      h += '<th class="n">答案</th><th class="n" title="着色的那一列就是最终预测（几何值蓝 / 拟合值紫 / 2D 场景值绿 / Qwen 值橙）">几何值</th>' +
        (hasFit ? '<th class="n" title="3D 场景拟合（scene_fit）的值；S3 题拟合置信度 ≥ 0.6 时作为最终预测">拟合值</th>' : '') +
        (s2d ? '<th class="n" title="图层场景（pipeline.scene2d）的值，作为第二个提示给 Qwen；置信度 ≥ 0.85 时直接作为最终预测">2D 场景值</th>' : '') +
        '<th class="n">Qwen 值</th><th class="n">MRA</th>' + tqHead(qs) + '</tr>';
      qs.forEach((q) => {
        h += `<tr><td><span class="tag">${q.cat}</span><span class="muted">${esc(q.type)}</span></td>` +
          `<td class="q">${esc(q.q)}<div class="known">${known(q)}</div></td>` +
          // the column the final prediction was taken from is tinted (geo / s2d / vlm), so the route is visible at a glance
          `<td class="n">${fmtNum(q.answer)}</td>` +
          `<td class="n${q.src === 'geo' ? ' picked src-geo' : ''}" title="${q.src === 'geo' ? '最终预测取自几何值' : ''}">${fmtNum(q.geo)}</td>` +
          (hasFit ? `<td class="n${q.src === 'fit' ? ' picked src-fit' : ''}" title="${[q.src === 'fit' ? '最终预测取自拟合值' : '', confT(q.fit_conf), q.unobs ? '这一维在侧拍画面里看不见（unobservable），路由对它用单独的门槛 --fit-conf-unobservable 0.3' : ''].filter(Boolean).join('；')}">${fmtNum(q.fit)}</td>` : '') +
          (s2d ? `<td class="n${q.src === 's2d' ? ' picked src-s2d' : ''}" title="${[q.src === 's2d' ? '最终预测取自 2D 场景值' : '', confT(q.s2d_conf)].filter(Boolean).join('；')}">${fmtNum(q.s2d)}</td>` : '') +
          `<td class="n${q.src === 'vlm' ? ' picked src-vlm' : ''}" title="${q.src === 'vlm' ? '最终预测取自 Qwen 值' : ''}">${fmtNum(q.vlm)}</td><td class="n">${mraBadge(q.mra)}</td>` + tqCells(qs, q) + '</tr>';
      });
    } else {
      h += '<th class="n">行号</th><th class="n">预测值</th>' + tqHead(qs) + '</tr>';
      qs.forEach((q) => {
        h += `<tr><td><span class="tag">${q.cat}</span><span class="muted">${esc(q.type)}</span></td>` +
          `<td class="q">${esc(q.q)}<div class="known">${known(q)}</div></td><td class="n">${q.row}</td><td class="n pred">${fmtNum(q.pred)}</td>` + tqCells(qs, q) + '</tr>';
      });
    }
    return h + '</table></div>';
  }

  // one dimension with its source (scene_fit dim_source): observed = plain, prior = tagged, bound = grey (not seen)
  function dimCell(o, key, prefix = '') {
    const d = o.dims || {}, src = (o.dim_source || {})[key];
    const v = prefix + fmtNum(d[key]);
    if (src === 'bound') return `<span class="dim-bound" title="看不见、靠先验：框和深度都没看到这一维，取防塌缩下限">${v}<sup>看不见</sup></span>`;
    if (src === 'prior') return `<span title="这一维来自尺寸先验">${v}<sup class="dim-prior">先验</sup></span>`;
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

  // top of an opened card: original video (left) and, for 3D videos with a scene fit, the 3D view (right)
  function mediaBlock(v, split) {
    let vid;
    if (v.video) {
      const note = v.video_from !== split ? `<p class="muted vnote">验证集的原文件是 MPEG-4 Part 2 编码，浏览器放不了；这里放的是测试集里同一视频的 H.264 版本（帧数相同，分辨率更高）。<a href="${esc(v.orig)}">下载验证集原文件</a></p>` : '';
      vid = `<h4>原视频</h4><video class="orig" controls muted loop playsinline preload="metadata"></video>${note}`;
    } else {
      vid = `<h4>原视频</h4><div class="noscene">这个视频的编码是 MPEG-4 Part 2（<code>${esc(v.codec)}</code>），浏览器放不了，也没有可替代的 H.264 版本。<a href="${esc(v.orig)}">下载原文件</a></div>`;
    }
    let scene = '';
    if (v.dim === 3 && v.scene) {
      scene = `<div><h4>3D 场景（<code>${esc(v.scene.file)}</code>）</h4>` +
        '<div class="viewbar"><span class="seg" role="group" aria-label="视角"><button type="button" data-view="free" aria-pressed="false">自由视角</button>' +
        '<button type="button" data-view="cam" aria-pressed="true" title="从原视频相机的位置和视野看，可以和左边的视频逐帧对比">相机视角</button></span>' +
        (v.video ? '<label class="off" title="相机视角下把原视频画面垫在渲染下面"><input type="checkbox" class="overlay" checked disabled> 叠加原视频</label>' : '') +
        (v.scene.f ? '' : '<span class="fwarn" title="场景拟合没有给出焦距，用图像宽度代替">焦距为默认值（f = 图像宽度）</span>') + '</div>' +
        '<div class="viewer"><div class="msg">加载 three.js …</div></div>' +
        '<div class="timebar"><button type="button" class="play">播放</button><span class="tq-track"><input type="range" min="0" max="1000" value="0" aria-label="时间"><span class="tq-marks"></span></span><span class="tlabel">t = 0.00 s</span></div>' +
        '<p class="muted" style="font-size:12.5px;margin:6px 0 0">灰色网格 = 拟合出的地面；黑色线框 = 相机（原点）与视野；彩色 = 物体及其匀加速轨迹（虚线）。物体只在视频里跟踪到它的时段画出：相机视角下其余时段不画，自由视角下画成淡色轮廓。' +
        '左上角计数：视频可见 = 当前帧跟踪有框的物体数，场景视野内 = 拟合位置投影在原相机画面内的物体数，两者不等时标红。自由视角下拖动旋转，滚轮缩放；相机视角固定在原视频相机处，画面比例和原视频相同。' +
        (v.video ? '时间轴和左边的视频联动。' : '') + '</p></div>';
    }
    if (v.dim === 2 && v.scene2d) {
      const s = v.scene2d;
      scene = `<div><h4>图层场景（<code>${esc(s.file)}</code>）</h4>` +
        '<div class="viewbar"><span class="seg" role="group" aria-label="背景">' +
        `<button type="button" data-bg="video" aria-pressed="${v.video ? 'true' : 'false'}"${v.video ? '' : ' disabled'} title="把原视频当前帧垫在图层下面">叠加原视频</button>` +
        `<button type="button" data-bg="layers" aria-pressed="${v.video ? 'false' : 'true'}" title="背景换成静态背景层（采样帧的中位数）">只看图层</button></span>` +
        '<label><input type="checkbox" class="raw"> 跟踪质心</label>' +
        (s.scale.degenerate ? '<span class="fwarn">尺度退化：prior 量不出像素长度</span>' : '') + '</div>' +
        '<div class="viewer cam l2d"><canvas></canvas></div>' +
        '<div class="timebar"><button type="button" class="play">播放</button><span class="tq-track"><input type="range" min="0" max="1000" value="0" aria-label="时间"><span class="tq-marks"></span></span><span class="tlabel">t = 0.00 s</span></div>' +
        '<p class="muted" style="font-size:12.5px;margin:6px 0 0">每个物体一层、一个标签，颜色和下方图层表对应；标签第 1 行物体名，有尺寸题时第 2 行是题目问的尺寸（L 沿朝向、W 垂直朝向、H 竖直、D 直径），prior 所在的层再加一行黄字 prior；标签放在凸包外侧，不压任何物体。' +
        '只画当前帧跟踪到的层，左上角计数"视频可见 / 画出"应相等。' +
        '半透明填充 = 掩码凸包；运动的层另画粗虚线轨迹（各分段运动模型，空心方块 = 分段起止点）和实心点（当前时刻的模型位置）。黄线 = prior 本身（尺寸画成带端线的标注线，速度画成 1 s 位移的箭头）。' +
        '模型类型和残差见下方图层表。左下角是比例尺和尺度来源' + (s.gravity_dir ? '，右上角箭头是重力方向' : '') + '。' +
        (v.video ? '时间轴和左边的视频联动。' : '') + '</p></div>';
    }
    return `<div class="media"><div>${vid}</div>${scene}</div>`;
  }

  // focal-length votes of the scene fit (scene_fit_eval json "focal_votes": gravity parabolas, size priors, lab prior …)
  const F_SRC = { gravity: '重力抛物线', lab_prior: '实验室先验', size_prior_depth: '尺寸先验 + 深度', typical_votes: '典型尺寸', typical_height: '典型高度', prior_kinematic: '运动学先验', person: '人身高', votes_first: '投票优先' };
  function focalVotes(s) {
    if (!s.focal_votes || !s.focal_votes.length) return '';
    const tw = s.focal_votes.reduce((a, x) => a + (x.weight || 0), 0);
    return '<details class="sfres" open><summary>焦距投票（' + s.focal_votes.length + ' 票' + (s.f_first ? '，速度 prior 前先由投票定焦距' : '') + '）</summary><div class="scroll-x"><table><tr><th>来源</th><th class="n">f (px)</th><th class="n">权重</th></tr>' +
      s.focal_votes.map((x) => `<tr><td>${esc(F_SRC[x.source] || x.source)}</td><td class="n">${fmtNum(x.value)}</td><td class="n">${fmtNum(x.weight)}${tw ? ` <span class="muted">(${fmt((100 * x.weight) / tw, 0)}%)</span>` : ''}</td></tr>`).join('') +
      `</table></div><p class="muted" style="font-size:12.5px">最终 f = ${fmtNum(s.f)} px。</p></details>`;
  }

  // conf-lowering signals of one scene-fit answer (quality fields written by scene_fit)
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

  function sceneTables(v, split) {
    if (!v.scene) return '<h4>3D 场景</h4><div class="noscene">无场景拟合（拟合失败或没有结果）。</div>';
    const s = v.scene;
    const hasSup = s.objects.some((o) => o.support), hasVis = s.objects.some((o) => o.visible);
    const hasSrc = s.objects.some((o) => o.dim_source);
    let rows = '';
    s.objects.forEach((o, i) => {
      const [L, W, H] = dimsText(o);
      rows += `<tr><td><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span></td><td>${esc(o.name)}</td><td>${SHAPE_LABEL[o.shape] || esc(o.shape)}</td>` +
        `<td class="n">${L}</td><td class="n">${W}</td><td class="n">${H}</td>` + (hasSup ? `<td>${esc(supportText(o))}</td>` : '') +
        (hasVis ? `<td>${o.visible ? o.visible.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ') + ' s' : '<span class="muted">无跟踪</span>'}</td>` : '') +
        `<td class="n">${fmt(o.rms, 1)}</td></tr>`;
    });
    let res = '';
    const hasConf = s.results.some((r) => r.conf != null);
    if (s.results.length) {
      res = `<details class="sfres"><summary>场景拟合自己的答案（${s.results.length} 题；S3 题置信度 ≥ 0.6 时作为最终预测）</summary><div class="scroll-x"><table><tr><th>问题</th><th>物体</th><th>方法</th><th class="n">预测</th>` +
        (hasConf ? '<th title="压低置信度的信号：自有平面、支撑不确定、支撑不一致、看不见的维度（bound）、塌缩、静态结构">信号</th>' : '') +
        (hasConf ? '<th class="n" title="逐题置信度：固定规则的乘积（残差、深度锚点、尺寸来源、焦距来源、地面、时刻是否在可见区间、可见性一致度、结构、塌缩），悬停看各项">置信度</th>' : '') +
        (split === 'val' ? '<th class="n">答案</th><th class="n">MRA</th>' : '') + '</tr>';
      s.results.forEach((r) => {
        const qt = r.quality ? Object.entries(r.quality).map(([k, x]) => `${k}=${typeof x === 'number' ? fmtNum(x) : x}`).join('\n') : '';
        res += `<tr><td>${esc(r.q)}</td><td>${esc(r.obj || '')}</td><td><code>${esc(r.method || '')}</code></td><td class="n">${fmtNum(r.pred)}</td>` +
          (hasConf ? `<td>${confSignals(r.quality).map((x) => `<span class="sigtag">${esc(x)}</span>`).join(' ')}</td>` : '') +
          (hasConf ? `<td class="n" title="${esc(qt)}">${r.conf == null ? '—' : fmt(r.conf, 2)}</td>` : '') +
          (split === 'val' ? `<td class="n">${fmtNum(r.answer)}</td><td class="n">${r.pred == null ? '<span class="muted">未答</span>' : mraBadge(r.mra)}</td>` : '') + '</tr>';
      });
      res += '</table></div></details>';
    }
    return graphBlocks(s.inventory, s.scene_graph, null, null) + `<h4>场景拟合的物体</h4><div><div class="scroll-x"><table><tr><th></th><th>物体</th><th>形状</th><th class="n">长 L (m)</th><th class="n">宽 W (m)</th><th class="n">高 H (m)</th>` +
      (hasSup ? '<th>支撑</th>' : '') + (hasVis ? '<th title="跟踪有框的时段；相机视角只在这些时段画出物体">视频里可见</th>' : '') + `<th class="n">残差 (px)</th></tr>${rows}</table></div>` +
      `<p class="muted" style="font-size:12.5px">焦距 f = ${fmtNum(s.f)} px` + (s.f_source ? `（来源 ${esc(s.f_source)}）` : '') + (s.main_support ? `，主支撑面 ${esc(s.main_support)}` : '') + '。残差是拟合框与跟踪框的均方根误差。' +
      (hasSrc ? '尺寸来源：不加标记的是观测到的（框、深度）；<sup class="dim-prior">先验</sup> 来自尺寸先验；<span class="dim-bound">灰色<sup>看不见</sup></span> 表示框和深度都没看到这一维，只能靠先验（取防塌缩下限）。' : '') +
      `</p>${focalVotes(s)}${res}</div>`;
  }


  // ---------- time axis shared by the 3D and 2D viewers ----------
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

  // ---------- 2D layered scene (pipeline/scene2d.py) ----------
  const MODEL_LABEL = { static: '静止', const_vel: '匀速', const_acc: '匀加速', circle: '圆周' };
  function segPos(s, t) {
    const p = s.p, dt = t - s.t0;
    if (s.model === 'static') return p.p;
    if (s.model === 'const_vel') return [p.p0[0] + p.v[0] * dt, p.p0[1] + p.v[1] * dt];
    if (s.model === 'const_acc') return [0, 1].map((k) => p.p0[k] + p.v0[k] * dt + 0.5 * p.a[k] * dt * dt);
    if (s.model === 'circle') { const th = p.theta0 + p.omega * dt; return [p.center[0] + p.r * Math.cos(th), p.center[1] + p.r * Math.sin(th)]; }
    return null;
  }
  function mPerPx(sc, y) {
    if (!sc.m_per_px_ref) return NaN;
    return sc.model === 'loglinear_y' ? sc.m_per_px_ref * Math.exp(-(sc.k_per_px || 0) * (y - sc.y_ref)) : sc.m_per_px_ref;
  }
  function niceLen(x) { // largest 1/2/5 x 10^k not above x
    const e = Math.pow(10, Math.floor(Math.log10(x))), m = x / e;
    return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * e;
  }
  const PRIOR_COL = '#ffd400';
  function noScaleReason(sc) {
    const n = (sc.notes || []).join(' ');
    if (/calibre|graduation/.test(n)) return '刻度不可跟踪';
    if (/scene structure/.test(n)) return 'prior 物体是场景结构，图层尺寸不是它说的那个尺寸';
    return n || '量不出像素长度';
  }
  function fmtLen(m) { return m >= 1 ? `${fmtNum(m)} m` : m >= 0.01 ? `${fmtNum(m * 100)} cm` : `${fmtNum(m * 1000)} mm`; }

  function startLayers(box, bar, v, video) {
    const s = v.scene2d, W = s.width, H = s.height, fps = s.fps || 30;
    const T = s.n_frames && fps ? s.n_frames / fps : (v.duration || 2);
    const canvas = box.querySelector('canvas'), ctx = canvas.getContext('2d');
    const vbar = box.previousElementSibling;
    const btns = [...vbar.querySelectorAll('.seg button')], rawBox = vbar.querySelector('.raw');
    let mode = video ? 'video' : 'layers', k = 1, dpr = 1, tCur = 0;
    const bg = new Image();
    if (s.bg) { bg.onload = () => draw(tCur); bg.src = s.bg; }
    const ac = new AbortController();

    function resize() {
      const bw = box.clientWidth, bh = box.clientHeight;
      if (!bw || !bh) return;
      k = Math.min(bw / W, bh / H);
      dpr = Math.min(devicePixelRatio || 1, 2);
      const cw = Math.round(W * k), ch = Math.round(H * k);
      canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
      canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px';
      draw(tCur);
    }
    let placed = [];   // label boxes already drawn this frame: a new label moves down until it does not overlap
    // light = true: white box with black text (prior annotations, so they stand apart from the layer colours)
    function label(x, y, txt, col, fixed, light) {
      ctx.font = '600 11.5px system-ui, sans-serif';
      const w = ctx.measureText(txt).width + 8;
      x = Math.max(0, Math.min(x, W * k - w)); y = Math.max(15, Math.min(y, H * k));
      placed.push([x, y - 15, w, 16]);   // [left, top, width, height]
      ctx.fillStyle = light ? 'rgba(255,255,255,.92)' : 'rgba(0,0,0,.62)'; ctx.fillRect(x, y - 15, w, 16);
      if (light) { ctx.strokeStyle = PRIOR_COL; ctx.lineWidth = 1.5; ctx.strokeRect(x, y - 15, w, 16); }
      ctx.fillStyle = light ? '#111' : col; ctx.fillText(txt, x + 4, y - 3);
    }
    // a dimension line from p0 to p1 (canvas px) with short end bars
    function dimLine(p0, p1, col) {
      const dx = p1[0] - p0[0], dy = p1[1] - p0[1], n = Math.hypot(dx, dy) || 1, nx = -dy / n * 6, ny = dx / n * 6;
      ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
      ctx.moveTo(p0[0] - nx, p0[1] - ny); ctx.lineTo(p0[0] + nx, p0[1] + ny);
      ctx.moveTo(p1[0] - nx, p1[1] - ny); ctx.lineTo(p1[0] + nx, p1[1] + ny); ctx.stroke();
    }
    function arrow(x0, y0, x1, y1, col) {
      const a = Math.atan2(y1 - y0, x1 - x0);
      ctx.strokeStyle = ctx.fillStyle = col; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 9 * Math.cos(a - 0.45), y1 - 9 * Math.sin(a - 0.45)); ctx.lineTo(x1 - 9 * Math.cos(a + 0.45), y1 - 9 * Math.sin(a + 0.45));
      ctx.closePath(); ctx.fill();
    }
    // one label per layer, up to three lines: object name (wrapped at word boundaries, ~160 px wide), the asked sizes,
    // the prior (yellow). Every layer's hull / box is an obstacle: try just above / below / right / left of the own
    // hull (6 px gap), then the free spot nearest to it with a thin leader; a label never covers a hull.
    const LINE_H = 15, MAX_W = 160;
    function wrapName(name) {
      ctx.font = '600 11.5px system-ui, sans-serif';
      const words = name.split(/\s+/), lines = [];
      let cur = '';
      words.forEach((wd) => {
        const t = cur ? cur + ' ' + wd : wd;
        if (cur && ctx.measureText(t).width > MAX_W) { lines.push(cur); cur = wd; } else cur = t;
      });
      if (cur) lines.push(cur);
      return lines;
    }
    function placeLabel(lines, col, anchor, bb, obstacles) {
      ctx.font = '600 11.5px system-ui, sans-serif';
      const cw = W * k, ch = H * k;
      const w = Math.max(...lines.map((l) => ctx.measureText(l.t).width)) + 8, h = lines.length * LINE_H + 3;
      const hit = (x, y, r) => x < r[0] + r[2] && r[0] < x + w && y < r[1] + r[3] && r[1] < y + h;
      const ok = (x, y) => x >= 0 && y >= 0 && x + w <= cw && y + h <= ch &&
        !obstacles.some((r) => hit(x, y, r)) && !placed.some((r) => hit(x, y, r));
      const cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2, g = 6;
      const cands = [[cx - w / 2, bb[1] - g - h], [cx - w / 2, bb[3] + g], [bb[2] + g, cy - h / 2], [bb[0] - g - w, cy - h / 2]]
        .map(([x, y]) => [Math.max(0, Math.min(x, cw - w)), Math.max(0, Math.min(y, ch - h))]);
      let pos = cands.find(([x, y]) => ok(x, y)), lead = false;
      // free spot nearest to the own hull; when hulls cover the whole frame (a road, a lane), only other labels count
      const nearest = (test) => {
        let best = null, bd = Infinity;
        for (let y = 0; y + h <= ch; y += 6) {
          for (let x = 0; x + w <= cw; x += 6) {
            const d = Math.hypot(x + w / 2 - cx, y + h / 2 - cy);
            if (d < bd && test(x, y)) { bd = d; best = [x, y]; }
          }
        }
        return best;
      };
      if (!pos) {
        pos = nearest(ok) || nearest((x, y) => !placed.some((r) => hit(x, y, r))) || cands[0];
        lead = true;
      }
      const [x, y] = pos;
      placed.push([x, y, w, h]);
      if (lead) {
        const lx = Math.max(x, Math.min(anchor[0], x + w)), ly = Math.max(y, Math.min(anchor[1], y + h));
        ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.globalAlpha = 0.85;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(anchor[0], anchor[1]); ctx.stroke(); ctx.globalAlpha = 1;
      }
      ctx.fillStyle = 'rgba(0,0,0,.66)'; ctx.fillRect(x, y, w, h);
      lines.forEach((l, i) => { ctx.fillStyle = l.kind === 'prior' ? PRIOR_COL : col; ctx.fillText(l.t, x + 4, y + (i + 1) * LINE_H - 2); });
    }
    const SIZE_KEY = { length: ['L', 'along'], width: ['W', 'across'], height: ['H', 'h'], diameter: ['D', 'diam'] };
    function priorText(pr) {
      if (pr.quantity === 'speed' || pr.quantity === 'velocity') {
        const mt = /at\s*t\s*=\s*([\d.]+)/i.exec(pr.text);
        return `prior v = ${fmtNum(pr.value_si)} m/s` + (mt ? ` @${mt[1]}s` : '');
      }
      if (pr.quantity === 'gravity') return `prior g = ${fmtNum(pr.value_si)} m/s²`;
      if (SIZE_KEY[pr.quantity]) return `prior ${pr.quantity} = ${fmtLen(pr.value_si)}`;
      return `prior ${pr.text}`;
    }
    function layerText(L, mpp) {
      const lines = wrapName(L.name).map((t) => ({ t, kind: 'name' }));
      const ext = L.extent || {};
      if (isFinite(mpp)) {
        const sz = (L.size_q || []).filter((q) => SIZE_KEY[q] && ext[SIZE_KEY[q][1]] != null)
          .map((q) => `${SIZE_KEY[q][0]} ${fmtLen(ext[SIZE_KEY[q][1]] * mpp)}`);
        if (sz.length) lines.push({ t: sz.join(' × '), kind: 'size' });
      }
      if (L.name === s.prior.layer && s.prior.value_si != null) lines.push({ t: priorText(s.prior), kind: 'prior' });
      return lines;
    }
    function draw(t) {
      tCur = t; placed = [];
      let priorPose = null;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cw = W * k, ch = H * k;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cw, ch);
      if (mode === 'video' && video && video.readyState >= 2) ctx.drawImage(video, 0, 0, cw, ch);
      else if (bg.complete && bg.naturalWidth) ctx.drawImage(bg, 0, 0, cw, ch);
      ctx.fillStyle = 'rgba(0,0,0,.18)'; ctx.fillRect(0, 0, cw, ch);
      const X = (p) => p[0] * k, Y = (p) => p[1] * k;
      const half = 0.5 / fps + 1e-6;
      const sc = s.scale, pr = s.prior;
      const pending = [];   // labels are placed after all shapes, so the scale bar / g box can reserve their spots first
      let nTracked = 0;
      s.layers.forEach((L, i) => {
        const col = PALETTE[i % PALETTE.length];
        const still = !L.moving || L.segments.every((g) => g.model === 'static');
        if (rawBox.checked) {
          ctx.fillStyle = col; ctx.globalAlpha = 0.45;
          L.c.forEach((c) => { ctx.beginPath(); ctx.arc(X(c), Y(c), 1.6, 0, 2 * Math.PI); ctx.fill(); });
          ctx.globalAlpha = 1;
        }
        // model trajectories per segment (moving layers only)
        if (!still) {
          ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash([7, 4]);
          L.segments.forEach((sg) => {
            if (sg.model === 'static') return;
            ctx.beginPath();
            const nS = 80;
            for (let j = 0; j <= nS; j++) {
              const p = segPos(sg, sg.t0 + ((sg.t1 - sg.t0) * j) / nS);
              if (j) ctx.lineTo(X(p), Y(p)); else ctx.moveTo(X(p), Y(p));
            }
            ctx.stroke();
            ctx.setLineDash([]); ctx.lineWidth = 1.5;
            [sg.t0, sg.t1].forEach((tb) => { const q = segPos(sg, tb); ctx.strokeRect(X(q) - 3.5, Y(q) - 3.5, 7, 7); });
            if (sg.model === 'circle') {
              const c = sg.p.center; ctx.lineWidth = 1.4;
              ctx.beginPath(); ctx.moveTo(X(c) - 6, Y(c)); ctx.lineTo(X(c) + 6, Y(c)); ctx.moveTo(X(c), Y(c) - 6); ctx.lineTo(X(c), Y(c) + 6); ctx.stroke();
            }
            ctx.setLineDash([7, 4]); ctx.lineWidth = 2.2;
          });
          ctx.setLineDash([]);
        }
        let bi = -1, bd = Infinity;
        L.t.forEach((tt, j) => { const d = Math.abs(tt - t); if (d < bd) { bd = d; bi = j; } });
        const sg = L.segments.find((x) => t >= x.t0 - half && t <= x.t1 + half);
        let cur = null, bb = null, anchor = null;
        if (bi >= 0 && bd <= half) {
          const b = L.box[bi];
          cur = { c: L.c[bi], box: b, hull: hullAt(L, bi) };
          if (cur.hull) {
            ctx.fillStyle = col; ctx.globalAlpha = 0.3;
            ctx.beginPath(); cur.hull.forEach((q, j) => (j ? ctx.lineTo(X(q), Y(q)) : ctx.moveTo(X(q), Y(q)))); ctx.closePath(); ctx.fill();
            ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
            const xs = cur.hull.map((q) => q[0] * k), ys = cur.hull.map((q) => q[1] * k);
            bb = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
            anchor = [xs.reduce((a, x) => a + x, 0) / xs.length, ys.reduce((a, y) => a + y, 0) / ys.length];
          } else {
            ctx.strokeStyle = col; ctx.lineWidth = 2;
            ctx.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
            bb = [b[0] * k, b[1] * k, b[2] * k, b[3] * k]; anchor = [X(cur.c), Y(cur.c)];
          }
        } else if (sg) {
          cur = { c: segPos(sg, t) };   // between tracked frames: the model dot only, no shape and no label
        }
        nTracked += bi >= 0 && bd <= half;
        if (sg && !still) {
          const p = segPos(sg, Math.min(Math.max(t, sg.t0), sg.t1));
          ctx.fillStyle = col; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(X(p), Y(p), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        }
        if (L.name === pr.layer) priorPose = { L, cur, sg };
        if (bb) pending.push([layerText(L, sc.degenerate ? NaN : mPerPx(sc, cur.c[1])), col, anchor, bb]);
      });
      drawPrior(priorPose, t);
      // one line at the bottom left: scale bar + where the scale comes from
      const yb = 0.93 * H, mpp = mPerPx(sc, yb), x0 = 10, yy = yb * k;
      ctx.font = '600 11.5px system-ui, sans-serif';
      if (isFinite(mpp) && !sc.degenerate) {
        const len = niceLen(0.22 * W * mpp), px = (len / mpp) * k;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x0 + px, yy); ctx.moveTo(x0, yy - 5); ctx.lineTo(x0, yy + 5); ctx.moveTo(x0 + px, yy - 5); ctx.lineTo(x0 + px, yy + 5); ctx.stroke();
        label(x0 + px + 8, yy + 6, fmtLen(len) + (pr.how ? ` · 尺度来自 prior：${pr.how}` : ''), '#fff', true);
      } else {
        let msg = `prior 不能定尺度（${noScaleReason(sc)}）`;
        ctx.font = '600 11.5px system-ui, sans-serif';
        while (msg.length > 12 && ctx.measureText(msg).width + 8 > W * k - x0) msg = msg.slice(0, -2) + '…';
        label(x0, yy + 6, msg, '#ff8a80', true);
      }
      if (s.gravity_dir) {
        const g = s.gravity_dir, cx = W * k - 40, cy = 40;
        arrow(cx - g[0] * 22, cy - g[1] * 22, cx + g[0] * 22, cy + g[1] * 22, '#ffd43b');
        label(cx + 8, cy - 20, 'g', '#ffd43b', true);
      }
      // consistency count: layers tracked in this frame vs layers drawn (should always agree)
      label(8, 20, `视频可见 ${nTracked} / 画出 ${pending.length}`, nTracked === pending.length ? '#fff' : '#ff8a80', true);
      const obstacles = pending.map((a) => [a[3][0], a[3][1], a[3][2] - a[3][0], a[3][3] - a[3][1]]);
      pending.forEach((a) => placeLabel(a[0], a[1], a[2], a[3], obstacles));
    }
    // hull of the key frame nearest to tracked frame bi, shifted by the centroid motion between the two frames
    function hullAt(L, bi) {
      if (!L.hulls || !L.hulls.length) return null;
      const t = L.t[bi];
      let h = L.hulls[0];
      L.hulls.forEach((x) => { if (Math.abs(x[0] - t) < Math.abs(h[0] - t)) h = x; });
      let hj = 0;
      L.t.forEach((tt, j) => { if (Math.abs(tt - h[0]) < Math.abs(L.t[hj] - h[0])) hj = j; });
      const dx = L.c[bi][0] - L.c[hj][0], dy = L.c[bi][1] - L.c[hj][1];
      return h[1].map((q) => [q[0] + dx, q[1] + dy]);
    }
    // the prior drawn on its own layer (its text is part of that layer's label): a dimension line for sizes,
    // a 1 s displacement arrow for speeds (bright near the time the prior names, faint elsewhere)
    function drawPrior(pp, t) {
      const pr = s.prior;
      if (!pp || !pp.cur || !pr.px || pr.quantity === 'gravity') return;
      const { L, cur, sg } = pp, c = cur.c, ext = L.extent || {};
      const o = L.orient && Math.hypot(L.orient[0], L.orient[1]) > 0 ? L.orient : [1, 0], perp = [-o[1], o[0]];
      const P = (q) => [q[0] * k, q[1] * k];
      if (SIZE_KEY[pr.quantity]) {
        let d, off;
        if (pr.quantity === 'height') { d = [0, 1]; off = [(ext.w || 0) / 2 + 8, 0]; }
        else if (pr.quantity === 'length') { d = o; off = perp.map((v) => v * ((ext.across || ext.h || 0) / 2 + 8)); }
        else if (pr.quantity === 'width') { d = perp; off = o.map((v) => v * ((ext.along || ext.w || 0) / 2 + 8)); }
        else { d = [1, 0]; off = [0, (ext.h || 0) / 2 + 8]; }
        if (pr.quantity !== 'height' && off[1] < 0) off = off.map((v) => -v);   // below the object: the label sits above it
        const m = [c[0] + off[0], c[1] + off[1]], h = pr.px / 2;
        dimLine(P([m[0] - d[0] * h, m[1] - d[1] * h]), P([m[0] + d[0] * h, m[1] + d[1] * h]), PRIOR_COL);
        return;
      }
      if (pr.quantity === 'speed' || pr.quantity === 'velocity') {
        let dir = o;
        if (sg && sg.model !== 'static') {
          const a = segPos(sg, Math.max(sg.t0, t - 0.02)), b = segPos(sg, Math.min(sg.t1, t + 0.02)), n = Math.hypot(b[0] - a[0], b[1] - a[1]);
          if (n > 1e-6) dir = [(b[0] - a[0]) / n, (b[1] - a[1]) / n];
        }
        const mt = /at\s*t\s*=\s*([\d.]+)/i.exec(pr.text), tp = mt ? parseFloat(mt[1]) : NaN;
        ctx.globalAlpha = !isFinite(tp) || Math.abs(t - tp) < 0.15 ? 1 : 0.35;
        const p0 = P(c), p1 = P([c[0] + dir[0] * pr.px, c[1] + dir[1] * pr.px]);
        arrow(p0[0], p0[1], p1[0], p1[1], PRIOR_COL);
        ctx.globalAlpha = 1;
      }
    }
    function setMode(m) {
      mode = m;
      btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.bg === m)));
      draw(tCur);
    }
    btns.forEach((b) => { b.onclick = () => { if (!b.disabled) setMode(b.dataset.bg); }; });
    rawBox.onchange = () => draw(tCur);
    const tl = linkTime(bar, video, T, draw, ac.signal);
    if (video) {
      ['loadeddata', 'canplay', 'seeked'].forEach((ev) => video.addEventListener(ev, () => draw(tCur), { signal: ac.signal }));
      video.addEventListener('error', () => { btns[0].disabled = true; setMode('layers'); }, { signal: ac.signal });
    }
    const ro = new ResizeObserver(resize); ro.observe(box);
    resize(); tl.redraw();
    return {
      dispose() {
        ac.abort(); tl.stop(); ro.disconnect();
        btns.forEach((b) => { b.onclick = null; }); rawBox.onchange = null; bg.onload = null;
        canvas.width = canvas.height = 0;
      },
    };
  }

  // ---------- object inventory + scene graph (2D scenes now; 3D scene fits once they carry them) ----------
  function noteZh(n) {
    const m = /(\d+) of (\d+) segment\(s\) leave the surface line/.exec(n || '');
    if (m) return `${m[2]} 段里有 ${m[1]} 段离开支撑面那条线，自由模型拟合更好`;
    if (/track is not along one line/.test(n || '')) return '轨迹不在一条直线上';
    return n || '';
  }
  function rejectZh(r) {
    const m = /duplicate track of '(.*)'/.exec(r || '');
    return m ? `和 "${esc(m[1])}" 用的是同一条跟踪，而场景图判断两者不是同一个物体，所以拒绝较弱的这一层` : esc(r);
  }
  function graphBlocks(inv, sg, rejected, layers) {
    let out = '';
    if (inv && inv.length) {
      const o2 = (sg && sg.objects_2d) || {};
      let rows = '';
      inv.forEach((o) => {
        const al = o.aliases.filter((a) => a !== o.name);
        const role = [o.track === false ? '不跟踪（场景结构）' : '跟踪', o.dyn ? '动态' : (o.dyn === false ? '静态' : ''), o.role || ''].filter(Boolean).join(' · ');
        const mp = o2[o.name] ? [o2[o.name].motion_plane, o2[o.name].surface].filter(Boolean).join(' / ') : '';
        rows += `<tr><td>${esc(o.name)}${rejected && rejected[o.name] ? ' <span class="fwarn">（层被拒绝）</span>' : ''}</td><td>${al.length ? al.map(esc).join('；') : '<span class="muted">—</span>'}</td>` +
          `<td>${esc(o.where.join(', '))}</td><td>${esc(role)}</td><td>${esc(mp)}</td></tr>`;
      });
      out += '<h4>物体清单</h4><div class="scroll-x"><table><tr><th>物体（canonical 名）</th><th>别名</th><th>出现在</th><th>角色</th><th>运动平面 / 支撑面</th></tr>' + rows + '</table></div>';
    }
    if (sg || (rejected && Object.keys(rejected).length)) {
      const items = [];
      ((sg && sg.edges) || []).forEach((e) => {
        const tr = e.t0 != null && e.t1 != null ? ` <span class="muted">${fmt(e.t0, 2)}–${fmt(e.t1, 2)} s</span>` : '';
        const cf = e.conf != null ? ` <span class="muted">(conf ${fmt(e.conf, 2)})</span>` : '';
        items.push(e.to ? `${esc(e.from)} <b>${esc(e.rel)}</b> ${esc(e.to)}${cf}${tr}` : `${esc(e.from)} — <b>${esc(e.rel)}</b>${cf}${tr}`);
      });
      ((sg && sg.same) || []).forEach((x) => items.push(`${esc(x.a)} <b>${x.same ? '=' : '≠'}</b> ${esc(x.b)}` + (x.conf != null ? ` <span class="muted">(conf ${fmt(x.conf, 2)})</span>` : '')));
      Object.entries(rejected || {}).forEach(([k, r]) => items.push(`<span class="fwarn">拒绝的层</span> ${esc(k)}：${rejectZh(r)}`));
      (layers || []).forEach((L) => {
        const c = L.constraint || {};
        if (!c.mode) return;
        const rel = c.mode === 'surface' ? `on ${c.surface || '支撑面'}` : c.mode;
        if (c.applied === false) items.push(`<span class="fwarn">约束被轨迹否决</span> ${esc(L.name)} ${esc(rel)}：${esc(noteZh(c.note))}`);
        else if (c.applied === true) items.push(`约束已用于拟合：${esc(L.name)} ${esc(rel)}`);
      });
      if (sg && sg.support && sg.support.length) items.push(`<span class="muted">支撑面节点：${sg.support.map(esc).join('、')}</span>`);
      out += `<h4>场景图${sg && sg.source ? `<span class="muted" style="font-weight:400">（来源 ${esc(sg.source)}）</span>` : ''}</h4>` +
        (items.length ? `<ul class="sg-list">${items.map((x) => `<li>${x}</li>`).join('')}</ul>` : '<p class="muted">没有边。</p>');
    }
    return out;
  }

  // s2d_v5 consistency checks (each block only when the scene json has it)
  function checkBlocks(s) {
    const ok = (x) => (x === false ? '<span class="fwarn">✗</span>' : x === true ? '✓' : '');
    const items = [];
    (s.relation_checks || []).forEach((r) => r.candidates.forEach((c) => c.checks.forEach((x) =>
      items.push(`${ok(x.ok)} ${esc(r.from)} <b>${esc(x.rel)}</b> ${esc(x.to)} <span class="muted">（候选 ${esc(c.cache)}：${esc(x.why)}）</span>`))));
    (s.size_checks || []).forEach((x) =>
      items.push(`${ok(x.ok)} ${esc(x.smaller)} &lt; ${esc(x.larger)} <span class="muted">（${(x.px || []).map(fmtNum).join(' / ')} px，${esc(x.why)}）</span>`));
    Object.entries(s.refmeasure || {}).forEach(([k, m]) =>
      items.push(`参考测量 <b>${esc(k)}</b> = ${fmtNum(m.px)} px <span class="muted">（第 ${esc(m.frame)} 帧，${esc(m.relation)}，conf ${fmt(m.conf, 2)}${m.spread != null ? `，帧间离散 ${fmt(m.spread, 3)}` : ''}）</span>`));
    if (s.critique) {
      s.critique.applied.forEach((a) => items.push(`评审采纳：<code>${esc(a)}</code>`));
      s.critique.rejected.forEach((a) => items.push(`<span class="muted">评审未采纳：<code>${esc(a)}</code></span>`));
    }
    return items.length ? `<h4>一致性检查</h4><ul class="sg-list">${items.map((x) => `<li>${x}</li>`).join('')}</ul>` : '';
  }

  function layerTables(v) {
    const s = v.scene2d, sc = s.scale, pr = s.prior;
    let rows = '';
    s.layers.forEach((L, i) => {
      const segs = L.segments.map((g) => `${MODEL_LABEL[g.model] || esc(g.model)} <span class="muted">${fmt(g.t0, 2)}–${fmt(g.t1, 2)} s · rms ${fmt(g.rms, 1)}</span>`).join('<br>');
      const ext = L.extent || {}, mpp = sc.degenerate ? NaN : sc.m_per_px_ref;
      const ext_m = ext.along != null && isFinite(mpp) ? `<div class="muted">${fmtLen(ext.along * mpp)} × ${fmtLen(ext.across * mpp)}</div>` : '';
      rows += `<tr><td><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span></td><td>${esc(L.name)}` +
        (L.flags.length ? `<div class="known">${L.flags.map(esc).join('；')}</div>` : '') + '</td>' +
        `<td>${L.visible.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ')} s</td><td>${segs}</td>` +
        `<td class="n">${fmtNum(ext.along)} × ${fmtNum(ext.across)}${ext_m}</td><td class="n">${fmtNum(ext.h)}</td><td>${esc(ext.src || '')}</td><td class="n">${fmt(L.rms, 2)}</td></tr>`;
    });
    const scaleTxt = !sc.m_per_px_ref || sc.degenerate ? `没有可用的尺度（${noScaleReason(sc)}）` : sc.model === 'loglinear_y'
      ? `透视模型 m/px(y) = ${fmtNum(sc.m_per_px_ref)} · exp(−${fmtNum(sc.k_per_px)} · (y − ${fmtNum(sc.y_ref)}))`
      : `均匀 ${fmtNum(sc.m_per_px_ref)} m/px`;
    return graphBlocks(s.inventory, s.scene_graph, s.rejected, s.layers) + checkBlocks(s) +
      '<h4>图层</h4><div class="scroll-x"><table><tr><th></th><th>图层（物体）</th><th>可见时段</th><th>分段运动模型</th>' +
      '<th class="n" title="length = 沿朝向（运动方向；静止物体取掩码主轴），width = 垂直朝向">length × width (px)</th><th class="n">竖直高度 (px)</th><th>尺寸来源</th><th class="n">残差 (px)</th></tr>' +
      rows + '</table></div>' +
      `<p class="muted" style="font-size:12.5px">尺度：${scaleTxt}（${s.width}×${s.height} px 坐标）；` +
      (pr.layer ? `来自 prior "${esc(pr.text)}"，在图层 "${esc(pr.layer)}" 上量得 ${fmtNum(pr.px)} ${pr.quantity === 'speed' ? 'px/s' : pr.quantity === 'gravity' ? 'px/s²' : 'px'}（${esc(pr.how)}）。` : `prior "${esc(pr.text)}"。`) +
      (sc.model !== 'loglinear_y' && sc.persp_reason ? `没用透视模型：${esc(sc.persp_reason)}。` : '') +
      (sc.notes.length ? sc.notes.map(esc).join('；') + '。' : '') +
      (s.gravity_dir ? `重力方向（图像坐标）(${fmt(s.gravity_dir[0], 3)}, ${fmt(s.gravity_dir[1], 3)})${s.gravity_source ? `，来自 ${esc(s.gravity_source)}` : ''}。` : '') +
      '残差是该段模型位置与跟踪质心的均方根误差。</p>';
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
    const T = v.duration || 2;
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
    const halfF = 0.5 / (v.fps || 30) + 1e-6;
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
    const [iw, ih] = v.size && v.size[0] ? v.size : [16, 9];
    const f = s.f || iw;
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
      if (o.shape === 'sphere') { geo = new THREE.SphereGeometry((d.D || 0.1) / 2, 24, 16); axes = [h, e2, n]; }
      else if (o.shape === 'cylinder') { geo = new THREE.CylinderGeometry((d.D || 0.1) / 2, (d.D || 0.1) / 2, d.H || 0.1, 24); axes = [h, n, e2.map((x) => -x)]; }
      else { geo = new THREE.BoxGeometry(d.L || 0.1, d.W || 0.1, d.H || 0.1); axes = [h, e2, n]; }
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
  function renderVal() {
    const cat = $('v-cat').value, src = $('v-src').value, vsrc = $('v-vsrc').value, mf = $('v-mra').value, sort = $('v-sort').value;
    const needle = $('v-q').value.trim().toLowerCase();
    const okM = (m) => !mf || (mf === 'lt0.2' ? m < 0.2 : mf === 'lt0.5' ? m < 0.5 : m >= 0.8);
    let items = DATA.val.videos.filter((v) => !vsrc || v.source === vsrc).map((v) => {
      const qs = v.questions.filter((q) => (!cat || q.cat === cat) && (!src || q.src === src) && okM(q.mra) && matchText(v, q, needle));
      return { v, qs, m: mean(qs.map((q) => q.mra)) };
    }).filter((x) => x.qs.length);
    const nanLast = (f) => (a, b) => (isNaN(a.m) ? 1 : 0) - (isNaN(b.m) ? 1 : 0) || f(a, b);
    if (sort === 'mra-asc') items.sort(nanLast((a, b) => a.m - b.m));
    else if (sort === 'mra-desc') items.sort(nanLast((a, b) => b.m - a.m));
    else if (sort === 'n-desc') items.sort((a, b) => b.qs.length - a.qs.length);
    const allQ = items.flatMap((x) => x.qs);
    const filtered = cat || src || vsrc || mf || needle;
    $('v-status').textContent = `${items.length} 个视频、${allQ.length} 题` + (filtered && allQ.length ? `；筛选后逐题平均 MRA ${fmt(mean(allQ.map((q) => q.mra)))}` : '') + '。视频右侧是该视频（当前筛选下）的平均 MRA。';
    const list = $('val-list'); list.innerHTML = '';
    items.forEach((x) => list.appendChild(videoCard(x.v, x.qs, 'val')));
  }
  function renderTest() {
    const cat = $('t-cat').value, vsrc = $('t-vsrc').value, fit = $('t-fit').value, sort = $('t-sort').value;
    const needle = $('t-q').value.trim().toLowerCase();
    let items = DATA.test.videos.filter((v) => (!vsrc || v.source === vsrc) &&
      (!fit || (fit === 'yes' ? hasScene(v) : v.dim === 3 && !hasScene(v)))).map((v) => {
      const qs = v.questions.filter((q) => (!cat || q.cat === cat) && matchText(v, q, needle));
      return { v, qs };
    }).filter((x) => x.qs.length);
    if (sort === 'n-desc') items.sort((a, b) => b.qs.length - a.qs.length);
    $('t-status').textContent = `${items.length} 个视频、${items.reduce((s, x) => s + x.qs.length, 0)} 题。`;
    const list = $('test-list'); list.innerHTML = '';
    items.forEach((x) => list.appendChild(videoCard(x.v, x.qs, 'test')));
  }
  function debounce(fn, ms) { let t = 0; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  fetch('data/videos.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then((d) => {
    DATA = d;
    $('gen').textContent = `数据生成于 ${d.generated}（tools/build_videos.py）。`;
    valSummary(d.val); testSummary(d.test);
    fillSourceSelect($('v-vsrc'), d.val.videos); fillSourceSelect($('t-vsrc'), d.test.videos);
    ['v-cat', 'v-src', 'v-vsrc', 'v-mra', 'v-sort'].forEach((id) => $(id).addEventListener('change', renderVal));
    ['t-cat', 't-vsrc', 't-fit', 't-sort'].forEach((id) => $(id).addEventListener('change', renderTest));
    $('v-q').addEventListener('input', debounce(renderVal, 200));
    $('t-q').addEventListener('input', debounce(renderTest, 200));
    renderVal(); renderTest();
    let tab = null;
    try { tab = localStorage.getItem('qp-videos-tab'); } catch (e) { /* storage unavailable */ }
    // deep link: videos.html#val/<video_id> or #test/<video_id> opens that video
    function openFromHash() {
      const m = decodeURIComponent(location.hash.slice(1)).match(/^(val|test)\/(.+)$/);
      if (!m) return false;
      $('tab-' + m[1]).click();
      const card = [...$(m[1] + '-list').children].find((d) => d.dataset.id === m[2]);
      if (card) { card.open = true; card.scrollIntoView(); }
      return true;
    }
    if (!openFromHash()) $(tab === 'test' ? 'tab-test' : 'tab-val').click();
    addEventListener('hashchange', openFromHash);
  }).catch((e) => {
    $('val-list').innerHTML = `<p class="noscene">读取 data/videos.json 失败（${esc(e.message)}）。先运行 tools/build_videos.py。</p>`;
  });
})();
