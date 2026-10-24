// videos.html: per-video question browser. Data: data/videos.json (built by tools/build_videos.py).
(function () {
  'use strict';
  const CATS = ['S2', 'D2', 'S3', 'D3'];
  const SRC_LABEL = { geo: '几何', vlm: 'Qwen' };
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
    $('val-macro').textContent = fmt(m.macro, 3);
    $('val-src-file').innerHTML = `${m.n} 题、${val.videos.length} 个视频，来自 <code>${esc(val.file)}</code>；micro（逐题平均）${fmt(m.micro, 3)}`;
    let h = '<tr><th>类别</th><th class="n">题数</th><th class="n">MRA</th><th class="n">几何 题数</th><th class="n">几何 MRA</th><th class="n">Qwen 题数</th><th class="n">Qwen MRA</th></tr>';
    const row = (label, qs, strong) => {
      const g = qs.filter((q) => q.src === 'geo'), v = qs.filter((q) => q.src === 'vlm');
      const s = (x) => (strong ? `<b>${x}</b>` : x);
      return `<tr><td>${s(label)}</td><td class="n">${qs.length}</td><td class="n">${s(fmt(mean(qs.map((q) => q.mra))))}</td>` +
        `<td class="n">${g.length}</td><td class="n">${g.length ? fmt(mean(g.map((q) => q.mra))) : '—'}</td>` +
        `<td class="n">${v.length}</td><td class="n">${v.length ? fmt(mean(v.map((q) => q.mra))) : '—'}</td></tr>`;
    };
    CATS.forEach((c) => { h += row(c, all.filter((q) => q.cat === c)); });
    h += `<tr><td><b>macro</b></td><td class="n">${all.length}</td><td class="n"><b>${fmt(m.macro)}</b></td><td colspan="4" class="muted">四类 MRA 的平均</td></tr>`;
    $('val-sum').innerHTML = h;
  }
  function testSummary(test) {
    const all = test.videos.flatMap((v) => v.questions);
    const v3 = test.videos.filter((v) => v.dim === 3);
    const fit = v3.filter((v) => v.scene).length;
    $('test-src-file').innerHTML = `${all.length} 题、${test.videos.length} 个视频，预测值来自 <code>${esc(test.file)}</code>（按行号和 <code>load_split('test')</code> 对齐）。测试集没有答案，不显示准确度。3D 视频 ${v3.length} 个，其中 ${fit} 个有场景拟合结果。`;
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

  function videoCard(v, qs, split) {
    const d = document.createElement('details');
    d.className = 'vid';
    d.dataset.id = v.id;
    const m = split === 'val' ? mean(qs.map((q) => q.mra)) : NaN;
    const sceneTag = v.dim === 3 ? (v.scene ? '<span class="tag">场景拟合</span>' : '<span class="tag" style="opacity:.6">无场景拟合</span>')
      : (v.scene2d ? '<span class="tag">图层场景</span>' : '');
    const thumb = v.thumb ? `<img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}">` : '<span class="thumb"></span>';
    d.innerHTML = `<summary>${thumb}<div class="vmeta"><div class="vid-id">${esc(v.id)}</div>` +
      `<div class="sub">${esc(v.source)} · ${esc(v.video_type)} · ${v.fps} fps${v.duration ? ' · ' + fmt(v.duration, 1) + ' s' : ''} · ${qs.length} 题 ` +
      v.cats.map((c) => `<span class="tag">${c}</span>`).join('') + sceneTag + '</div></div>' +
      (split === 'val' ? `<div title="该视频（当前筛选下）的平均 MRA">${mraBadge(m)}</div>` : '') + '</summary>';
    let built = false, viewer = null, video = null;
    d.addEventListener('toggle', () => {
      if (d.open && !built) {
        built = true;
        const body = document.createElement('div');
        body.className = 'vbody';
        body.innerHTML = mediaBlock(v, split) + questionTable(qs, split, v.dim === 2 && !!v.scene2d) +
          (v.dim === 3 ? sceneTables(v, split) : v.scene2d ? layerTables(v) : '');
        d.appendChild(body);
        video = body.querySelector('video');
      }
      // the <video> only holds a src while its card is open, so collapsed cards do not keep connections or buffers
      if (video) {
        if (d.open && !video.getAttribute('src')) video.src = v.video;
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
    });
    return d;
  }

  function questionTable(qs, split, s2d) {
    const kn = (q) => (q.prior ? `已知：${esc(q.prior)}` : '') + (q.depth ? `\ndepth_info：${esc(q.depth)}` : '');
    // when every question of the video has the same prior / depth_info, show it once above the table
    const shared = qs.every((q) => q.prior === qs[0].prior && q.depth === qs[0].depth);
    const known = (q) => (shared ? '' : kn(q));
    let h = '<h4>问题</h4>' + (shared && kn(qs[0]) ? `<div class="known-shared">${kn(qs[0])}</div>` : '') +
      '<div class="scroll-x"><table class="qtable"><tr><th>类别</th><th>' + (shared ? '问题' : '问题与已知条件') + '</th>';
    if (split === 'val') {
      h += '<th class="n">答案</th><th class="n">最终预测</th><th>来源</th><th class="n">几何值</th>' +
        (s2d ? '<th class="n" title="图层场景（pipeline.scene2d）的值，作为第二个提示给 Qwen">2D 场景值</th>' : '') +
        '<th class="n">Qwen 值</th><th class="n">MRA</th></tr>';
      qs.forEach((q) => {
        h += `<tr><td><span class="tag">${q.cat}</span><span class="muted">${esc(q.type)}</span></td>` +
          `<td class="q">${esc(q.q)}<div class="known">${known(q)}</div></td>` +
          `<td class="n">${fmtNum(q.answer)}</td><td class="n pred">${fmtNum(q.pred)}</td><td>${SRC_LABEL[q.src] || esc(q.src)}</td>` +
          `<td class="n">${fmtNum(q.geo)}</td>` + (s2d ? `<td class="n">${fmtNum(q.s2d)}</td>` : '') +
          `<td class="n">${fmtNum(q.vlm)}</td><td class="n">${mraBadge(q.mra)}</td></tr>`;
      });
    } else {
      h += '<th class="n">行号</th><th class="n">预测值</th></tr>';
      qs.forEach((q) => {
        h += `<tr><td><span class="tag">${q.cat}</span><span class="muted">${esc(q.type)}</span></td>` +
          `<td class="q">${esc(q.q)}<div class="known">${known(q)}</div></td><td class="n">${q.row}</td><td class="n pred">${fmtNum(q.pred)}</td></tr>`;
      });
    }
    return h + '</table></div>';
  }

  function dimsText(o) {
    const d = o.dims || {};
    if (o.shape === 'sphere') return [`直径 ${fmtNum(d.D)}`, '', ''];
    if (o.shape === 'cylinder') return [`直径 ${fmtNum(d.D)}`, '', fmtNum(d.H)];
    return [fmtNum(d.L), fmtNum(d.W), fmtNum(d.H)];
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
        (v.video ? '<label class="off" title="相机视角下把原视频画面垫在渲染下面"><input type="checkbox" class="overlay" disabled> 叠加原视频</label>' : '') +
        (v.scene.f ? '' : '<span class="fwarn" title="场景拟合没有给出焦距，用图像宽度代替">焦距为默认值（f = 图像宽度）</span>') + '</div>' +
        '<div class="viewer"><div class="msg">加载 three.js …</div></div>' +
        '<div class="timebar"><button type="button" class="play">播放</button><input type="range" min="0" max="1000" value="0" aria-label="时间"><span class="tlabel">t = 0.00 s</span></div>' +
        '<p class="muted" style="font-size:12.5px;margin:6px 0 0">灰色网格 = 拟合出的地面；黑色线框 = 相机（原点）与视野；彩色 = 物体及其匀加速轨迹（虚线）。自由视角下拖动旋转，滚轮缩放；相机视角固定在原视频相机处，画面比例和原视频相同。' +
        (v.video ? '时间轴和左边的视频联动。' : '') + '</p></div>';
    }
    if (v.dim === 2 && v.scene2d) {
      const s = v.scene2d;
      scene = `<div><h4>图层场景（<code>${esc(s.file)}</code>）</h4>` +
        '<div class="viewbar"><span class="seg" role="group" aria-label="背景">' +
        `<button type="button" data-bg="video" aria-pressed="${v.video ? 'true' : 'false'}"${v.video ? '' : ' disabled'} title="把原视频当前帧垫在图层下面">叠加原视频</button>` +
        `<button type="button" data-bg="layers" aria-pressed="${v.video ? 'false' : 'true'}" title="背景换成静态背景层（采样帧的中位数）">只看图层</button></span>` +
        '<label><input type="checkbox" class="raw" checked> 跟踪质心</label>' +
        (s.scale.degenerate ? '<span class="fwarn">尺度退化：prior 量不出像素长度</span>' : '') + '</div>' +
        '<div class="viewer cam l2d"><canvas></canvas></div>' +
        '<div class="timebar"><button type="button" class="play">播放</button><input type="range" min="0" max="1000" value="0" aria-label="时间"><span class="tlabel">t = 0.00 s</span></div>' +
        '<p class="muted" style="font-size:12.5px;margin:6px 0 0">每个物体一层，颜色和下方图层表对应。实线框 = 当前帧的掩码包围框；实心点 = 运动模型在当前时刻的位置；' +
        '粗虚线 = 各分段运动模型的轨迹（圆周模型画圆弧，＋ 为圆心），空心方块 = 分段的起止点；小点 = 跟踪到的逐帧质心。左下角比例尺按 prior 标定的 m/px 画' +
        (s.scale.model === 'loglinear_y' ? '（透视模型，不同高度的比例尺长度不同）' : '') + (s.gravity_dir ? '；右上角箭头是重力方向' : '') + '。' +
        (v.video ? '时间轴和左边的视频联动。' : '') + '</p></div>';
    }
    return `<div class="media"><div>${vid}</div>${scene}</div>`;
  }

  function sceneTables(v, split) {
    if (!v.scene) return '<h4>3D 场景</h4><div class="noscene">无场景拟合（拟合失败或没有结果）。</div>';
    const s = v.scene;
    let rows = '';
    s.objects.forEach((o, i) => {
      const [L, W, H] = dimsText(o);
      rows += `<tr><td><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span></td><td>${esc(o.name)}</td><td>${SHAPE_LABEL[o.shape] || esc(o.shape)}</td>` +
        `<td class="n">${L}</td><td class="n">${W}</td><td class="n">${H}</td><td class="n">${fmt(o.rms, 1)}</td></tr>`;
    });
    let res = '';
    if (s.results.length) {
      res = `<details class="sfres"><summary>场景拟合自己的答案（${s.results.length} 题；最终提交只把网格物体的结果当作 Qwen 的提示）</summary><div class="scroll-x"><table><tr><th>问题</th><th>物体</th><th>方法</th><th class="n">预测</th>` +
        (split === 'val' ? '<th class="n">答案</th><th class="n">MRA</th>' : '') + '</tr>';
      s.results.forEach((r) => {
        res += `<tr><td>${esc(r.q)}</td><td>${esc(r.obj || '')}</td><td><code>${esc(r.method || '')}</code></td><td class="n">${fmtNum(r.pred)}</td>` +
          (split === 'val' ? `<td class="n">${fmtNum(r.answer)}</td><td class="n">${r.pred == null ? '<span class="muted">未答</span>' : mraBadge(r.mra)}</td>` : '') + '</tr>';
      });
      res += '</table></div></details>';
    }
    return `<h4>场景拟合的物体</h4><div><div class="scroll-x"><table><tr><th></th><th>物体</th><th>形状</th><th class="n">长 L (m)</th><th class="n">宽 W (m)</th><th class="n">高 H (m)</th><th class="n">残差 (px)</th></tr>${rows}</table></div>` +
      `<p class="muted" style="font-size:12.5px">焦距 f = ${fmtNum(s.f)} px。残差是拟合框与跟踪框的均方根误差。</p>${res}</div>`;
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
    function label(x, y, txt, col, fixed) {
      ctx.font = '600 11.5px system-ui, sans-serif';
      const w = ctx.measureText(txt).width + 8;
      x = Math.max(0, Math.min(x, W * k - w)); y = Math.max(15, y);
      if (!fixed) {
        for (let n = 0; n < 30 && placed.some((r) => x < r[0] + r[2] && r[0] < x + w && y - 15 < r[1] && r[1] - 15 < y); n++) y += 16;
        placed.push([x, y, w]);
      }
      ctx.fillStyle = 'rgba(0,0,0,.62)'; ctx.fillRect(x, y - 15, w, 16);
      ctx.fillStyle = col; ctx.fillText(txt, x + 4, y - 3);
    }
    function arrow(x0, y0, x1, y1, col) {
      const a = Math.atan2(y1 - y0, x1 - x0);
      ctx.strokeStyle = ctx.fillStyle = col; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 9 * Math.cos(a - 0.45), y1 - 9 * Math.sin(a - 0.45)); ctx.lineTo(x1 - 9 * Math.cos(a + 0.45), y1 - 9 * Math.sin(a + 0.45));
      ctx.closePath(); ctx.fill();
    }
    function draw(t) {
      tCur = t; placed = [];
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cw = W * k, ch = H * k;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, cw, ch);
      if (mode === 'video' && video && video.readyState >= 2) ctx.drawImage(video, 0, 0, cw, ch);
      else if (bg.complete && bg.naturalWidth) ctx.drawImage(bg, 0, 0, cw, ch);
      ctx.fillStyle = 'rgba(0,0,0,.18)'; ctx.fillRect(0, 0, cw, ch);
      const X = (p) => p[0] * k, Y = (p) => p[1] * k;
      const half = 0.5 / fps + 1e-6;
      s.layers.forEach((L, i) => {
        const col = PALETTE[i % PALETTE.length];
        // tracked centroids
        if (rawBox.checked) {
          ctx.fillStyle = col; ctx.globalAlpha = 0.45;
          L.c.forEach((c) => { ctx.beginPath(); ctx.arc(X(c), Y(c), 1.6, 0, 2 * Math.PI); ctx.fill(); });
          ctx.globalAlpha = 1;
        }
        // model trajectories per segment
        ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash([7, 4]);
        L.segments.forEach((sg) => {
          if (sg.model === 'static') {
            const p = sg.p.p; ctx.setLineDash([]); ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.moveTo(X(p) - 5, Y(p) - 5); ctx.lineTo(X(p) + 5, Y(p) + 5); ctx.moveTo(X(p) + 5, Y(p) - 5); ctx.lineTo(X(p) - 5, Y(p) + 5); ctx.stroke();
            ctx.setLineDash([7, 4]); ctx.lineWidth = 2.2;
            return;
          }
          ctx.beginPath();
          const nS = 80;
          for (let j = 0; j <= nS; j++) {
            const p = segPos(sg, sg.t0 + ((sg.t1 - sg.t0) * j) / nS);
            if (j) ctx.lineTo(X(p), Y(p)); else ctx.moveTo(X(p), Y(p));
          }
          ctx.stroke();
          // segment ends: hollow squares, so splits (collisions, stops) are visible on a straight path
          ctx.setLineDash([]); ctx.lineWidth = 1.5;
          [sg.t0, sg.t1].forEach((tb) => { const q = segPos(sg, tb); ctx.strokeRect(X(q) - 3.5, Y(q) - 3.5, 7, 7); });
          ctx.setLineDash([7, 4]); ctx.lineWidth = 2.2;
          if (sg.model === 'circle') {
            const c = sg.p.center; ctx.setLineDash([]); ctx.lineWidth = 1.4;
            ctx.beginPath(); ctx.moveTo(X(c) - 6, Y(c)); ctx.lineTo(X(c) + 6, Y(c)); ctx.moveTo(X(c), Y(c) - 6); ctx.lineTo(X(c), Y(c) + 6); ctx.stroke();
            ctx.setLineDash([7, 4]); ctx.lineWidth = 2.2;
          }
        });
        ctx.setLineDash([]);
        // current box (nearest tracked frame within half a frame)
        let bi = -1, bd = Infinity;
        L.t.forEach((tt, j) => { const d = Math.abs(tt - t); if (d < bd) { bd = d; bi = j; } });
        const sg = L.segments.find((x) => t >= x.t0 - half && t <= x.t1 + half);
        const tag = `${L.name} · ` + (sg ? `${sg.model}, rms ${fmt(sg.rms, 1)} px` : '不可见');
        if (bi >= 0 && bd <= half) {
          const b = L.box[bi];
          ctx.strokeStyle = col; ctx.lineWidth = 2;
          ctx.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
          label(b[0] * k, b[1] * k - 2, tag, col);
        } else if (sg) {
          const p = segPos(sg, t); label(X(p) + 6, Y(p) - 6, tag, col);
        }
        if (sg) {
          const p = segPos(sg, Math.min(Math.max(t, sg.t0), sg.t1));
          ctx.fillStyle = col; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(X(p), Y(p), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        }
      });
      // scale bar(s): one metric length, drawn at several image heights when the scale depends on y
      const sc = s.scale;
      if (isFinite(mPerPx(sc, H / 2))) {
        const ys = sc.model === 'loglinear_y' ? [0.3, 0.6, 0.9].map((f) => f * H) : [0.93 * H];
        const len = niceLen(Math.min(...ys.map((y) => 0.22 * W * mPerPx(sc, y))));
        ys.forEach((y) => {
          const px = (len / mPerPx(sc, y)) * k, x0 = 10, yy = y * k;
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x0 + px, yy); ctx.moveTo(x0, yy - 5); ctx.lineTo(x0, yy + 5); ctx.moveTo(x0 + px, yy - 5); ctx.lineTo(x0 + px, yy + 5); ctx.stroke();
          label(x0, yy - 7, fmtLen(len), '#fff', true);
        });
      }
      if (s.gravity_dir) {
        const g = s.gravity_dir, cx = W * k - 40, cy = 40;
        arrow(cx - g[0] * 22, cy - g[1] * 22, cx + g[0] * 22, cy + g[1] * 22, '#ffd43b');
        label(cx + 8, cy - 20, 'g', '#ffd43b', true);
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
      video.addEventListener('loadeddata', () => draw(tCur), { signal: ac.signal });
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

  function layerTables(v) {
    const s = v.scene2d, sc = s.scale, pr = s.prior;
    let rows = '';
    s.layers.forEach((L, i) => {
      const segs = L.segments.map((g) => `${MODEL_LABEL[g.model] || esc(g.model)} <span class="muted">${fmt(g.t0, 2)}–${fmt(g.t1, 2)} s · rms ${fmt(g.rms, 1)}</span>`).join('<br>');
      const ext = L.extent || {};
      rows += `<tr><td><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span></td><td>${esc(L.name)}` +
        (L.flags.length ? `<div class="known">${L.flags.map(esc).join('；')}</div>` : '') + '</td>' +
        `<td>${L.visible.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ')} s</td><td>${segs}</td>` +
        `<td class="n">${fmtNum(ext.w)} × ${fmtNum(ext.h)}</td><td class="n">${fmt(L.rms, 2)}</td></tr>`;
    });
    const scaleTxt = !sc.m_per_px_ref ? '没有可用的尺度' : sc.model === 'loglinear_y'
      ? `透视模型 m/px(y) = ${fmtNum(sc.m_per_px_ref)} · exp(−${fmtNum(sc.k_per_px)} · (y − ${fmtNum(sc.y_ref)}))`
      : `均匀 ${fmtNum(sc.m_per_px_ref)} m/px`;
    return '<h4>图层</h4><div class="scroll-x"><table><tr><th></th><th>图层（物体）</th><th>可见时段</th><th>分段运动模型</th><th class="n">尺寸 w × h (px)</th><th class="n">残差 (px)</th></tr>' +
      rows + '</table></div>' +
      `<p class="muted" style="font-size:12.5px">尺度：${scaleTxt}（${s.width}×${s.height} px 坐标）；来自 prior "${esc(pr.text)}"，在图层 "${esc(pr.layer)}" 上量得 ${fmtNum(pr.px)} px（${esc(pr.how)}）。` +
      (sc.model !== 'loglinear_y' && sc.persp_reason ? `没用透视模型：${esc(sc.persp_reason)}。` : '') +
      (sc.notes.length ? sc.notes.map(esc).join('；') + '。' : '') +
      (s.gravity_dir ? `重力方向（图像坐标）(${fmt(s.gravity_dir[0], 3)}, ${fmt(s.gravity_dir[1], 3)})。` : '') +
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
    const pos = (o, t) => add(add(o.p0, o.v0, t), o.a, 0.5 * t * t);

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
      meshes.push({ o, grp, mat });
    });

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
      meshes.forEach(({ o, mat }) => { mat.opacity = scene.background ? 0.45 : (o.shape === 'mesh' ? 0.35 : 0.8); });
      render();
    }
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

    function setTime(t) {
      meshes.forEach(({ o, grp }) => grp.position.copy(toW(pos(o, t))));
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
      video.addEventListener('loadeddata', () => { if (scene.background) render(); }, o);
      video.addEventListener('error', () => { if (ovBox) { ovBox.checked = false; setView(active === camC ? 'cam' : 'free'); } }, o);
    }
    const ro = new ResizeObserver(resize); ro.observe(box);
    resize(); tl.redraw(); if (vbar) setView("cam");
    return {
      dispose() {
        ac.abort(); tl.stop();
        ro.disconnect(); controls.dispose();
        scene.background = null; if (vtex) vtex.dispose();
        btns.forEach((b) => { b.onclick = null; b.setAttribute("aria-pressed", String(b.dataset.view === "cam")); });
        if (ovBox) { ovBox.onchange = null; ovBox.checked = false; ovBox.disabled = true; ovLabel.classList.add('off'); }
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
      (!fit || (fit === 'yes' ? !!v.scene : v.dim === 3 && !v.scene))).map((v) => {
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
