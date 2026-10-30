// videos.html: per-video question browser. Data: data/videos.json (built by tools/build_videos.py).
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
    $('val-summary').hidden = false;
    const all = val.videos.flatMap((v) => v.questions);
    const m = macro(all);
    // the headline is the build script's pipeline.metrics.evaluate value; the table recomputes from the per-question MRA
    $('val-macro').textContent = fmt(val.macro != null ? val.macro : m.macro, 4);
    $('val-src-file').textContent = `${m.n} 题、${val.videos.length} 个视频`;
    // per answer source: how many questions took their final prediction from it and their MRA
    const SRCS = Object.entries(SRC_LABEL).filter(([k]) => all.some((q) => q.src === k));
    let h = '<tr><th>类别</th><th class="n">题数</th><th class="n">MRA</th>' + SRCS.map(([, l]) => `<th class="n">${l} 题数</th><th class="n">${l} MRA</th>`).join('') + '</tr>';
    const row = (label, qs, strong) => {
      const s = (x) => (strong ? `<b>${x}</b>` : x);
      return `<tr><td>${s(label)}</td><td class="n">${qs.length}</td><td class="n">${s(fmt(mean(qs.map((q) => q.mra))))}</td>` +
        SRCS.map(([k]) => { const g = qs.filter((q) => q.src === k); return `<td class="n">${g.length}</td><td class="n">${g.length ? fmt(mean(g.map((q) => q.mra))) : '—'}</td>`; }).join('') + '</tr>';
    };
    CATS.forEach((c) => { h += row(c, all.filter((q) => q.cat === c)); });
    h += `<tr><td><b>macro</b></td><td class="n">${all.length}</td><td class="n"><b>${fmt(m.macro, 4)}</b></td><td colspan="${2 * SRCS.length}"></td></tr>`;
    $('val-sum').innerHTML = h;
  }
  function testSummary(test) {
    const all = test.videos.flatMap((v) => v.questions);
    const fit = test.videos.filter(hasScene).length;
    $('test-src-file').innerHTML = `${all.length} 题、${test.videos.length} 个视频的预测（按行号和 <code>load_split('test')</code> 对齐）。测试集没有答案，不显示准确度。${fit} / ${test.videos.length} 个视频有 3D 场景。`;
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
  // per-frame visible-count check (scene json "count_check", written by qp-tools): n_video vs n_scene as a strip
  function countStrip(cc) {
    if (!cc) return '';
    let rows = [];
    if (Array.isArray(cc)) rows = cc.map((x) => (Array.isArray(x) ? { t: x[0], nv: x[1], ns: x[2] } : { t: x.t, nv: x.n_video, ns: x.n_scene }));
    else if (cc.t && cc.n_video && cc.n_scene) rows = cc.t.map((t, i) => ({ t, nv: cc.n_video[i], ns: cc.n_scene[i] }));
    else if (Array.isArray(cc.frames)) rows = cc.frames.map((x) => ({ t: x.t, nv: x.n_video, ns: x.n_scene, vo: x.video_only, so: x.scene_only }));
    if (!rows.length) return '';
    const bad = rows.filter((r) => r.nv !== r.ns).length;
    return `<div class="cstrip-wrap"><b>可见数量一致</b> <span class="muted">${rows.length - bad}/${rows.length} 帧一致</span><div class="cstrip">` +
      rows.map((r) => `<span class="${r.nv === r.ns ? 'ok' : 'bad'}" title="${esc(`t = ${fmt(r.t, 2)} s：视频 ${r.nv} / 场景 ${r.ns}` +
        (r.vo && r.vo.length ? `\n只在视频里：${r.vo.join('、')}` : '') + (r.so && r.so.length ? `\n只在场景里：${r.so.join('、')}` : ''))}"></span>`).join('') + '</div></div>';
  }
  function videoCard(v, qs, split) {
    const d = document.createElement('details');
    d.className = hasScene(v) ? 'vid' : 'vid no-scene';
    d.dataset.id = v.id;
    const m = split === 'val' ? mean(qs.map((q) => q.mra)) : NaN;
    const thumb = v.thumb ? `<img class="thumb" loading="lazy" alt="" src="${esc(v.thumb)}">` : '<span class="thumb"></span>';
    d.innerHTML = `<summary>${thumb}<div class="vmeta"><div class="vid-id">${esc(v.id)}</div>` +
      `<div class="sub">${esc(v.source)} · ${esc(v.video_type)} · ${v.fps} fps${v.duration ? ' · ' + fmt(v.duration, 1) + ' s' : ''} · ${qs.length} 题 ` +
      v.cats.map((c) => `<span class="tag">${c}</span>`).join('') + '</div></div>' +
      '<div class="vright"><span class="vright-row">' +
      (split === 'val' ? `<span title="该视频（当前筛选下）的平均 MRA">${mraBadge(m)}</span>` : '') + '</span>' +
      (v.run_at ? `<span class="runat" title="最近一次运行这个视频的时间（洛杉矶时间）">${esc(v.run_at)}</span>` : '') + '</div></summary>';
    let built = false, viewer = null, video = null;
    const original = v;
    const groups = (v.groups || []).filter((g) => qs.some((q) => q.group === g.group));
    const unavailable = qs.some((q) => q.group == null);
    let selectedGroup = groups.length ? groups[0].group : '__unavailable__';
    let activeQuestions = qs;
    const onToggle = () => {
      // test-set scenes live in one file per video (data/scenes/test/<id>.json) and are fetched when the card opens
      if (d.open && v.scene_ref && !v._loaded) {
        fetch(v.scene_ref).then((r) => (r.ok ? r.json() : {})).catch(() => ({})).then((j) => {
          v.scene = j.scene || null; v._loaded = true;
          if (d.open) onToggle();
        });
        return;
      }
      if (d.open && !built) {
        built = true;
        let selector = '', critique = '';
        if (original.harness) {
          const group = groups.find((g) => g.group === selectedGroup);
          v = { ...original, scene: (group && group.scene) || null };
          activeQuestions = qs.filter((q) => group ? q.group === group.group : q.group == null);
          selector = '<div class="controls"><label>独立 prior 组 <select class="group-select">' +
            groups.map((g) => `<option value="${esc(g.group)}"${g.group === selectedGroup ? ' selected' : ''}>组 ${esc(g.group)} · ${esc((g.prior || g.depth_info || '无 prior').slice(0, 85))}</option>`).join('') +
            (unavailable ? `<option value="__unavailable__"${!group ? ' selected' : ''}>scene_id 无法绑定的题目</option>` : '') + '</select></label></div>' +
            (group ? `<div class="known-shared">${[group.prior, group.depth_info].filter(Boolean).map(esc).join('\n')}\nscene_id: ${esc(group.scene_id)}</div>` :
              '<p class="fwarn">scene_id 缺失或不唯一；不选择其他组代替。</p>');
          critique = group ? harnessCritique(group.critique) : '';
        }
        const body = document.createElement('div');
        body.className = 'vbody';
        body.innerHTML = selector + mediaBlock(v, split) + questionTable(activeQuestions, split) + critique + sceneGraphTree(original.sgraph) + sceneTables(v, split);
        d.appendChild(body);
        video = body.querySelector('video');
        tqMarks(body, v, activeQuestions);
        const groupSelect = body.querySelector('.group-select');
        if (groupSelect) groupSelect.addEventListener('change', (e) => {
          selectedGroup = e.target.value;
          if (viewer) { viewer.then((x) => x && x.dispose()); viewer = null; }
          if (video) { video.pause(); video.removeAttribute('src'); video.load(); video = null; }
          body.remove(); built = false; onToggle();
        });
        body.addEventListener('click', (e) => {
          const b = e.target.closest('.tq-btn');
          if (b) jumpTo(body, v, b.dataset.t === '' ? null : Number(b.dataset.t));
        });
      }
      // the <video> only holds a src while its card is open, so collapsed cards do not keep connections or buffers
      if (video) {
        // preload="auto" + a tiny seek decodes the first frame right away, so the overlay canvas
        // can draw it before the user presses play (with metadata-only preload the canvas stays black)
        if (d.open && !video.getAttribute('src')) {
          video.preload = 'auto'; video.src = v.video;
          video.addEventListener('loadedmetadata', () => { if (video.readyState < 2) video.currentTime = 0.001; }, { once: true });
        }
        if (!d.open) { video.pause(); video.removeAttribute('src'); video.load(); }
      }
      if (v.scene) {
        if (d.open && !viewer) viewer = startViewer(d.querySelector('.viewer'), d.querySelector('.timebar'), v, video);
        if (!d.open && viewer) { viewer.then((x) => x && x.dispose()); viewer = null; }
      }
    };
    d.addEventListener('toggle', onToggle);
    return d;
  }

  // ---------- question time t_q: jump button, alignment check, ticks on the time axis ----------
  const hasAlign = (qs) => qs.some((q) => q.align);
  function tqHead(qs) {
    return '<th title="把原视频和 3D 视图跳到题目问的时刻 t_q（没写时刻的题跳到被问物体可见区间的中点），并切到相机视角、叠加原视频">问到的帧</th>' +
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
  // scene time span of a card's viewer: the video duration
  function sceneT(v) { return v.duration || 2; }
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

  function harnessQuestionTable(qs) {
    const rows = qs.map((q) => {
      const binding = q.binding_status === 'matched';
      const images = binding ? (q.runtime_overlays || []).filter((im) => im.scene_id === q.scene_id) : [];
      const previews = images.map((im) => im.path ? `<figure class="runtime-frame"><a href="${esc(im.path)}" target="_blank" rel="noopener">` +
        `<img loading="lazy" src="${esc(im.path)}" alt="本题 scene_id 的 runtime 叠加图，t=${esc(im.t)}s"></a>` +
        `<figcaption>t=${fmt(im.t, 3)} s · ${esc(im.renderer)}</figcaption></figure>` :
        `<p class="fwarn">t=${fmt(im.t, 3)} s 图不可用：${esc(im.unavailable)}</p>`).join('');
      const info = binding ? `<details class="runtime-diagnostics"><summary>本题诊断与来源</summary><pre>${esc(JSON.stringify({
        scene_id: q.scene_id, row: q.row, method: q.method, source: q.src, status: q.status,
        scene_conf: q.scene_conf, confidence_kind: q.confidence_kind,
        execution_status: q.execution_status, measurement: q.measurement, measurement_id: q.measurement_id,
        measurement_record: q.measurement_record, quality: q.quality, provenance: q.provenance
      }, null, 2))}</pre></details>` : `<p class="fwarn">图与场景诊断 unavailable：${esc(q.binding_error)}</p>`;
      const independent = q.src === 'measurement' || ((q.measurement || {}).reasons || []).includes('direct_observation_not_rendered_geometry');
      return `<tr><td>${esc(q.cat)}<br><span class="muted">row ${q.row}</span></td>` +
        `<td class="q">${esc(q.q)}<div class="known">prior: ${[q.prior, q.depth].filter(Boolean).map(esc).join('\n')}</div>` +
        `<div class="scene-id">scene_id: <code>${esc(q.scene_id || 'unavailable')}</code></div>` +
        (independent ? '<p class="muted">此量来自独立测量记录，不是渲染几何；图像对齐本身不能证明该数值。</p>' : '') +
        `<div class="runtime-frames">${previews}</div>${info}</td>` +
        `<td class="n">${fmtNum(q.answer)}</td><td class="n pred">${fmtNum(q.pred)}</td>` +
        `<td>${esc(SRC_LABEL[q.src] || q.src)}<br><code>${esc(q.status || 'unavailable')}</code><br>` +
        (q.src === 'prior' ? `prior 置信度未知<br>场景参考 conf ${fmt(q.scene_conf == null ? q.conf : q.scene_conf, 3)}` :
          `conf ${fmt(q.conf, 3)}`) + '</td>' +
        `<td class="n">${mraBadge(q.mra)}</td>${tqCells(qs, q)}</tr>`;
    }).join('');
    return '<h4>本组逐题答案与叠加图</h4><div class="scroll-x"><table class="qtable">' +
      '<tr><th>类别 / 行号</th><th>问题、精确场景与诊断</th><th>验证答案</th><th>最终预测</th><th>来源 / 状态</th><th>MRA</th>' +
      tqHead(qs) + `</tr>${rows}</table></div>`;
  }

  function harnessCritique(critique) {
    const iterations = (critique || {}).iterations || [];
    return `<details class="sfres"><summary>本组 agent 回路：${iterations.length} 轮，${iterations.filter((i) => i.accepted).length} 轮接受；${esc(((critique || {}).status || {}).stop_reason || '未记录')}</summary>` +
      '<div class="scroll-x"><table><tr><th>轮次</th><th>诊断</th><th>动作</th><th>结果 / 停止原因</th></tr>' +
      iterations.map((i) => `<tr><td>${i.iter}</td><td>${esc(i.diagnosis || '')}</td><td><pre>${esc(JSON.stringify(i.tool_calls || [], null, 2))}</pre></td>` +
        `<td>${i.accepted ? '接受' : '未接受'} · ${esc(i.why || '')}<br>${esc(i.execution_status || '执行状态未记录')}` +
        `${i.decision_status ? ' / ' + esc(i.decision_status) : ''}${i.budget_exhausted ? ' / 观察预算耗尽' : ''}` +
        `<details><summary>工具执行记录</summary><pre>${esc(JSON.stringify({observations: i.observations, results: i.results, rejected_calls: i.rejected_calls,
          deferred_calls: i.deferred_calls, failure_stage: i.failure_stage,
          model_policy_results: (i.model_policy_results || []).map(({ artifact, ...r }) => r),
          control_before: i.control_before, control_after: i.control_after}, null, 2))}</pre></details></td></tr>`).join('') +
      '</table></div></details>';
  }

  function questionTable(qs, split) {
    if (qs.some((q) => q.harness)) return harnessQuestionTable(qs);
    if (!qs.length) return '<p class="muted">当前组没有匹配筛选条件的题目。</p>';
    const kn = (q) => (q.prior || q.depth ? `已知：${[q.prior, q.depth].filter(Boolean).map(esc).join('\n')}` : '');
    // when every question of the video has the same prior (incl. the distance anchor), show it once above the table
    const shared = qs.every((q) => q.prior === qs[0].prior && q.depth === qs[0].depth);
    const known = (q) => (shared ? '' : kn(q));
    let h = '<h4>问题</h4>' + (shared && kn(qs[0]) ? `<div class="known-shared">${kn(qs[0])}</div>` : '') +
      '<div class="scroll-x"><table class="qtable"><tr><th>类别</th><th>' + (shared ? '问题' : '问题与已知条件') + '</th>';
    if (split === 'val') {
      h += '<th class="n">答案</th><th class="n">预测</th><th>来源</th><th class="n">MRA</th>' + tqHead(qs) + '</tr>';
      qs.forEach((q) => {
        h += `<tr><td><span class="tag">${q.cat}</span><span class="muted">${esc(q.type)}</span></td>` +
          `<td class="q">${esc(q.q)}<div class="known">${known(q)}</div></td>` +
          `<td class="n">${fmtNum(q.answer)}</td><td class="n pred">${fmtNum(q.pred)}</td><td>${esc(SRC_LABEL[q.src] || q.src || '—')}</td>` +
          `<td class="n">${mraBadge(q.mra)}</td>` + tqCells(qs, q) + '</tr>';
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

  // top of an opened card: original video (left) and the 3D scene view (right; a placeholder until the video has one)
  function mediaBlock(v, split) {
    let vid;
    if (v.video) {
      const note = v.video_from !== split ? `<p class="muted vnote">验证集的原文件是 MPEG-4 Part 2 编码，浏览器放不了；这里放的是测试集里同一视频的 H.264 版本（帧数相同，分辨率更高）。<a href="${esc(v.orig)}">下载验证集原文件</a></p>` : '';
      vid = `<h4>原视频</h4><video class="orig" controls muted loop playsinline preload="metadata"></video>${note}`;
    } else {
      vid = `<h4>原视频</h4><div class="noscene">这个视频的编码是 MPEG-4 Part 2（<code>${esc(v.codec)}</code>），浏览器放不了，也没有可替代的 H.264 版本。<a href="${esc(v.orig)}">下载原文件</a></div>`;
    }
    let scene = '';
    if (v.scene) {
      scene = '<div><h4>3D 场景</h4>' +
        '<div class="viewbar"><span class="seg" role="group" aria-label="视角"><button type="button" data-view="free" aria-pressed="false">自由视角</button>' +
        '<button type="button" data-view="cam" aria-pressed="true" title="从原视频相机的位置和视野看，可以和左边的视频逐帧对比">相机视角</button></span>' +
        (v.video ? '<label class="off" title="相机视角下把原视频画面垫在渲染下面"><input type="checkbox" class="overlay" checked disabled> 叠加原视频</label>' : '') +
        (v.scene.f ? '' : '<span class="fwarn" title="Physics-guided Fitting 没有给出焦距，用图像宽度代替">焦距为默认值（f = 图像宽度）</span>') + '</div>' +
        '<div class="viewer"><div class="msg">加载 three.js …</div></div>' +
        '<div class="timebar"><button type="button" class="play">播放</button><span class="tq-track"><input type="range" min="0" max="1000" value="0" aria-label="时间"><span class="tq-marks"></span></span><span class="tlabel">t = 0.00 s</span></div>' +
        '<p class="muted" style="font-size:12.5px;margin:6px 0 0">灰色网格 = 拟合出的地面；黑色线框 = 相机（原点）与视野；彩色 = 物体及其匀加速轨迹（虚线）。物体只在视频里跟踪到它的时段画出：相机视角下其余时段不画，自由视角下画成淡色轮廓。' +
        '左上角计数：视频可见 = 当前帧跟踪有框的物体数，场景视野内 = 拟合位置投影在原相机画面内的物体数，两者不等时标红。自由视角下拖动旋转，滚轮缩放；相机视角固定在原视频相机处，画面比例和原视频相同。' +
        (v.video ? '时间轴和左边的视频联动。' : '') + '</p></div>';
    }
    return (v.harness && v.scene ? '<p class="muted">交互 3D 视图是网页里的简化投影，只作近似浏览；逐题检查看下方由流水线渲染的叠加图。独立 measurement 不等于渲染几何。</p>' : '') +
      `<div class="media"><div>${vid}</div>${scene}</div>`;
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

  // one table per video: inventory + scene graph + 3D fit, one row per object
  const ROLE_LABEL = { object: '物体', structure: '结构', class: '类别' };
  const PRES_LABEL = { visible: '在场', partial: '部分在场', absent: '不在场' };
  function objectTable(v, order) {
    const t = v.objtab;
    const ivs = (vis) => (vis && vis.length ? vis.map((iv) => `${fmt(iv[0], 2)}–${fmt(iv[1], 2)}`).join(', ') + ' s' : '');
    const size = (sc) => {
      if (sc.size) return sc.size.map(([k, x, src]) => {
        const dimText = `${k} ${fmtNum(x)} m`;
        return src === 'bound' ? `<span class="dim-bound" title="看不见、靠先验">${dimText}<sup>看不见</sup></span>` : src === 'prior' ? `${dimText}<sup class="dim-prior">先验</sup>` : dimText;
      }).join(' × ');
      return '';
    };
    let rows = '';
    t.rows.forEach((r) => {
      const scs = r.scene && r.scene.length ? r.scene : [null];
      scs.forEach((sc, k) => {
        const idx = sc ? order.indexOf(sc.name) : -1;
        const sw = idx >= 0 ? `<span class="swatch" style="background:${PALETTE[idx % PALETTE.length]}"></span>` : '';
        const first = k === 0;
        const name = first ? `${esc(r.name)}${r.aliases && r.aliases.length ? `<div class="muted" style="font-size:12px">别名：${r.aliases.map(esc).join('；')}</div>` : ''}` +
          (sc && sc.name !== r.name ? `<div class="muted" style="font-size:12px">场景里：${esc(sc.name)}</div>` : '') : `<span class="muted">↳ ${esc(sc.name)}</span>`;
        const pres = r.present ? `${PRES_LABEL[r.present] || esc(r.present)}${r.presence_conf != null ? ` <span class="muted">${fmt(r.presence_conf, 2)}</span>` : ''}` : '';
        const role = r.unmatched ? '' : `${ROLE_LABEL[r.role] || esc(r.role || '')}${r.reference_only ? '<div class="muted" style="font-size:12px">仅作参考</div>' : ''}${r.part_of ? `<div class="muted" style="font-size:12px">属于 ${esc(r.part_of)}</div>` : ''}`;
        const flags = sc ? (sc.flags || []).map((f) => `<span class="sigtag">${esc(f)}</span>`).join(' ') : '';
        rows += `<tr${r.unmatched ? ' class="obj-unmatched"' : ''}><td>${sw}</td><td>${name}${r.unmatched ? '<div class="fwarn" style="font-size:12px">清单里没有这个物体</div>' : ''}</td>` +
          `<td>${first ? role : ''}</td><td class="n">${first && r.count != null ? r.count : ''}</td><td>${first ? pres : ''}</td>` +
          `<td>${first && r.shape ? esc(r.shape) + (r.thin ? '（细长）' : '') : ''}</td>` +
          `<td style="font-size:13px">${first ? (r.relations || []).map(esc).join('<br>') : ''}</td>` +
          `<td>${sc ? size(sc) : `<span class="muted">未重建（${esc(r.not_built || '')}）</span>`}</td>` +
          `<td style="font-size:13px">${sc ? [sc.support ? esc(sc.support) : '', ivs(sc.visible)].filter(Boolean).join('<br>') : ''}</td>` +
          `<td style="font-size:13px">${sc ? esc(sc.motion || '') : ''}</td><td>${flags}</td></tr>`;
      });
    });
    return `<h4>物体（清单 + scene graph + Physics-guided Fitting）</h4><div class="scroll-x"><table class="objtab"><tr><th></th><th>物体（别名）</th><th>角色</th><th class="n">数量</th><th>在场性</th><th>形状提示</th><th>scene graph 关系</th>` +
      `<th>拟合尺寸</th><th>支撑面 / 可见区间</th><th>运动 / 残差</th><th>自检标记</th></tr>${rows}</table></div>` +
      (t.unmatched && t.unmatched.length ? `<p class="fwarn" style="font-size:12.5px">场景里有、清单里没有的物体（按名字匹配不上）：${t.unmatched.map(esc).join('、')}</p>`
        : '<p class="muted" style="font-size:12.5px">场景里的物体都能和清单对上。</p>');
  }

  function sceneTables(v, split) {
    if (!v.scene) return '';
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
      res = `<details class="sfres"><summary>${v.harness ? '本组保存的场景测量（最终答案见题表）' : '历史 Physics-guided Fitting 自己的答案'}（${s.results.length} 题）</summary><div class="scroll-x"><table><tr><th>问题</th><th>物体</th><th>方法</th><th class="n">预测</th>` +
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
    const objBlock = v.objtab ? objectTable(v, s.objects.map((o) => o.name))
      : `<h4>Physics-guided Fitting 的物体</h4><div class="scroll-x"><table><tr><th></th><th>物体</th><th>形状</th><th class="n">长 L (m)</th><th class="n">宽 W (m)</th><th class="n">高 H (m)</th>` +
        (hasSup ? '<th>支撑</th>' : '') + (hasVis ? '<th title="跟踪有框的时段；相机视角只在这些时段画出物体">视频里可见</th>' : '') + `<th class="n">残差 (px)</th></tr>${rows}</table></div>`;
    return countStrip(s.count_check) + graphBlocks(v.objtab ? null : s.inventory, v.sgraph ? null : s.scene_graph, { tree: v.stree, order: s.objects.map((o) => o.name) }) + objBlock + '<div>' +
      `<p class="muted" style="font-size:12.5px">焦距 f = ${fmtNum(s.f)} px` + (s.f_source ? `（来源 ${esc(s.f_source)}）` : '') + (s.main_support ? `，主支撑面 ${esc(s.main_support)}` : '') + '。残差是拟合框与跟踪框的均方根误差。' +
      (hasSrc ? '尺寸来源：不加标记的是观测到的（框、深度）；<sup class="dim-prior">先验</sup> 来自尺寸先验；<span class="dim-bound">灰色<sup>看不见</sup></span> 表示框和深度都没看到这一维，只能靠先验（取防塌缩下限）。' : '') +
      `</p>${focalVotes(s)}${res}</div>`;
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

  // ---------- scene graph (tools/build_videos.sgraph) as a nested tree ----------
  // supports at the root, an object under what it rests on / is in / hangs from / is held by; airborne objects in their
  // own group; an object that changes support appears under each one, with the time span of that relation
  const SG_REL_SHOWN = { in: 'in', attached_to: 'attached_to', above: 'above' };
  function sceneGraphTree(sg) {
    if (!sg || !sg.edges || !sg.edges.length) return '';
    const kids = new Map(), isChild = new Set();
    const span = (e) => (e.t0 == null && e.t1 == null ? '' : `${fmt(e.t0 == null ? 0 : e.t0, 2)}–${e.t1 == null ? '' : fmt(e.t1, 2)} s`);
    sg.edges.forEach((e) => {
      const parent = e.rel === 'airborne' || !e.to ? '\u0000' + e.rel : e.to;   // no target: a group named by the relation
      if (parent === e.from) return;
      if (!kids.has(parent)) kids.set(parent, []);
      const list = kids.get(parent);
      let k = list.find((x) => x.id === e.from && x.rel === e.rel);
      if (!k) { k = { id: e.from, rel: e.rel, spans: [] }; list.push(k); }
      const sp = span(e);
      if (sp) k.spans.push([e.t0 == null ? 0 : e.t0, sp]);
      if (e.to) isChild.add(e.from);
    });
    kids.forEach((l) => l.forEach((k) => k.spans.sort((a, b) => a[0] - b[0])));
    const type = Object.fromEntries((sg.nodes || []).map((n) => [n.id, n.type]));
    const grp = (id) => id.startsWith('\u0000');
    const roots = [...kids.keys()].filter((id) => !grp(id) && !isChild.has(id));
    const shown = new Set();
    const item = (id, k, path) => {
      shown.add(id);
      const rel = k && k.rel !== 'airborne' && SG_REL_SHOWN[k.rel] && !grp(path.size ? [...path].pop() : '') ? `<span class="sgrel">${esc(SG_REL_SHOWN[k.rel])}</span>` : '';
      const t = k && k.spans.length ? k.spans.map((x) => `<span class="sgt">${esc(x[1])}</span>`).join('') : '';
      const cls = grp(id) ? 'sgn air' : (type[id] && type[id] !== 'object') || !k ? 'sgn sup' : 'sgn';
      const ch = path.has(id) ? [] : (kids.get(id) || []);
      const sub = ch.length ? `<ul>${ch.map((c) => item(c.id, c, new Set([...path, id]))).join('')}</ul>` : '';
      return `<li><span class="${cls}">${grp(id) ? esc(id.slice(1)) : esc(id)}</span>${rel}${t}${sub}</li>`;
    };
    let html = roots.map((r) => item(r, null, new Set())).join('');
    [...kids.keys()].filter(grp).sort((a, b) => (a === '\u0000airborne' ? -1 : b === '\u0000airborne' ? 1 : 0)).forEach((g) => { html += item(g, null, new Set()); });
    const loose = (sg.nodes || []).filter((n) => n.type === 'object' && !shown.has(n.id));
    html += loose.map((n) => `<li><span class="sgn">${esc(n.id)}</span></li>`).join('');
    return `<h4>scene graph</h4><div class="scroll-x"><ul class="sgtree">${html}</ul></div>`;
  }

  // ---------- object inventory + scene graph ----------
  // scene graph drawn as a tree (tools/build_videos.scene_tree): supports at the top, children below their parent;
  // a second parent (on the slope, then on the desk) and contact / above are extra edges (dashed for contact / above)
  function sceneTreeSvg(tree, order) {
    const NW = 168, NH = 50, SLOT = 190, LV = 112, PAD = 14;
    const nodes = tree.nodes, byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const kids = {};
    nodes.forEach((n) => { if (n.parent && byId[n.parent] && n.parent !== n.id) (kids[n.parent] = kids[n.parent] || []).push(n.id); });
    const roots = nodes.filter((n) => !n.parent || !byId[n.parent]).map((n) => n.id);
    const pos = {}; let slot = 0, maxD = 0;
    const place = (id, d, seen) => {
      if (seen.has(id)) return; seen.add(id);
      maxD = Math.max(maxD, d);
      const ch = (kids[id] || []).filter((c) => !seen.has(c));
      if (!ch.length) { pos[id] = { x: slot++ * SLOT, d }; return; }
      ch.forEach((c) => place(c, d + 1, seen));
      const xs = ch.map((c) => pos[c] && pos[c].x).filter((x) => x != null);
      pos[id] = { x: xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : slot++ * SLOT, d };
    };
    const seen = new Set();
    roots.forEach((r) => place(r, 0, seen));
    nodes.forEach((n) => { if (!pos[n.id]) place(n.id, 0, seen); });
    const W = Math.max(slot, 1) * SLOT - (SLOT - NW) + 2 * PAD, H = (maxD + 1) * LV - (LV - NH) + 2 * PAD + 20;
    const cx = (id) => PAD + pos[id].x + NW / 2, top = (id) => PAD + pos[id].d * LV, bot = (id) => top(id) + NH;
    const wrap = (t, n) => { const w = []; let cur = ''; String(t).split(' ').forEach((x) => { if ((cur + ' ' + x).trim().length > n && cur) { w.push(cur); cur = x; } else cur = (cur + ' ' + x).trim(); }); if (cur) w.push(cur); return w.slice(0, 2).map((l, i, a) => (i === 1 && w.length > 2 ? l.slice(0, n - 1) + '…' : l)); };
    // edge label: one line per relation span ("on 0.00–0.77 s", "airborne 0.77–2.58 s", ...), stacked around (x, y)
    const lab = (x, y, t) => {
      const ls = String(t).split('；').filter(Boolean);
      const w = Math.min(Math.max(...ls.map((l) => l.length)) * 6.2 + 10, SLOT - 8), h = ls.length * 13 + 4, y0 = y - h / 2;
      return `<rect class="elab" x="${x - w / 2}" y="${y0}" width="${w}" height="${h}" rx="3"></rect>` +
        ls.map((l, i) => `<text class="etxt" x="${x}" y="${y0 + 12 + i * 13}" text-anchor="middle">${esc(l.length > 30 ? l.slice(0, 29) + '…' : l)}</text>`).join('');
    };
    let g = '';
    tree.edges.forEach((e) => {
      if (!e.to || !pos[e.from] || !pos[e.to]) return;
      if (e.tree) {
        const x1 = cx(e.to), y1 = bot(e.to), x2 = cx(e.from), y2 = top(e.from), my = (y1 + y2) / 2;
        // the label sits just above the child, so labels of siblings fanning out from one parent do not collide
        const nl = String(e.label || '').split('；').filter(Boolean).length;
        g += `<path class="edge" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}"></path>` + (e.label ? lab(x2, y2 - 6 - (nl * 13 + 4) / 2, e.label) : '');
      } else {
        // up to a shallower target: from the top of the child to the bottom of the target; same level: an arc above
        // the two boxes; down: bottom to top. The label sits at the curve's midpoint, between the levels.
        const df = pos[e.from].d, dt = pos[e.to].d, right = cx(e.to) > cx(e.from);
        let x1, y1, x2, y2, qx, qy;
        if (dt < df) { x1 = cx(e.from) + (right ? NW / 4 : -NW / 4); y1 = top(e.from); x2 = cx(e.to); y2 = bot(e.to); qx = (x1 + x2) / 2; qy = (y1 + y2) / 2; }
        else if (dt > df) { x1 = cx(e.from); y1 = bot(e.from); x2 = cx(e.to) + (right ? -NW / 4 : NW / 4); y2 = top(e.to); qx = (x1 + x2) / 2; qy = (y1 + y2) / 2; }
        else { x1 = cx(e.from); y1 = bot(e.from); x2 = cx(e.to); y2 = y1; qx = (x1 + x2) / 2; qy = bot(e.from) + 30 + Math.min(Math.abs(x2 - x1) / 8, 30); }
        const mx = 0.25 * x1 + 0.5 * qx + 0.25 * x2, my = 0.25 * y1 + 0.5 * qy + 0.25 * y2;
        g += `<path class="edge side${e.dashed ? ' dashed' : ''}" d="M${x1},${y1} Q${qx},${qy} ${x2},${y2}"></path>` + lab(mx, my, e.label);
      }
    });
    nodes.forEach((n) => {
      if (!pos[n.id]) return;
      const x = PAD + pos[n.id].x, y = top(n.id);
      const idx = n.scene ? order.indexOf(n.scene) : -1;
      const col = idx >= 0 ? PALETTE[idx % PALETTE.length] : null;
      const faint = n.absent || n.ref_only || (n.kind === 'object' && n.not_built);
      const cls = `nbox${n.kind === 'support' ? ' support' : ''}${faint ? ' faint' : ''}${n.unmatched ? ' bad' : ''}`;
      const lines = wrap(n.id, 19);
      const sub = n.kind === 'support' ? '支撑面' : [ROLE_LABEL[n.role] || n.role || '', n.count > 1 ? `×${n.count}` : '', n.absent ? '不在场' : '', n.ref_only ? '仅作参考' : '', !n.absent && !n.ref_only && n.not_built ? '未重建' : ''].filter(Boolean).join(' · ');
      g += `<g><title>${esc(n.id)}</title><rect class="${cls}" x="${x}" y="${y}" width="${NW}" height="${NH}" rx="7"${col ? ` style="stroke:${col}"` : ''}></rect>` +
        (col ? `<rect x="${x}" y="${y}" width="6" height="${NH}" rx="3" fill="${col}"></rect>` : '') +
        lines.map((l, i) => `<text class="ntxt" x="${x + 12}" y="${y + 16 + i * 14}">${esc(l)}</text>`).join('') +
        `<text class="nsub" x="${x + 12}" y="${y + NH - 6}">${esc(sub)}</text></g>`;
      const loose = tree.edges.filter((e) => e.from === n.id && !e.to);
      if (loose.length) g += lab(x + NW / 2, y + NH + 12, loose.map((e) => e.label).join('；'));
    });
    return `<div class="scroll-x stree-wrap"><svg class="stree" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="scene graph（树）">${g}</svg></div>` +
      '<p class="muted" style="font-size:12.5px">父节点在上：子节点放在（或在里面、挂在）父节点上。边上写关系和时段；实线 = 第二个父节点（换了支撑面），虚线 = contact / above。灰色虚框 = 不在场、仅作参考或未重建。色条和物体表、视图里的颜色一致。</p>';
  }

  function graphBlocks(inv, sg, treeOpt) {
    let out = '';
    if (inv && inv.length) {
      let rows = '';
      inv.forEach((o) => {
        const al = o.aliases.filter((a) => a !== o.name);
        const role = [o.track === false ? '不跟踪（场景结构）' : '跟踪', o.dyn ? '动态' : (o.dyn === false ? '静态' : ''), o.role || ''].filter(Boolean).join(' · ');
        rows += `<tr><td>${esc(o.name)}</td><td>${al.length ? al.map(esc).join('；') : '<span class="muted">—</span>'}</td>` +
          `<td>${esc(o.where.join(', '))}</td><td>${esc(role)}</td></tr>`;
      });
      out += '<h4>物体清单</h4><div class="scroll-x"><table><tr><th>物体（canonical 名）</th><th>别名</th><th>出现在</th><th>角色</th></tr>' + rows + '</table></div>';
    }
    if (sg) {
      const items = [];
      ((sg && sg.edges) || []).forEach((e) => {
        const tr = e.t0 != null && e.t1 != null ? ` <span class="muted">${fmt(e.t0, 2)}–${fmt(e.t1, 2)} s</span>` : '';
        const cf = e.conf != null ? ` <span class="muted">(conf ${fmt(e.conf, 2)})</span>` : '';
        items.push(e.to ? `${esc(e.from)} <b>${esc(e.rel)}</b> ${esc(e.to)}${cf}${tr}` : `${esc(e.from)} — <b>${esc(e.rel)}</b>${cf}${tr}`);
      });
      ((sg && sg.same) || []).forEach((x) => items.push(`${esc(x.a)} <b>${x.same ? '=' : '≠'}</b> ${esc(x.b)}` + (x.conf != null ? ` <span class="muted">(conf ${fmt(x.conf, 2)})</span>` : '')));
      if (sg && sg.support && sg.support.length) items.push(`<span class="muted">支撑面节点：${sg.support.map(esc).join('、')}</span>`);
      const list = items.length ? `<ul class="sg-list">${items.map((x) => `<li>${x}</li>`).join('')}</ul>` : '<p class="muted">没有边。</p>';
      out += `<h4>scene graph${sg && sg.source ? `<span class="muted" style="font-weight:400">（来源 ${esc(sg.source)}）</span>` : ''}</h4>` +
        (treeOpt && treeOpt.tree ? sceneTreeSvg(treeOpt.tree, treeOpt.order) + `<details class="sfres"><summary>关系文字（备查）</summary>${list}</details>` : list);
    }
    return out;
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
  // test questions without a scene answer (tools/build_videos.test_noanswer): summary + one flat table
  const SRC_ZH = { vlm: 'Qwen', geo: '几何', fit: 'Physics-guided Fitting' };
  function noAnswerSummary(na) {
    const tot = Object.values(na.by_cat).reduce((a, x) => a + x[0], 0);
    const cats = CATS.map((c) => `<tr><td>${c}</td><td class="n">${na.by_cat[c][0]}</td><td class="n">${na.by_cat[c][1]}</td><td class="n">${fmt((100 * na.by_cat[c][0]) / Math.max(na.by_cat[c][1], 1), 1)}%</td></tr>`).join('');
    const rs = na.by_reason.map(([r, n]) => `<tr><td>${esc(r)}</td><td class="n">${n}</td></tr>`).join('');
    return `<div class="noans-sum"><div><b>没有答案：${tot} / ${na.total} 题（${fmt((100 * tot) / na.total, 1)}%）</b><div class="scroll-x"><table><tr><th>类别</th><th class="n">没有答案</th><th class="n">总题数</th><th class="n">占比</th></tr>${cats}</table></div></div>` +
      `<div><b>原因分布</b><div class="scroll-x"><table><tr><th>原因</th><th class="n">题数</th></tr>${rs}</table></div></div></div>` +
      `<p class="muted" style="font-size:12.5px">口径："没有答案" = 3D 场景没给出可用值的题：Physics-guided Fitting 没有给出该题的值，或整个视频还没有 3D 场景。"预测值"取测试集预测文件，来源列取各题的 source（vlm = 模型估计，geo = 几何）；来源列和"场景有没有值"不是一回事，这里的筛选按场景有没有值判断。</p>`;
  }
  function renderNoAnswer() {
    const na = DATA.test.noans, cat = $('t-cat').value, needle = $('t-q').value.trim().toLowerCase();
    const rows = na.rows.filter((x) => (!cat || x.cat === cat) && (!needle || x.vid.toLowerCase().includes(needle) || x.q.toLowerCase().includes(needle)));
    $('t-status').textContent = `${rows.length} 题（当前筛选下）。点视频名跳到它的卡片。`;
    const list = $('test-list');
    list.innerHTML = '<div class="scroll-x"><table class="noans-tab"><tr><th>视频</th><th class="n">行号</th><th>类别</th><th>问题</th><th class="n">预测值</th><th>来源</th><th>没有场景值的原因</th></tr>' +
      rows.map((x) => `<tr><td><a href="#test/${esc(x.vid)}" class="na-link" data-vid="${esc(x.vid)}">${esc(x.vid)}</a></td><td class="n">${x.row}</td><td><span class="tag">${x.cat}</span></td><td>${esc(x.q)}</td>` +
        `<td class="n">${fmtNum(x.pred)}</td><td>${esc(SRC_ZH[x.src] || x.src || '—')}</td><td>${esc(x.reason)}${x.detail ? `<div class="muted" style="font-size:12px">${esc(x.detail)}</div>` : ''}</td></tr>`).join('') + '</table></div>';
    list.querySelectorAll('.na-link').forEach((a) => a.addEventListener('click', (e) => {
      e.preventDefault(); $('t-noans').checked = false; $('t-q').value = a.dataset.vid; renderTest();
      const d = list.querySelector(`details[data-id="${CSS.escape(a.dataset.vid)}"]`); if (d) { d.open = true; d.scrollIntoView(); }
    }));
  }
  function renderTest() {
    const na = DATA.test.noans;
    $('t-noans-sum').innerHTML = na && $('t-noans').checked ? noAnswerSummary(na) : '';
    if (na && $('t-noans').checked) { renderNoAnswer(); return; }
    const cat = $('t-cat').value, vsrc = $('t-vsrc').value, fit = $('t-fit').value, sort = $('t-sort').value;
    const needle = $('t-q').value.trim().toLowerCase();
    let items = DATA.test.videos.filter((v) => (!vsrc || v.source === vsrc) &&
      (!fit || (fit === 'yes' ? hasScene(v) : !hasScene(v)))).map((v) => {
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
    ['t-cat', 't-vsrc', 't-fit', 't-sort', 't-noans'].forEach((id) => $(id).addEventListener('change', renderTest));
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
