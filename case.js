// One question walked through from the video to the answer. Every number comes from data/case/<unit>/case.json,
// written by tools/build_case.py from the unit's products. English is written here; case-text.js holds the Chinese
// (CASE_ZH, keyed by the English string, as i18n.js does for the other pages).
(function () {
  'use strict';
  const UNIT = new URLSearchParams(location.search).get('unit') || 'captured_0021x_q1';
  const DIR = `data/case/${UNIT}/`;
  const root = document.getElementById('case');
  let C = null;

  const zh = () => window.I18N && window.I18N.lang === 'zh';
  const fill = (s, p) => (p ? s.replace(/\{(\w+)\}/g, (m, k) => (k in p ? p[k] : m)) : s);
  function t(s, p) {
    if (zh()) {
      const z = (window.CASE_ZH || {})[s];
      if (z != null) return fill(z, p);
    }
    return fill(s, p);
  }
  // a model's own words: shown as written; the Chinese of this case's quotes is in CASE_QUOTES_ZH
  function q(s) {
    if (s == null) return '';
    const z = zh() ? (window.CASE_QUOTES_ZH || {})[s] : null;
    return esc(z != null ? z : s);
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
  const num = (v, d) => (v == null ? '' : Number(v).toFixed(d));
  const sec = (v) => (v >= 100 ? Math.round(v) : Number(v).toFixed(1)) + ' s';
  const sci = (v) => (v >= 1e5 ? Number(v).toExponential(2).replace('e+', ' × 10^').replace(/\^(\d+)/, '<sup>$1</sup>') : num(v, 1));
  const deg = (r) => (r * 180 / Math.PI);

  // ---- plain names --------------------------------------------------------------------------------------------
  function obj(id) { return t('the ' + C.names[id]); }
  function sw(id) { return `<span class="sw" style="background:${C.colours[id]}"></span>`; }
  const ORD = ['first', 'second', 'third', 'fourth', 'fifth'];
  function stretch(k) { return t(`the ${ORD[k]} stretch`); }
  function lawText(st) {
    const tgt = st.target || st.to;
    if (st.law === 'static') return tgt ? t('lies still on {x}', { x: obj(tgt) }) : t('stands still');
    if (st.law === 'rolling') return t('rolls on {x}, slowing at a steady rate', { x: obj(tgt) });
    if (st.law === 'ballistic') return t('flies freely, pulled down by gravity');
    return st.law;
  }
  function pname(key) {
    const cam = {
      'cam.f': 'focal length', 'cam.pitch': 'camera tilt (up / down)', 'cam.roll': 'camera roll (sideways tilt)',
      'cam.cx': 'image centre, across', 'cam.cy': 'image centre, down', 'cam.position': 'camera position',
      'cam.yaw': 'camera heading', 'g': 'gravity',
    };
    if (cam[key]) return t(cam[key]);
    const m = key.match(/^(\w+?)\.(?:s(\d+)\.)?(\w+)$/);
    if (!m) return key;
    const [, o, s, p] = m;
    const sz = { D: 'diameter of {o}', L: 'length of {o}', W: 'width of {o}', H: 'height of {o}' };
    if (sz[p]) return t(sz[p], { o: obj(o) });
    const k = Number(s);
    const seg = { p: 'position of {o} in {s}', p0: 'position of {o} where {s} begins', v0: 'velocity of {o} where {s} begins',
      decel: 'slow-down rate of {o} in {s}', yaw: 'heading of {o} in {s}' };
    return seg[p] ? t(seg[p], { o: obj(o), s: stretch(k) }) : key;
  }
  function source(s) {
    return t({ prior: 'the prior', convention: 'a fixed convention', fit: 'the fit', vlm: 'the model' }[s] || s);
  }
  // a value with its unit, vectors as forward / left / up from the camera
  function val(v, unit, key) {
    if (v == null) return '';
    if (unit === 'rad') return `${num(deg(v), 1)}°`;
    if (Array.isArray(v)) {
      if (v.every((x) => x == null)) return '';
      const u = unit === 'm/s' ? 'm/s' : 'm';
      const lab = [t('forward'), t('left'), t('up')];
      return v.map((x, i) => `${lab[i]} ${x == null ? '?' : num(x, 3)}`).join(', ') + ' ' + u;
    }
    if (unit === 'px') return `${num(v, 1)} px`;
    if (unit === 'm' && key && /\.D$/.test(key)) return `${num(v * 1000, 0)} mm`;
    return `${num(v, 3)} ${unit === 'm/s^2' ? 'm/s²' : unit}`;
  }
  function sigma(v, unit) {
    if (v == null) return '';
    if (Array.isArray(v)) return v.every((x) => x == null) ? '' : '± (' + v.map((x) => (x == null ? '?' : num(x, 3))).join(', ') + ')';
    if (unit === 'rad') return `± ${num(deg(v), 1)}°`;
    if (unit === 'px') return `± ${num(v, 1)} px`;
    return `± ${num(v, 3)}`;
  }
  function why(w) {
    if (!w) return '';
    if (w === 'no observation') return t('nothing in the video shows it: the table has no track');
    if (/heading of a sphere/.test(w)) return t('a ball looks the same from every heading');
    if (/no effect on any fitted frame/.test(w)) return t('one part of it changes nothing the fit can see');
    const m = w.match(/sigma ([\d.]+) natural units > ([\d.]+)/);
    if (m) return t('too uncertain: its spread is {a} of its natural size, more than {b}', { a: m[1], b: m[2] });
    return esc(w);
  }

  // ---- blocks ---------------------------------------------------------------------------------------------------
  function block(n, title, body, time) {
    const tm = time ? `<div class="cost">${time}</div>` : '';
    return `<section class="cstep" id="step${n}"><h2><span class="n">${n}</span>${title}</h2>${tm}${body}</section>`;
  }
  function cost(seconds, calls) {
    const parts = [];
    if (seconds != null) parts.push(`<span><b>${sec(seconds)}</b></span>`);
    if (calls != null) parts.push(`<span>${t(calls === 1 ? '{n} model call' : '{n} model calls', { n: calls })}</span>`);
    return parts.join('');
  }
  function legend(items) {
    return `<div class="lg">${items.map(([cls, txt, style]) => `<span><i class="${cls}" style="${style || ''}"></i>${txt}</span>`).join('')}</div>`;
  }
  const ballId = () => C.plan.query.object;
  const ballCol = () => C.colours[ballId()];

  function b1() {
    const v = C.video;
    const st = C.statements.map((s) => `<li>${esc(s.raw)}</li>`).join('');
    return block(1, t('The input'), `
      <div class="two">
        <figure class="vid"><video controls muted loop playsinline preload="metadata">${v.files.map((f) => `<source src="${DIR}${f}" type="video/${f.split('.').pop()}">`).join('')}</video></figure>
        <div>
          <p>${t('A clip of {n} frames, {fps} frames per second, so {d} seconds long, {w} × {h} pixels.', { n: v.n_frames, fps: v.fps, d: num(v.n_frames / v.fps, 2), w: v.width, h: v.height })}</p>
          <h3>${t('The question')}</h3>
          <p class="quote">${esc(C.question)}</p>
          <h3>${t('What we are told (the prior)')}</h3>
          <ol class="stmts">${st}</ol>
          <p>${t('Nothing else is given: no camera settings, no size of the table.')}</p>
        </div>
      </div>`);
  }

  function b2() {
    const o = C.objects.list.map((x) => `<li>${sw(C.plan.query.object)}<b>${q(x.name)}</b> · ${t('also called')} ${x.aliases.map(esc).join(', ')} · ${t('shape')}: ${t(x.shape_hint.shape)} · ${t('to be tracked')}</li>`).join('');
    const r = C.objects.rendered;
    const e = C.scenegraph.edges.map((x) => `<li>${q(x.from)} ${x.rel === 'on' ? t('on') + ' ' + q(x.to) : t('in the air')}: ${num(x.t_start, 1)} – ${num(x.t_end, 1)} s</li>`).join('');
    const tp = C.task_plan, tc = Object.values(C.tracks)[0].coverage;
    const ht = C.harness_timing.steps;
    return block(2, t('Naming the objects'), `
      <p>${t('A model reads the question and the prior and lists the things that must be followed in the video. It merges different names for the same thing.')}</p>
      <ul class="plain">${o}</ul>
      <p>${t('It also looks at the frames and judges that the clip is a computer rendering:')} <span class="quote">${q(r.why)}</span> ${t('So it will not use everyday sizes of {x} to set the scale.', { x: C.objects.reference_rejected.map((x) => q(x.name)).join(zh() ? '、' : ', ') })}</p>
      <p>${t('A second call writes which frames the ball must be followed over: frames {a} to {b}.', { a: tc.required_window[0], b: tc.required_window[1] })}</p>
      <p>${t('A third call says what rests on what, read from six frames:')}</p>
      <ul class="plain">${e}</ul>
      <table class="tcost"><tr><th>${t('Part')}</th><th>${t('Time')}</th><th>${t('Model calls')}</th></tr>
        <tr><td>${t('Object list')}</td><td>${sec(ht.objects.seconds)}</td><td></td></tr>
        <tr><td>${t('Which frames to follow')}</td><td>${sec(tp.s)}</td><td>${tp.calls}</td></tr>
        <tr><td>${t('What rests on what')}</td><td>${sec(ht.scenegraph_grid.seconds)}</td><td></td></tr></table>`,
      cost(ht.objects.seconds + tp.s + ht.scenegraph_grid.seconds, null));
  }

  function b3() {
    const name = Object.keys(C.detect)[0], d = C.detect[name];
    const scan = Object.entries(d.scan.labels).map(([f, l]) => `<span class="chip ${l}">${f}: ${t(l)}</span>`).join('');
    const b = d.boxes[d.choice];
    const bind = d.binding;
    return block(3, t('Finding the ball'), `
      <p>${t('A model marks, on {n} frames spread over the clip, whether the ball is fully in view, partly, or absent:', { n: Object.keys(d.scan.labels).length })}</p>
      <p class="chips">${scan}</p>
      <p>${t('Frame {f} is used to pick the ball, because it is a frame where the ball is fully in view.', { f: d.frame })} ${t('An object detector, given the words “{p}”, proposes boxes there.', { p: esc(C.objects.list[0].grounding_phrase) })}</p>
      <div class="two">
        <figure><img src="${DIR}${d.image}" alt="">${legend([['bx', t('box the detector proposed and the model chose'), `border-color:${ballCol()}`]])}<figcaption>${t('Frame {f}, enlarged around the table', { f: d.frame })}</figcaption></figure>
        <div>
          <p>${t('The detector found {n} candidate box. The model chose it and gave this reason:', { n: d.n_candidates })}</p>
          <p class="quote">${q(d.reason)}</p>
          <p>${t('Box: {a} to {b} pixels (left, top to right, bottom).', { a: `(${num(b[0], 0)}, ${num(b[1], 0)})`, b: `(${num(b[2], 0)}, ${num(b[3], 0)})` })}</p>
          <p>${t('Rejected boxes: none.')}</p>
          <p>${t('The box must still sit on the tracked ball afterwards: they overlap by {iou} (at least {min} needed).', { iou: num(bind.box_iou_at_start, 2), min: num(bind.bound, 1) })}</p>
        </div>
      </div>`);
  }

  function b4() {
    const name = Object.keys(C.tracks)[0], tr = C.tracks[name], id = tr.identity;
    const strip = C.masks.strip.map((m) => `<figure class="mk"><img src="${DIR}${m.image}" alt="" loading="lazy"><figcaption>${t('frame')} ${m.frame} · ${num(m.t, 2)} s · ${m.area_px} px</figcaption></figure>`).join('');
    const ends = (e) => ({ clip_end: t('the clip begins here'), out_of_frame: t('the ball leaves the picture at the bottom edge') }[e.reason] || esc(e.why));
    const pr = id.presence;
    const checks = [
      [t('It moves when the plan says it moves'), t('moved {px} px between {a} s and {b} s', { px: num(id.moves.px, 1), a: num(id.moves.window[0], 1), b: num(id.moves.window[1], 1) })],
      [t('A model confirms the box shows the ball'), t('{n} of {m} say yes', { n: id.votes.filter((v) => v === 'yes').length, m: id.votes.length })],
      [t('The ball is in its box over the whole track'), t('{n} of {m} sampled frames say yes', { n: pr.yes, m: pr.n_votes })],
      [t('It is not only at the image edge'), t('{p} % of frames touch the edge', { p: num(id.border_frac * 100, 1) })],
      [t('The track covers the frames the plan needs'), t('needs {a}–{b}, has {c}–{d}', { a: tr.coverage.required_window[0], b: tr.coverage.required_window[1], c: tr.coverage.tracked_span[0], d: tr.coverage.tracked_span[1] })],
    ].map(([a, b]) => `<tr><td>${a}</td><td>${b}</td><td class="ok">${t('passed')}</td></tr>`).join('');
    const ht = C.harness_timing.steps.track.seconds;
    return block(4, t('Following the ball'), `
      <p>${t('A video segmenter (SAM 2) spreads the chosen box forwards and backwards through the clip and cuts out the ball on every frame: its mask.')}</p>
      <div class="strip">${strip}</div>
      ${legend([['ob', t('outline of the ball’s mask on that frame')]])}
      <p>${t('The ball is followed on {n} frames, from frame {a} to frame {b}. At the start: {s}. At the end: {e}.', { n: tr.n_frames, a: tr.span[0], b: tr.span[1], s: ends(tr.ends.start), e: ends(tr.ends.end) })}</p>
      <table class="tchk"><tr><th>${t('Check')}</th><th>${t('Result')}</th><th></th></tr>${checks}</table>
      <p>${t('The fit later reads these masks again from disk: {n} masks, cut in {s}.', { n: C.masks.log.n, s: sec(C.masks.log.s) })}</p>`,
      cost(ht + C.driver.masks.seconds, null) + `<span class="sub">${t('{a} finding and following, {b} masks for the fit', { a: sec(ht), b: sec(C.driver.masks.seconds) })}</span>`);
  }

  function b5() {
    const f = C.focal, n = C.mask_noise[ballId()];
    const vals = f.frames.map((fr) => f.per_frame[String(fr)]);
    const lo = Math.min(...vals) * 0.9, hi = Math.max(...vals) * 1.05;
    const W = 520, Hh = 150, pad = 30;
    const x = (i) => pad + i * (W - pad - 8) / (vals.length - 1);
    const y = (v) => Hh - 22 - (v - lo) / (hi - lo) * (Hh - 34);
    const dots = vals.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="var(--c-blue)"/><text x="${x(i)}" y="${Hh - 6}" text-anchor="middle" class="ax">${f.frames[i]}</text>`).join('');
    const med = `<line x1="${pad}" x2="${W - 8}" y1="${y(f.f)}" y2="${y(f.f)}" stroke="var(--text)" stroke-dasharray="5 4"/>`;
    const svg = `<svg viewBox="0 0 ${W} ${Hh}" class="chart" role="img">${med}${dots}<text x="${pad}" y="12" class="ax">${num(hi, 0)} px</text><text x="${pad}" y="${Hh - 22}" class="ax">${num(lo, 0)} px</text></svg>`;
    return block(5, t('What is measured before any fitting'), `
      <h3>${t('How much the mask jitters')}</h3>
      <p>${t('Two masks of consecutive frames are laid on top of each other, centre on centre, and the distance between their outlines is measured. The middle value over {n} pairs is {v} px; the masks overlap by {iou} on average. The method never trusts less than {floor} px, so {floor} px becomes the unit for “close enough” in every later check.', { n: n.pairs, v: num(n.noise_px, 2), iou: num(n.iou_consecutive, 2), floor: num(Math.SQRT2, 2) })}</p>
      <h3>${t('A first guess of the focal length')}</h3>
      <p>${t('A depth network (Depth Pro) guesses the focal length on {n} frames. The middle value, {f} px, is the starting point; the guesses spread by {s} px ({p} %).', { n: f.frames.length, f: num(f.f, 1), s: num(f.sigma, 1), p: num(f.mad_rel * 100, 1) })}</p>
      <figure>${svg}${legend([['dot', t('focal length guessed on one frame (frame number below)'), 'background:var(--c-blue)'], ['dash', t('middle value, the starting focal length')]])}</figure>`,
      cost(C.harness_timing.steps.focal.seconds, null));
  }

  function segRows(o) {
    return o.states.map((s, k) => `<tr><td>${o.states.length > 1 ? stretch(k) : ''}</td><td>${t('frames')} ${s.frames[0]}–${s.frames[1]}</td><td>${num(s.frames[0] / C.video.fps, 2)}–${num((s.frames[1] + 1) / C.video.fps, 2)} s</td><td>${lawText(s)}</td></tr>`).join('');
  }
  function b6() {
    const P = C.plan, pl = P.plan.plan || P.plan;
    const objs = pl.objects.map((o) => `<h4>${sw(o.id)}${q(o.name)} <span class="small">· ${o.track ? t('followed in the video') : t('not followed: no track')} · ${t('shape')}: ${t(o.shape)}</span></h4>
      <p class="quote">${q(o.description)}</p>
      <table class="tseg">${segRows(o)}</table>`).join('');
    const binds = pl.prior_bindings.map((b) => {
      const s = C.statements[b.statement];
      const what = b.binds === 'D' ? t('fixes the diameter of {o}', { o: obj(b.object) }) : t('becomes a rule the fit must meet exactly: the distance of {o} from the camera at that moment', { o: obj(b.object) });
      return `<li><span class="quote">${esc(s.raw)}</span> → ${what}</li>`;
    }).join('');
    const qb = P.query;
    const rev = P.review;
    const changed = (rev.added.length || Object.keys(rev.sharpened).length || rev.bindings_added.length);
    const reps = Object.keys(P.repairs || {}).length;
    const tm = P.timing;
    const calls = C.plan_calls.filter((c) => c.stage === 'plan' || c.stage === 'review').length;
    return block(6, t('The plan the model wrote'), `
      <p>${t('A model looks at nine frames in a grid, with the ball’s box drawn on each, and writes a plan of the scene: which objects there are, what shape they have, and how each one moves, as a list of stretches with their frames and the physical law of each.')}</p>
      <figure><img src="${DIR}plan_grid.jpg" alt="" loading="lazy">${legend([['bx', t('the tracked ball’s box and name, as the model saw them'), 'border-color:#C0564F']])}</figure>
      ${objs}
      <p>${t('It adds the table on its own: the question does not mention it, but the ball rests on it. Gravity is {g} m/s², because the scene is on Earth:', { g: C.structure.params.g.value })} <span class="quote">${q(pl.environment.reason)}</span></p>
      <h3>${t('What each given fact is tied to')}</h3>
      <ul class="plain">${binds}</ul>
      <p>${t('The question asks for the speed of {o} relative to the camera at {tt} s, in {u}.', { o: obj(qb.object), tt: qb.time, u: qb.unit })}</p>
      <h3>${t('Review and repairs')}</h3>
      <p>${changed ? '' : t('A second call reviews the plan against the frames. It changed nothing:')} <span class="quote">${q(rev.notes)}</span></p>
      <p>${reps ? '' : t('Code then checks the plan against fixed rules (for example: a fact can only be tied to a tracked object). Nothing broke a rule, so nothing was repaired.')}</p>`,
      cost(tm.plan_s, calls));
  }

  function b7() {
    const A = C.plan.authoring, S = C.shape.bodies;
    const figs = Object.keys(A).map((id) => {
      const img = C.copied.includes(`shape_${id}.png`) ? `<img src="${DIR}shape_${id}.png" alt="" loading="lazy">` : '';
      return `<figure class="shape">${img}<figcaption>${sw(id)}${q(C.names[id])}: ${t('written in {n} attempt, compiled, {f} failures', { n: A[id].attempts, f: A[id].failures.length })}</figcaption><pre>${esc(A[id].module)}</pre></figure>`;
    }).join('');
    const ball = S[ballId()];
    const rows = ball.before.per_frame.map((r) => `<tr><td>${r.frame}</td><td>${num(r.boundary_px, 2)} px</td><td>${num(r.extents[0].observed_px, 1)} / ${num(r.extents[0].rendered_px, 1)} px</td></tr>`).join('');
    const tm = C.plan.timing;
    const ac = C.plan_calls.filter((c) => c.stage === 'author').length;
    return block(7, t('Shapes'), `
      <p>${t('The model writes each object as a small 3D program (OpenSCAD) at unit size; the real size is a number the fit sets later.')}</p>
      <div class="shapes">${figs}</div>
      ${legend([['', t('the shapes as compiled, seen from a few directions')]])}
      <h3>${t('Checking the shape against the masks')}</h3>
      <p>${t('The table has no track, so there is nothing to compare it with. The ball is a sphere: its outline is measured, not sent to a model. On five clean frames the distance between the sphere’s outline and the mask’s outline is:')}</p>
      <table class="tchk"><tr><th>${t('Frame')}</th><th>${t('Outline distance')}</th><th>${t('Width seen / width of the sphere')}</th></tr>${rows}</table>
      <p>${t('The middle value is {v} px, below the {u} px unit, so no change was proposed: 0 proposals, 0 kept, 0 reverted. The larger gap on frame {f} is the blurred ball falling fast.', { v: num(ball.before.boundary_px, 2), u: num(ball.noise_unit_px, 2), f: ball.before.per_frame.reduce((m, r) => (r.boundary_px > m.boundary_px ? r : m)).frame })}</p>`,
      cost(tm.author_s + tm.shape_check_s, ac) + `<span class="sub">${t('{a} writing, {b} checking (0 calls)', { a: sec(tm.author_s), b: sec(tm.shape_check_s) })}</span>`);
  }

  function b8() {
    const P = C.structure.params;
    const rows = Object.entries(P).map(([k, p]) => {
      const fixed = p.fixed;
      return `<tr class="${fixed ? 'fx' : ''}"><td>${pname(k)}</td><td>${fixed ? t('fixed') : t('free')}</td><td>${fixed ? val(p.value, p.unit, k) : ''}</td><td>${fixed ? source(p.source) : ''}</td></tr>`;
    }).join('');
    const held = C.fit.constraints.filter((c) => c.quantity === 'distance_to_camera').map((c) => `<li><span class="quote">${esc(c.statement)}</span></li>`).join('');
    const nc = C.fit.constraints.filter((c) => /^continuity/.test(c.quantity)).length;
    return block(8, t('What the fit may change'), `
      <p>${t('Every number of the scene is listed. A number is fixed only when the prior or a fixed convention gives it; all others are left free for the fit.')}</p>
      <table class="tpar"><tr><th>${t('Number')}</th><th></th><th>${t('Value')}</th><th>${t('Given by')}</th></tr>${rows}</table>
      ${legend([['fxs', t('fixed before the fit')]])}
      <p>${t('Rules that must hold exactly while the fit moves the free numbers:')}</p>
      <ul class="plain">${held}<li>${t('{n} joins: the ball’s position (and, from rolling into flight, its velocity) must carry on without a jump where one stretch meets the next', { n: nc })}</li></ul>
      <p>${t('The plan said the ball is on the table, but the table is not tracked, so “on the table” cannot be checked; the ball’s stretches are placed freely in space instead.')}</p>`,
      cost(null, 0) + `<span class="sub">${t('part of the 48 s spent on plan, shapes and this list')}</span>`);
  }

  function b9() {
    const S = C.fit.camera_starts, cc = C.fit.camera_candidates;
    const best = Math.min(...S.map((s) => s.final_objective));
    const rows = S.map((s, i) => `<tr class="${i === cc.chosen ? 'kept' : ''}"><td>${num(s.pitch_deg, 0)}°</td><td>${num(s.yaw_deg, 0)}°</td><td>${sci(s.objective_before)} → ${sci(s.objective_after)}</td><td>${num(s.refined_objective, 1)}</td><td>${num(s.final_objective, 1)}</td><td>${s.valid ? t('valid') : t('invalid')}</td></tr>`).join('');
    const fl = C.fit_log;
    const ts = fl.screen.reduce((a, b) => a + b.s, 0), tr = fl.refine.reduce((a, b) => a + b.s, 0);
    return block(9, t('Searching for the camera'), `
      <p>${t('Nobody tells the method how the camera is tilted. It tries {n} tilts, from looking steeply up ({a}°) to steeply down ({b}°). For each tilt it places the ball (heading included) to match its masks, and scores how badly the drawn ball misses the masks (lower is better). The ball must sit {d} m from the camera at {tt} s, as stated.', { n: S.length, a: num(S[0].pitch_deg, 0), b: num(S[S.length - 1].pitch_deg, 0), d: C.statements.find((x) => x.quantity === 'distance_to_camera').value_si, tt: C.statements.find((x) => x.quantity === 'distance_to_camera').time })}</p>
      <table class="tcam"><tr><th>${t('Starting tilt')}</th><th>${t('Heading found')}</th><th>${t('Quick placement: miss before → after')}</th><th>${t('After full refinement')}</th><th>${t('After the whole fit')}</th><th></th></tr>${rows}</table>
      ${legend([['kp', t('the start kept: lowest miss after the whole fit')]])}
      <p>${t('No start was invalid: none put the camera on the wrong side of a surface, and the stated distances could be met from every one. After full refinement all {n} land within {th} of the best ({b}), so all {n} were carried through the whole fit. The one kept ends with a miss of {k}, the lowest.', { n: cc.within_noise, th: num(cc.threshold_total, 0), b: num(cc.best_objective, 1), k: num(best, 1) })}</p>`,
      cost(ts + tr, 0) + `<span class="sub">${t('{a} quick placements, {b} refinements', { a: sec(ts), b: sec(tr) })}</span>`);
  }

  function b10() {
    const id = ballId(), O = C.fit_objects[id], P = C.fit.params;
    const keys = Object.keys(P).filter((k) => (k.startsWith(id + '.') || k.startsWith('cam.')) && !/yaw$/.test(k));
    const unitOf = (k) => (C.structure.params[k] || {}).unit;
    const rows = keys.map((k) => {
      const p = P[k];
      const st = k === 'cam.f' ? C.fit.f_start.value : p.init;
      return `<tr><td>${pname(k)}</td><td>${val(st, unitOf(k), k)}</td><td>${val(p.value, unitOf(k), k)} <span class="small">${sigma(p.sigma, unitOf(k))}</span></td></tr>`;
    }).join('');
    const ref = O.refine;
    const rr = ref.rounds.map((r, i) => `<tr><td>${i + 1}</td><td>${num(r.objective_before, 1)} → ${num(r.objective_after, 1)}</td><td>${r.nfev}</td><td>${sec(r.time_s)}</td></tr>`).join('');
    const sh = O.boundary_shifts.map((b, i) => `<li>${t('{s} → {s2}: frame {a} → frame {b} ({d} frames later); miss afterwards {o}', { s: stretch(i), s2: stretch(i + 1), a: b.frame_from, b: b.frame_to, d: `+${b.frame_to - b.frame_from}`, o: num(b.objective, 1) })}</li>`).join('');
    const pairs = C.pairs.map((p) => `<div class="pair"><figure><img src="${DIR}${p.start.image}" alt="" loading="lazy"><figcaption>${t('start')} · ${t('overlap')} ${num(p.start.iou[id], 2)}</figcaption></figure><figure><img src="${DIR}${p.fitted.image}" alt="" loading="lazy"><figcaption>${t('fitted')} · ${t('overlap')} ${num(p.fitted.iou[id], 2)}</figcaption></figure><div class="pf">${t('frame')} ${p.frame} · ${num(p.t, 2)} s</div></div>`).join('');
    const tbl = C.fit.untracked_bodies.map((b) => obj(b)).join(', ');
    return block(10, t('Fitting the ball'), `
      <p>${t('From the kept start, every free number of the ball and the camera is adjusted together until the drawn ball sits on its masks on {n} frames, while the stated distances and the joins hold exactly.', { n: O.frames_fitted })}</p>
      <table class="tpar"><tr><th>${t('Number')}</th><th>${t('Start')}</th><th>${t('Fitted')}</th></tr>${rows}</table>
      <p class="small">${t('Positions are measured from the camera: forward along the ground, to the left, and up.')}</p>
      <h3>${t('The miss, round by round')}</h3>
      <table class="tchk"><tr><th>${t('Round')}</th><th>${t('Miss before → after')}</th><th>${t('Tries')}</th><th>${t('Time')}</th></tr>${rr}</table>
      <h3>${t('Where the stretches meet')}</h3>
      <p>${t('The plan’s frame boundaries may move by up to three frames if that lowers the miss. Both moved:')}</p>
      <ul class="plain">${sh}</ul>
      <p>${t('So the ball starts rolling at {a} s instead of {b} s and leaves the table at {c} s instead of {d} s.', { a: num(O.boundary_shifts[0].frame_to / C.video.fps, 2), b: num(O.boundary_shifts[0].frame_from / C.video.fps, 2), c: num(O.boundary_shifts[1].frame_to / C.video.fps, 2), d: num(O.boundary_shifts[1].frame_from / C.video.fps, 2) })}</p>
      <h3>${t('Before and after, on the frames')}</h3>
      <div class="pairs">${pairs}</div>
      ${legend([['ob', t('the ball’s mask (observed)')], ['pv', t('the drawn ball, visible part'), `border-color:${ballCol()}`], ['po', t('the drawn ball, hidden part'), `border-color:${ballCol()}`]])}
      <p class="small">${t('The start pictures are drawn from the start values the fit recorded, with the starting focal length, no roll and the plan’s boundaries; the pipeline’s own renderer draws both. Overlap is the shared area of the drawn ball and the mask divided by their combined area: 1 is a perfect match, 0 no contact.')}</p>
      <p>${t('{x} is not fitted: it has no track.', { x: tbl })}</p>`,
      cost(ref.time_s, 0) + `<span class="sub">${t('the refinement of the kept start; all starts together are counted in step 9')}</span>`);
  }

  function b11() {
    const J = C.fit.joint, F = C.fit.focal, fs = C.fit.f_start;
    const rr = J.rounds.map((r, i) => `<tr><td>${r.phase === 'alternation' ? i + 1 : t('final step')}</td><td>${num(r.objective_before, 1)} → ${num(r.objective_after, 1)}</td><td>${r.nfev}</td><td>${sec(r.time_s)}</td></tr>`).join('');
    const st = C.fit_objects[ballId()].size_test;
    const tj = C.fit_log.joint.reduce((a, b) => a + b.s, 0);
    const fp = C.fit.params['cam.f'];
    return block(11, t('The joint pass'), `
      <p>${t('Last, all {v} free numbers move together once more, now including the focal length, which was held at its first guess until here.', { v: J.vars })}</p>
      <table class="tchk"><tr><th>${t('Round')}</th><th>${t('Miss before → after')}</th><th>${t('Tries')}</th><th>${t('Time')}</th></tr>${rr}</table>
      <p>${t('The miss drops from {a} to {b}.', { a: num(J.objective_before, 1), b: num(J.objective_after, 1) })}</p>
      <p>${t('Focal length: freed, and kept, because the fit pins it and it is not at a limit: {a} px → {b} px (± {s} px).', { a: num(fs.value, 1), b: num(F.value, 1), s: num(fp.sigma, 1) })}</p>
      <p>${t('Sizes: nothing was freed. The ball’s only size is its diameter, which the prior fixes at {d} mm.', { d: num(C.structure.params[ballId() + '.D'].value * 1000, 0) })}</p>`,
      cost(J.time_s, 0) + `<span class="sub">${t('for the kept start; {a} for all {n} starts', { a: sec(tj), n: C.fit.camera_candidates.completed.length })}</span>`);
  }

  function b12() {
    const P = C.fit.params;
    const rows = Object.entries(P).map(([k, p]) => `<tr class="${p.undetermined ? 'und' : 'det'}"><td>${pname(k)}</td><td>${p.undetermined ? t('not pinned') : t('pinned')}</td><td>${p.undetermined ? why(p.why) : ''}</td></tr>`).join('');
    const sc = C.fit.identifiability.scale;
    const fb = sc.fixed_by.map((f) => {
      if (f.kind === 'statement') return `<li><span class="quote">${esc(f.statement)}</span>: ${t('changing the scale would break this stated distance')}</li>`;
      const m = f.how.match(/silhouettes change by ([\d.]+)/);
      const what = f.name === 'g' ? t('gravity, {g} m/s², through the fall at the end', { g: f.value }) : t('the diameter, {d} mm', { d: num(f.value * 1000, 0) });
      return `<li>${what}: ${t('changing the scale would make the drawn ball miss its masks by {m} more', { m: m ? m[1] : '' })}</li>`;
    }).join('');
    return block(12, t('What the fit can vouch for'), `
      <p>${t('For every number the method asks: if it changed a little, would the masks or the stated facts notice? If not, or if its spread is large, it is marked not pinned, with the reason.')}</p>
      <table class="tpar"><tr><th>${t('Number')}</th><th></th><th>${t('Why not')}</th></tr>${rows}</table>
      ${legend([['dets', t('pinned by the video and the prior')], ['unds', t('not pinned')]])}
      <h3>${t('What holds the real-world scale')}</h3>
      <p>${t('Scaling the whole scene up or down would look the same in the video. Four things stop it:')}</p>
      <ul class="plain">${fb}</ul>`, cost(null, 0) + `<span class="sub">${t('part of the fit')}</span>`);
  }

  function b13() {
    const A = C.answer.answer_scene, Q = C.fit.query.values[0];
    const ss = Q.statement_sigmas.map((s) => `<li><span class="quote">${esc(s.statement)}</span>: ± ${num(s.sigma * 100, 2)} cm/s</li>`).join('');
    return block(13, t('The read-out'), `
      <p class="big">${num(A.value, 1)} <span>± ${num(A.sigma, 1)} cm/s</span></p>
      <p>${t('The fitted scene is asked directly: how fast does the ball move at {tt} s? At that moment it is in {s}, rolling on the table, so the speed comes from that stretch’s law.', { tt: Q.time, s: stretch(Q.segment) })}</p>
      <p>${A.determined ? t('It counts as determined: everything it reads is pinned, the stated facts hold, and its spread is small next to its size.') : esc(A.reason)}</p>
      <p>${t('The spread ± {s} cm/s is mostly how well the drawn ball matches the masks in that stretch (± {r} cm/s); the stated distances, each written to its last digit, add:', { s: num(A.sigma, 1), r: num(Q.sigma_residual_in_unit, 1) })}</p>
      <ul class="plain">${ss}</ul>`, cost(null, 0));
  }

  function b14() {
    const F = C.answer.answer_final, call = F.calls[0];
    const nFrames = C.answer.answer_final.images.filter((p) => /answer_frames/.test(p)).length;
    const nOv = C.answer.answer_final.images.filter((p) => /overlay_/.test(p)).length;
    const gt = C.ground_truth != null ? Number(C.ground_truth) : null;
    return block(14, t('The answer'), `
      <p>${t('One last model call sees the question, the prior, {n} frames of the video, {o} pictures of the fitted scene drawn over the frames, and the scene written out in words with its read-out. It must judge for itself whether the scene can be trusted.', { n: nFrames, o: nOv })}</p>
      <p>${t('Its reason:')}</p>
      <p class="quote">${q(F.rationale)}</p>
      <table class="tans"><tr><th></th><th>cm/s</th></tr>
        <tr><td>${t('Read-out of the fitted scene')}</td><td>${num(C.answer.answer_scene.value, 1)} ± ${num(C.answer.answer_scene.sigma, 1)}</td></tr>
        <tr class="kept"><td>${t('Answer')}</td><td>${num(F.value, 1)}</td></tr>
        <tr><td>${t('Released ground truth')}</td><td>${gt != null ? num(gt, 2) : ''}</td></tr></table>
      ${gt != null ? `<p>${t('The answer is off by {d} cm/s ({p} %).', { d: num(F.value - gt, 1), p: num((F.value - gt) / gt * 100, 1) })}</p>` : ''}`,
      cost(C.driver.answer.seconds, F.model_calls));
  }

  function timeline() {
    const h = C.harness_timing.steps, tp = C.task_plan, pt = C.plan.timing, fl = C.fit_log;
    const ts = fl.screen.reduce((a, b) => a + b.s, 0), tr = fl.refine.reduce((a, b) => a + b.s, 0), tj = fl.joint.reduce((a, b) => a + b.s, 0);
    const fitRest = C.driver.fit.seconds - ts - tr - tj;
    const buildRest = C.driver.build.seconds - pt.plan_s - pt.author_s - pt.shape_check_s;
    const harnessRest = C.harness_timing.total_seconds - h.objects.seconds - tp.s - h.scenegraph_grid.seconds - h.track.seconds - h.focal.seconds;
    const rows = [
      [2, t('Object list'), h.objects.seconds, null],
      [2, t('Which frames to follow'), tp.s, tp.calls],
      [2, t('What rests on what'), h.scenegraph_grid.seconds, null],
      [3, t('Finding and following the ball'), h.track.seconds, null],
      [5, t('Focal length guess'), h.focal.seconds, null],
      ['', t('Between these steps'), harnessRest, null],
      [4, t('Masks for the fit'), C.driver.masks.seconds, null],
      [6, t('Plan and review'), pt.plan_s, C.plan_calls.filter((c) => c.stage === 'plan' || c.stage === 'review').length],
      [7, t('Shapes and shape check'), pt.author_s + pt.shape_check_s, C.plan_calls.filter((c) => c.stage === 'author').length],
      [8, t('Rest of the build (loading, the list of numbers)'), buildRest, 0],
      [9, t('Camera search: quick placements'), ts, 0],
      [9, t('Camera search: refinements of all starts'), tr, 0],
      [11, t('Joint passes of all starts'), tj, 0],
      [12, t('Rest of the fit (boundary search, uncertainty, pictures)'), fitRest, 0],
      [14, t('Answer'), C.driver.answer.seconds, C.answer.answer_final.model_calls],
    ];
    const total = rows.reduce((a, r) => a + r[2], 0);
    const max = Math.max(...rows.map((r) => r[2]));
    const calls = rows.reduce((a, r) => a + (r[3] || 0), 0);
    const tr_ = rows.map(([n, l, s, c]) => `<tr><td>${n ? `<a href="#step${n}">${n}</a>` : ''}</td><td>${l}</td><td class="bar"><span style="width:${(s / max * 100).toFixed(1)}%"></span></td><td>${sec(s)}</td><td>${c == null ? '' : c}</td></tr>`).join('');
    return `<section class="cstep" id="time"><h2>${t('Where the time went')}</h2>
      <table class="ttime"><tr><th>${t('Step')}</th><th></th><th></th><th>${t('Time')}</th><th>${t('Model calls')}</th></tr>${tr_}
      <tr class="tot"><td></td><td>${t('Total')}</td><td></td><td>${sec(total)} (${num(total / 60, 1)} min)</td><td>${calls}</td></tr></table>
      <p>${t('The fit takes {p} % of the time, because all {n} camera starts are carried through to the end.', { p: num((C.driver.fit.seconds) / total * 100, 0), n: C.fit.camera_candidates.completed.length })} ${t('Model calls are counted where the run records them: the object list, the scene graph and the tracking checks also ask a model, but their logs do not count the calls.')}</p></section>`;
  }

  function render() {
    if (!C) return;
    const nav = document.getElementById('nav-case');
    if (nav) nav.textContent = t('One case');
    document.title = t('One Case');
    root.innerHTML = `<h1>${t('One question, from the video to the answer')}</h1>
      <p class="lede">${t('Every step of the method on one question, in the order it ran: what went in, what the step changed, and how long it took. All numbers and pictures are this question’s own records.')}</p>
      <ol class="toc">${[t('The input'), t('Naming the objects'), t('Finding the ball'), t('Following the ball'), t('What is measured before any fitting'), t('The plan the model wrote'), t('Shapes'), t('What the fit may change'), t('Searching for the camera'), t('Fitting the ball'), t('The joint pass'), t('What the fit can vouch for'), t('The read-out'), t('The answer')].map((x, i) => `<li><a href="#step${i + 1}">${x}</a></li>`).join('')}<li><a href="#time">${t('Where the time went')}</a></li></ol>`
      + b1() + b2() + b3() + b4() + b5() + b6() + b7() + b8() + b9() + b10() + b11() + b12() + b13() + b14() + timeline();
  }

  fetch(DIR + 'case.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then((d) => { C = d; render(); })
    .catch((e) => { root.textContent = String(e); });
  if (window.I18N) window.I18N.on(render);
})();
