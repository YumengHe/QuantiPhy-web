"""Build QuantiPhy-web/data/videos.json (+ first-frame thumbnails) for videos.html.

Reads only existing outputs; never runs the pipeline.

  /home/yumenghe/miniconda3/envs/quantiphy/bin/python QuantiPhy-web/tools/build_videos.py

Inputs (relative to the QuantiPhy repo root):
  outputs/val_mh3_valcur11_s2dv8_v9b_routed.csv  validation predictions shown (0.7479: valcur11 Qwen + scene2d v8 + scene fit c_v9b;
                                                 s2d / s2d_conf = 2D-scene value and its confidence, fit / fit_conf = 3D
                                                 scene-fit value and its confidence; source geo / vlm / s2d / fit)
  outputs/submission_v7.csv                      test predictions shown on the test tab (row-aligned with load_split('test'))
  outputs/scene_fit_eval/c_v9b/<video>.json      validation scene fits (scene graph, supports, dimension sources,
                                                 visible intervals, per-question quality / conf, ballistic paths)
  outputs/scene2d_eval/s2d_v8/<video>.json       validation 2D layered scenes (+ <video>_bg.jpg, copied to data/scene2d/);
                                                 object inventory, scene graph, rejected layers, constraints, mask hulls
  outputs/tracks[_test]/<video>.json             per-object visible intervals for the 3D viewer
  outputs/eval/overview_*.md, <video>.md         evaluator verdicts (summary table row; per-video report copied to data/eval/)
  outputs/scene_fit_eval/c_support_v7_test/, outputs/scene2d_eval/s2d_v7_test/  test scenes behind submission_v7
                                                 (written per video to data/scenes/test/<video>.json)
  data/QuantiPhy*/...mp4                         only to grab the first frame and duration

The page plays the original videos from media/val and media/test (git-ignored symlinks, see video_info).
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.data import cache_root, load_split, video_path  # noqa: E402
from pipeline.metrics import CATEGORIES, evaluate, mra_row  # noqa: E402
from pipeline.parse import parse_question  # noqa: E402

VAL_CSV = ROOT / "outputs" / "val_mh3_valcur11_s2dv8_v9b_routed.csv"
TEST_CSV = ROOT / "outputs" / "submission_v7.csv"
SCENE_DIRS = {"val": ROOT / "outputs" / "scene_fit_eval" / "c_v9b",   # scene_fit v9b, the run behind the 0.7479 routing
              "test": ROOT / "outputs" / "scene_fit_eval" / "c_support_v7_test"}   # the run behind submission_v7
SCENE2D_DIRS = {"val": ROOT / "outputs" / "scene2d_eval" / "s2d_v8",         # scene2d v8
                "test": ROOT / "outputs" / "scene2d_eval" / "s2d_v7_test"}     # the run behind submission_v7
SCENES_TEST = WEB / "data" / "scenes" / "test"   # one json per test video, fetched by the page when a card opens
BG_MAX_W = 960                                   # background images wider than this are shrunk (test is up to 2560 px)
EVAL_DIR = ROOT / "outputs" / "eval"
COUNT_CHECK_VAL = ROOT / "outputs" / "eval" / "count_check_val.json"   # per-frame visible-count check (qp-tools)
COUNT_CHECK: dict = {}    # video -> count_check, for scenes whose json has no "count_check" field
EVAL_WEB = WEB / "data" / "eval"
SCENE2D_WEB = WEB / "data" / "scene2d"
OUT = WEB / "data" / "videos.json"
THUMBS = WEB / "data" / "thumbs"
THUMB_W = 240
THUMB_MAX_BYTES = 30_000
HULL_KEYS = 12        # mask hulls kept per layer (evenly over its key frames), to bound videos.json
HULL_EPS = 1.0        # polygon simplification tolerance (px, cv2.approxPolyDP)


def num(x, sig=4):
    """Round to `sig` significant digits; NaN/inf/None -> None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def text(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def vec(v):
    return [num(a, 5) for a in v] if v is not None else None


def inventory(inv):
    """Object inventory (pipeline.objects) as shown on the page: canonical name, aliases, where it is mentioned."""
    if not inv:
        return None
    out = []
    for o in inv.get("objects") or []:
        where = []
        for m in o.get("mentions") or []:
            w = text(m.get("where")) if isinstance(m, dict) else ""
            if w and w not in where:
                where.append(w)
        out.append({"name": text(o.get("name")), "aliases": [text(a) for a in o.get("aliases") or []],
                    "track": o.get("track"), "dyn": o.get("dyn"), "role": o.get("role"), "where": where})
    return out


def graph(sg):
    """Scene graph (pipeline.scenegraph) without the raw LLM text: support nodes, edges, per-object 2D info, same-instance checks."""
    if not sg:
        return None
    edges = [{"from": text(e.get("from")), "to": e.get("to"), "rel": text(e.get("rel")), "conf": num(e.get("conf"), 3),
              "t0": num(e.get("t_start"), 4), "t1": num(e.get("t_end"), 4)} for e in sg.get("edges") or []]
    support = [text(n.get("id")) for n in sg.get("nodes") or [] if n.get("type") == "support"]
    same = [{"a": text(x.get("a")), "b": text(x.get("b")), "same": x.get("same"), "conf": num(x.get("conf"), 3)}
            for x in sg.get("same_instance") or []]
    return {"source": text(sg.get("source")), "support": support, "edges": edges, "same": same,
            "objects_2d": sg.get("objects_2d") or None}


def hulls(L, fps):
    """At most HULL_KEYS mask hulls per layer as [[t, [[x, y], ...]], ...], simplified to HULL_EPS px."""
    H = L.get("hulls") or {}
    keys = sorted(H, key=lambda f: int(f))
    if len(keys) > HULL_KEYS:
        keys = [keys[round(i * (len(keys) - 1) / (HULL_KEYS - 1))] for i in range(HULL_KEYS)]
    t_of = dict(zip(L.get("frames") or [], L.get("t") or []))
    out = []
    for f in keys:
        poly = np.asarray(H[f].get("hull") or [], dtype=np.float32)
        if len(poly) < 3:
            continue
        poly = cv2.approxPolyDP(poly.reshape(-1, 1, 2), HULL_EPS, True).reshape(-1, 2)
        t = t_of.get(int(f), int(f) / fps if fps else None)
        out.append([num(t, 6), [[round(float(x), 1), round(float(y), 1)] for x, y in poly]])
    return out


def eval_overview(video_ids) -> tuple[dict, str | None]:
    """Evaluator verdicts from the newest outputs/eval/overview_<n>.md: its markdown summary table
    (| video | MRA (n, 0分) | 主因层 | 根因 | 进行中 |) -> {video_id: {...}}. A row may name several videos
    ("a / b / c") and may drop a "_segmented" suffix, so names are matched to video ids by prefix."""
    files = sorted(EVAL_DIR.glob("overview_*.md"), key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1)
    if not files:
        return {}, None
    src = files[-1]
    out = {}
    header = None
    for line in src.read_text().splitlines():
        if not line.startswith("|"):
            header = None if not line.strip() else header
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if len(header) < 5 or "主因层" not in header:
            continue
        row = dict(zip(header, cells))
        for name in row[header[0]].split("/"):
            name = name.strip()
            vid = next((v for v in video_ids if v == name), None) or next((v for v in video_ids if v.startswith(name + "_")), None)
            if vid:
                out[vid] = {"mra": row.get(header[1], ""), "layer": row.get("主因层", ""), "cause": row.get("根因", ""),
                            "progress": row.get("进行中", ""), "source": str(src.relative_to(ROOT))}
    return out, str(src.relative_to(ROOT))


IMG_RE = re.compile(r"(/[^\s`'\"()<>]+\.(?:jpg|jpeg|png|gif|webp))", re.I)
IMG_MAX_BYTES = 5_000_000


def _inline(s: str) -> str:
    """Markdown inline -> HTML: escape, `code`, **bold**, [text](url)."""
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    return s


def md_to_html(md: str, title: str) -> str:
    """Small markdown subset (headings, pipe tables, - / 1. lists with indentation, paragraphs, inline code / bold,
    image paths already rewritten to relative ones) -> a standalone page for data/eval/."""
    out, para, i, lines = [], [], 0, md.splitlines()

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()
    while i < len(lines):
        l = lines[i]
        st = l.strip()
        if not st:
            flush(); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)", l)
        if m:
            flush(); n = len(m.group(1)); out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>"); i += 1; continue
        if st.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            if rows:
                t = "<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr>"
                t += "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f'<div class="scroll"><table>{t}</table></div>')
            continue
        m = re.match(r"^(\s*)(?:[-*]|\d+\.)\s+(.*)", l)
        if m:
            flush(); items = []
            while i < len(lines):
                m = re.match(r"^(\s*)(?:[-*]|\d+\.)\s+(.*)", lines[i])
                if m:
                    items.append([len(m.group(1)), m.group(2)])
                elif lines[i].strip() and items and lines[i].startswith(" "):
                    items[-1][1] += " " + lines[i].strip()          # continuation line of the previous item
                else:
                    break
                i += 1
            out.append("<ul>" + "".join(f'<li style="margin-left:{ind * 8}px">{_inline(t)}</li>' for ind, t in items) + "</ul>")
            continue
        para.append(st); i += 1
    flush()
    body = "\n".join(out)
    # images referenced by a relative path become inline figures under their mention
    body = re.sub(r'<code>([\w.-]+/[\w.-]+\.(?:jpg|jpeg|png|gif|webp))</code>',
                  r'<a href="\1"><code>\1</code></a><br><img src="\1" alt="\1" loading="lazy">', body)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1f2328; --muted:#656d76; --border:#d8dee4; --code:#f0f2f4; --th:#eef2fb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181c; --fg:#e6e8eb; --muted:#9aa4af; --border:#33383f; --code:#23272e; --th:#232a38; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.7 system-ui, -apple-system, "PingFang SC", "Noto Sans CJK SC", sans-serif; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 24px 16px 60px; }}
h1 {{ font-size: 24px; }} h2 {{ font-size: 19px; margin-top: 28px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
code {{ background: var(--code); padding: 0 4px; border-radius: 4px; font-size: 13px; overflow-wrap: anywhere; }}
.scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; font-size: 13.5px; margin: 8px 0; }}
th, td {{ border: 1px solid var(--border); padding: 4px 8px; vertical-align: top; text-align: left; }} th {{ background: var(--th); }}
img {{ max-width: 100%; border-radius: 6px; margin: 6px 0; }} a {{ color: #3b6fd8; }}
.back {{ font-size: 13px; color: var(--muted); }}
</style></head><body><main>
<p class="back"><a href="../../videos.html">← 视频与问题</a> · 评估员逐视频报告（由 outputs/eval/*.md 生成）</p>
{body}
</main></body></html>
"""


def eval_report(vid: str):
    """Per-video evaluator report outputs/eval/<video>.md -> data/eval/<video>.html (images it names by absolute path
    are copied to data/eval/<video>/ and linked relatively). Returns the page path, the conclusion excerpt and, when the
    report states them, the main layer and root cause (these override the overview row)."""
    p = EVAL_DIR / f"{vid}.md"
    if not p.exists():
        return None
    EVAL_WEB.mkdir(parents=True, exist_ok=True)
    txt = p.read_text()
    for src in sorted(set(IMG_RE.findall(txt))):
        sp = Path(src)
        if sp.is_file() and sp.stat().st_size <= IMG_MAX_BYTES:
            (EVAL_WEB / vid).mkdir(exist_ok=True)
            (EVAL_WEB / vid / sp.name).write_bytes(sp.read_bytes())
            rel = f"{vid}/{sp.name}"
            txt = txt.replace(src, rel)
            # a bare (not `quoted`) mention becomes `rel` too, so the page embeds the picture under it
            txt = re.sub(rf"(?<!`){re.escape(rel)}(?!`)", f"`{rel}`", txt)
    out = EVAL_WEB / f"{vid}.html"
    first = next((l.lstrip("# ").strip() for l in txt.splitlines() if l.startswith("# ")), vid)
    out.write_text(md_to_html(txt, first))
    stale = EVAL_WEB / p.name
    if stale.exists():
        stale.unlink()
    # main layer / root cause as the report states them: an explicit "主因层：" / "根因：" line, else the first bold item
    # of the section whose heading says 根因 (e.g. "**prior 物体跟错，导致尺度错（跟踪层，主因，影响 4 题）**")
    layer = cause = None
    lines = txt.splitlines()
    for l in lines:
        m = re.match(r"^\s*[-*]?\s*\**主因层\**[:：]\s*(.+)", l)
        layer = layer or (m and m.group(1).strip("* "))
        m = re.match(r"^\s*[-*]?\s*\**根因\**[:：]\s*(.+)", l)
        cause = cause or (m and m.group(1).strip("* "))
    if cause is None:
        for i, l in enumerate(lines):
            if l.startswith("#") and "根因" in l:
                for m_ in lines[i + 1:]:
                    if m_.startswith("#"):
                        break
                    b = re.search(r"\*\*([^*]+)\*\*", m_)
                    if b:
                        cause = b.group(1).rstrip("：:")
                        break
                break
    if layer is None and cause:
        m = re.search(r"([^\s，,（(]+?)层[，,]\s*主因", cause)
        layer = m and m.group(1)
    # excerpt: the section whose heading says 结论 (conclusion), else the start of the report; markdown emphasis dropped
    body = None
    for i, l in enumerate(lines):
        if l.startswith("#") and "结论" in l:
            body = []
            for m in lines[i + 1:]:
                if m.startswith("#"):
                    break
                body.append(m)
            title = l.lstrip("# ").strip()
            break
    if body is None:
        # no conclusion section: the report's opening text, without the 主因层 / 根因 lines already shown on their own
        body, title = [l for l in lines if not l.startswith("#") and not re.match(r"^\s*[-*]?\s*\**(主因层|根因)", l)], ""
    # prose only: markdown table rows and the "数据来源 / 结果来源" provenance line are left to the full report
    body = [l for l in body if l.strip() and not l.lstrip().startswith("|") and not re.match(r"^\s*(数据来源|结果来源)", l)]
    text_ = " ".join(l.strip() for l in body).replace("**", "").replace("`", "")
    return {"file": str(out.relative_to(WEB)), "title": title, "excerpt": text_[:400], "layer": layer or None,
            "cause": cause or None}


def _norm(name) -> str:
    """Name key for matching inventory / scene-graph / scene objects: lower case, no leading article, no punctuation."""
    x = re.sub(r"[^a-z0-9#]+", " ", str(name or "").lower()).strip()
    return re.sub(r"^(the|a|an) ", "", x)


def object_table(split: str, vid: str, is3d: bool) -> dict | None:
    """One row per object: the inventory (pipeline.objects.inventory, presence merged), the per-frame scene graph
    (scenegraph_pfc), and the 3D fit (scene_fit_eval) or the 2D layers (scene2d_eval), matched by normalised name and
    aliases. Inventory objects without a scene object get a "not reconstructed" reason; scene objects the inventory
    does not have are flagged (a self-check finding) and listed in "unmatched"."""
    try:
        from pipeline.objects import inventory
        inv = inventory(split, vid)["objects"]
    except Exception:   # noqa: BLE001 - no inventory: no table, the page falls back to the scene tables
        return None
    key = {}
    for o in inv:
        for a in [o["name"], *(o.get("aliases") or [])]:
            key.setdefault(_norm(a), o["name"])
    sgp = cache_root("scenegraph_pfc", split) / f"{vid}.json"
    edges = []
    if sgp.exists():
        try:
            edges = json.load(open(sgp)).get("edges") or []
        except (OSError, json.JSONDecodeError):
            pass
    rels = {}
    for e in edges:
        n = key.get(_norm(e.get("from")), e.get("from"))
        t = f" {e['t_start']:.2f}–{e['t_end']:.2f} s" if e.get("t_start") is not None and e.get("t_end") is not None else ""
        rels.setdefault(n, []).append(f"{e.get('rel')}{(' ' + str(e['to'])) if e.get('to') else ''}{t}")
    scene, s2 = None, None
    d = (SCENE_DIRS if is3d else SCENE2D_DIRS).get(split)
    p = d / f"{vid}.json" if d else None
    if p is not None and p.exists():
        try:
            j = json.load(open(p))
            scene, s2 = (j, None) if is3d else (None, j)
        except (OSError, json.JSONDecodeError):
            pass
    sc_objs = {}
    if scene:
        collapsed = {r.get("object") for r in scene.get("results") or [] if (r.get("quality") or {}).get("collapsed")}
        for n, o in (scene.get("objects") or {}).items():
            dims = o.get("dims") or {}
            ds = o.get("dim_source") or {}
            sp = o.get("support") or {}
            flags = []
            if o.get("out_of_frame"):
                flags.append("出画 " + ", ".join(f"{a:.2f}–{b:.2f} s" for a, b in o["out_of_frame"]))
            va = (o.get("visibility") or {}).get("agree")
            if va is not None and va < 0.9:
                flags.append(f"可见性一致度 {va:.2f}")
            if n in collapsed:
                flags.append("尺寸塌缩")
            sc_objs[n] = {"size": [[k, num(v), ds.get(k)] for k, v in dims.items()], "shape": o.get("shape"),
                          "support": " ".join(x for x in [sp.get("mode") or o.get("mode") or "", sp.get("target") or ""] if x),
                          "visible": [[num(a, 4), num(b, 4)] for a, b in o.get("visible") or []],
                          "motion": " ".join(x for x in [o.get("mode") or "", f"rms {o['rms']:.1f} px" if o.get("rms") is not None else ""] if x),
                          "flags": flags}
    rejected = {}
    if s2:
        sc = s2.get("scale") or {}
        mpp = None if sc.get("degenerate") else sc.get("m_per_px_ref")
        rejected = {k: v for k, v in (s2.get("rejected_layers") or {}).items()}
        for L in s2.get("layers") or []:
            e = L.get("extent_px") or {}
            c = L.get("constraint") or {}
            segs = ", ".join(sorted({g.get("model") for g in L.get("segments") or [] if g.get("model")}))
            sc_objs[L["name"]] = {"size_px": [num(e.get("along")), num(e.get("across")), num(e.get("h"))],
                                  "mpp": num(mpp, 5), "src": e.get("src"),
                                  "support": " ".join(x for x in [c.get("mode") or "", c.get("surface") or ""] if x),
                                  "visible": [[num(a, 4), num(b, 4)] for a, b in L.get("visible") or []],
                                  "motion": " ".join(x for x in [segs, f"rms {L['rms_px']:.1f} px" if L.get("rms_px") is not None else ""] if x),
                                  "flags": [text(f) for f in L.get("flags") or []]}
    rows, used = [], set()
    for o in inv:
        names = {_norm(a) for a in [o["name"], *(o.get("aliases") or [])]}
        hit = next((n for n in sc_objs if _norm(n) in names), None)
        hits = [n for n in sc_objs if _norm(n).split(" #")[0] in names]   # second instances "<name> #2"
        row = {"name": o["name"], "aliases": [a for a in o.get("aliases") or [] if a != o["name"]],
               "role": o.get("role"), "reference_only": bool(o.get("reference_only")), "count": o.get("count"),
               "present": o.get("present"), "presence_conf": num(o.get("presence_conf"), 3),
               "shape": (o.get("shape_hint") or {}).get("shape"), "thin": bool((o.get("shape_hint") or {}).get("thin")),
               "track": o.get("track"), "part_of": o.get("part_of"), "relations": rels.get(o["name"], []),
               "scene": [{"name": n, **sc_objs[n]} for n in (hits or ([hit] if hit else []))]}
        used.update(hits or ([hit] if hit else []))
        if not row["scene"]:
            why = ("仅作参考（reference_only）" if o.get("reference_only") else "不在场（absent）" if o.get("present") == "absent"
                   else "场景结构，不跟踪" if o.get("track") is False else ("被拒绝：" + text(rejected[o["name"]])) if o["name"] in rejected
                   else "场景里没有这个物体")
            row["not_built"] = why
        rows.append(row)
    unmatched = [n for n in sc_objs if n not in used]
    for n in unmatched:
        rows.append({"name": n, "aliases": [], "unmatched": True, "relations": rels.get(n, []), "scene": [{"name": n, **sc_objs[n]}]})
    return {"rows": rows, "unmatched": unmatched, "dim": 3 if is3d else 2}


def track_intervals(split: str, vid: str) -> dict:
    """{track name: [[t0, t1], ...]} — the frames where the track cache (outputs/tracks[_test]/<video>.json) has a box,
    merged into runs of consecutive frames, in seconds (frame / fps of the cache)."""
    p = cache_root("tracks", split) / f"{vid}.json"
    if not p.exists():
        return {}
    try:
        j = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}
    fps = (j.get("info") or {}).get("fps") or 30.0
    out = {}
    for name, tr in (j.get("tracks") or {}).items():
        fr = sorted(int(k) for k, b in (tr.get("boxes") or {}).items() if b is not None)
        runs = []
        for f in fr:
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        out[name] = [[num(a / fps, 5), num(b / fps, 5)] for a, b in runs]
    return out


PATH_DT = 1 / 24     # sampling step (s) of non-parabolic object paths for the 3D viewer


def sampled_path(o: dict, j: dict):
    """[[t, x, y, z], ...] of an object's centre from pipeline.scene_fit._json_pos (ballistic replay or motion pieces),
    over the times its fit covers (to the last results / visibility time, at least 2 s)."""
    from pipeline.scene_fit import _json_pos
    ends = [iv[1] for k in ("visible", "out_of_frame") for iv in (o.get(k) or [])]
    T = max(ends + [2.0])
    n = int(round(T / PATH_DT))
    out = []
    for i in range(n + 1):
        t = i * PATH_DT
        try:
            P = _json_pos(o, t)
        except Exception:   # noqa: BLE001 - a broken entry just falls back to the parabola on the page
            return None
        out.append([num(t, 4)] + [num(float(x), 5) for x in P])
    return out


def align(a):
    """t_q alignment check of one answer (scene json "align_tq"): a number (IoU) or a dict -> {"iou", "dc"} | None."""
    if a is None:
        return None
    if isinstance(a, (int, float)):
        return {"iou": num(a, 3), "dc": None, "px": None, "t": None, "frame": None}
    if isinstance(a, dict):
        # 3D: iou, dcenter_rel (centre offset / box size); 2D: iou, dev_rel, dev_px
        iou = next((a[k] for k in ("iou", "IoU", "box_iou") if a.get(k) is not None), None)
        dc = next((a[k] for k in ("dcenter_rel", "dev_rel") if a.get(k) is not None), None)
        px = next((a[k] for k in ("dev_px", "dcenter_px", "center_px") if a.get(k) is not None), None)
        if iou is None and dc is None and px is None:
            return None
        return {"iou": num(iou, 6), "dc": num(dc, 6), "px": num(px, 3), "t": num(a.get("t"), 4), "frame": a.get("frame")}
    return None


def question_times(q: str, meta: dict | None, scene: dict | None, is3d: bool):
    """t_q of a question for the page: the times it names (pipeline.parse), and the time to jump to — the named time
    (midpoint of a [t1, t2] window), else the middle of the asked object's visible interval, else None."""
    try:
        times = [float(t) for t in parse_question(q).times]
    except Exception:   # noqa: BLE001 - a question the parser cannot read simply gets no time
        times = []
    jump = (sum(times[:2]) / len(times[:2])) if times else None
    if jump is None and meta and (meta.get("align") or {}).get("t") is not None:   # the frame the alignment check used
        jump = meta["align"]["t"]
    if jump is None and meta and scene:
        obj = meta.get("obj")
        items = scene.get("objects") if is3d else scene.get("layers")
        it = next((o for o in items or [] if o.get("name") == obj), None)
        vis = (it or {}).get("visible") or []
        if vis:
            jump = (vis[0][0] + vis[-1][1]) / 2
    return [num(t, 4) for t in times], num(jump, 4)


def scene_fit(split: str, vid: str):
    p = SCENE_DIRS[split] / f"{vid}.json"
    if not p.exists():
        return None
    try:
        j = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    pl = j.get("plane") or {}
    plane = {k: vec(pl.get(k)) for k in ("normal", "u1", "u2")} | {"dist": num(pl.get("dist"), 5)} if pl else None
    objs = []
    for name, o in (j.get("objects") or {}).items():
        objs.append({"name": name, "shape": o.get("shape"), "dims": {k: num(v) for k, v in (o.get("dims") or {}).items()},
                     "rms": num(o.get("rms"), 3), "p0": vec(o.get("p0")), "v0": vec(o.get("v0")), "a": vec(o.get("a")),
                     "yaw": num(o.get("yaw"), 5), "t_on": num(o.get("t_on"), 5),
                     # where each dimension comes from: observed / prior / bound (not seen, anti-collapse floor)
                     "dim_source": o.get("dim_source"), "support": o.get("support"), "hang": o.get("hang"),
                     "length": num(o.get("length")), "mode": o.get("mode"),
                     "out_of_frame": o.get("out_of_frame"), "visibility": o.get("visibility"),
                     "overhead": bool(o.get("overhead")), "events": ((o.get("ballistic") or {}).get("events") or [])[:12],
                     # bounce model / motion pieces are not a single parabola: sample the centre for the viewer
                     "path": sampled_path(o, j) if (o.get("ballistic") or o.get("segments")) else None})
    res = []
    for r in j.get("results") or []:
        qual = r.get("quality") or {}
        res.append({"q": r.get("question"), "obj": r.get("object"), "pred": num(r.get("pred")),
                    "method": r.get("method"), "answer": num(r.get("answer")) if split == "val" else None,
                    "mra": num(r.get("mra"), 3) if split == "val" else None,
                    "conf": num(qual.get("conf"), 3), "quality": {k: (num(v, 3) if isinstance(v, float) else v)
                                                                  for k, v in qual.items() if k != "conf"} or None})
    # per question: the fitted object and the t_q alignment check (quality.align_tq), when the scene json has them
    q_meta = {r.get("question"): {"obj": r.get("object"), "align": align((r.get("quality") or {}).get("align_tq"))}
              for r in j.get("results") or [] if r.get("question")}
    sg = j.get("scene_graph")
    inv = inventory(j.get("inventory") or j.get("objects_inventory"))
    if inv is None and sg:   # no inventory in the json: the scene graph's object nodes carry the same names and mentions
        inv = inventory({"objects": [{"name": n.get("id"), "aliases": list(dict.fromkeys(
            m.get("alias") for m in n.get("mentions") or [] if isinstance(m, dict) and m.get("alias"))),
            "mentions": n.get("mentions") or []} for n in sg.get("nodes") or [] if n.get("type") == "object"]})
    # visible intervals: the scene json's own "visible" when it has one, else the object's track (its name, else an alias)
    tracks = track_intervals(split, vid)
    aliases = {o["name"]: o["aliases"] for o in inv or []}
    for o, (name, raw) in zip(objs, (j.get("objects") or {}).items()):
        vis = raw.get("visible")
        if vis is None:
            key = next((k for k in [name, *aliases.get(name, [])] if k in tracks), None)
            vis = tracks.get(key) if key else None
        o["visible"] = [[num(a, 5), num(b, 5)] for a, b in vis] if vis else None
    return {"file": str(p.relative_to(ROOT)), "f": num(j.get("f"), 5), "plane": plane, "objects": objs, "results": res,
            "main_support": j.get("main_support"), "inventory": inv, "scene_graph": graph(sg), "q_meta": q_meta,
            "count_check": j.get("count_check") or (COUNT_CHECK.get(vid) if split == "val" else None),
            "f_source": text(j.get("f_source")) or None, "f_first": j.get("f_first"),
            "focal_votes": [{"source": text(x.get("source")), "value": num(x.get("value"), 4), "weight": num(x.get("weight"), 3)}
                            for x in j.get("focal_votes") or []] or None}


def scene2d_checks(j: dict) -> dict:
    """s2d_v5 consistency checks, compacted: relation checks per track candidate, size-order checks, frame-level
    reference measurements and the critic's applied / rejected issues (each only when present)."""
    out = {}
    rc = []
    for r in j.get("relation_checks") or []:
        rc.append({"from": text(r.get("from")), "candidates": [
            {"cache": text(c.get("cache")), "ok": c.get("ok"),
             "checks": [{"rel": text(x.get("rel")), "to": text(x.get("to")), "ok": x.get("ok"), "why": text(x.get("why"))}
                        for x in c.get("checks") or []]} for c in r.get("candidates") or []]})
    if rc:
        out["relation_checks"] = rc
    sc = [{"smaller": text(x.get("smaller")), "larger": text(x.get("larger")), "ok": x.get("ok"),
           "px": [num(a, 4) for a in x.get("px") or []], "why": text(x.get("why"))} for x in j.get("size_checks") or []]
    if sc:
        out["size_checks"] = sc
    rm = {text(k): {"px": num(v.get("px"), 4), "conf": num(v.get("conf"), 3), "relation": text(v.get("relation")),
                    "frame": v.get("frame"), "p0": vec(v.get("p0")), "p1": vec(v.get("p1")), "spread": num(v.get("spread"), 3)}
          for k, v in (j.get("reference_measurements") or {}).items() if isinstance(v, dict)}
    if rm:
        out["refmeasure"] = rm
    cq = j.get("critique") or {}
    if cq.get("applied") or cq.get("rejected"):
        out["critique"] = {"applied": [json.dumps(a, ensure_ascii=False)[:200] for a in cq.get("applied") or []],
                           "rejected": [json.dumps(a, ensure_ascii=False)[:200] for a in cq.get("rejected") or []],
                           "rerun": text(cq.get("rerun"))}
    return out


def scene2d(split: str, vid: str):
    """Compact copy of the 2D layered scene (pipeline/scene2d.py docstring); pixel coordinates of that json's width/height."""
    d = SCENE2D_DIRS.get(split)
    p = d / f"{vid}.json" if d else None
    if p is None or not p.exists():
        return None
    try:
        j = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    bg = None
    bgname = (j.get("background") or {}).get("image")
    if bgname and (d / bgname).exists():
        # val keeps data/scene2d/<name>; test (same video ids, higher resolution) goes to data/scene2d/test/, shrunk
        out_dir = SCENE2D_WEB if split == "val" else SCENE2D_WEB / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / bgname
        img = cv2.imread(str(d / bgname))
        if img is not None and img.shape[1] > BG_MAX_W:
            img = cv2.resize(img, (BG_MAX_W, round(img.shape[0] * BG_MAX_W / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        else:
            out.write_bytes((d / bgname).read_bytes())
        bg = str(out.relative_to(WEB))
    sc = j.get("scale") or {}
    persp = sc.get("perspective") or {}
    scale = {"model": sc.get("model"), "m_per_px_ref": num(sc.get("m_per_px_ref"), 5), "y_ref": num(sc.get("y_ref"), 5),
             "k_per_px": num(sc.get("k_per_px"), 5), "degenerate": bool(sc.get("degenerate")),
             "persp_accepted": bool(persp.get("accepted")), "persp_reason": text(persp.get("reason")),
             "notes": [text(x) for x in sc.get("notes") or []]}
    pr = j.get("prior") or {}
    prior = {k: text(pr.get(k)) for k in ("text", "quantity", "object", "layer", "how")} | {"value_si": num(pr.get("value_si")),
                                                                                         "px": num(pr.get("px"))}

    def par(v):
        if isinstance(v, (list, tuple)):
            return [num(a, 6) for a in v]
        return v if isinstance(v, bool) else num(v, 6)

    # size quantities the questions ask of each layer, so the page labels a layer with those sizes only
    size_q = {}
    for r in j.get("results") or []:
        q = r.get("quantity")
        if q in ("length", "width", "height", "diameter") and r.get("layer"):
            size_q.setdefault(r["layer"], [])
            if q not in size_q[r["layer"]]:
                size_q[r["layer"]].append(q)
    layers = []
    for L in j.get("layers") or []:
        segs = [{"t0": num(s.get("t0"), 6), "t1": num(s.get("t1"), 6), "model": s.get("model"),
                 "rms": num(s.get("rms_px"), 3), "p": {k: par(v) for k, v in (s.get("params") or {}).items()}}
                for s in L.get("segments") or []]
        layers.append({"name": text(L.get("name")), "moving": bool(L.get("moving")),
                       "t": [num(t, 6) for t in L.get("t") or []],
                       "box": [[round(float(a), 1) for a in b] for b in L.get("box_px") or []],
                       "c": [[round(float(a), 1) for a in c] for c in L.get("center_px") or []],
                       "visible": [[num(a, 5) for a in iv] for iv in L.get("visible") or []],
                       "extent": {k: (text(v) if k == "src" else num(v, 4)) for k, v in (L.get("extent_px") or {}).items()},
                       "rms": num(L.get("rms_px"), 3), "flags": [text(f) for f in L.get("flags") or []], "segments": segs,
                       "size_q": size_q.get(L.get("name"), []), "orient": vec(L.get("orientation")), "hulls": hulls(L, j.get("fps")),
                       "constraint": {k: (L.get("constraint") or {}).get(k) for k in ("mode", "surface", "motion_plane", "applied", "note")}})
    return {"file": str(p.relative_to(ROOT)), "fps": num(j.get("fps")), "width": j.get("width"), "height": j.get("height"),
            "n_frames": j.get("n_frames"), "bg": bg, "prior": prior, "scale": scale,
            "gravity_dir": vec(j.get("gravity_dir")), "gravity_source": text(j.get("gravity_source")),
            "inventory": inventory(j.get("objects")), "scene_graph": graph(j.get("scene_graph")),
            "rejected": {text(k): text(v) for k, v in (j.get("rejected_layers") or {}).items()}, "layers": layers,
            "count_check": j.get("count_check") or (COUNT_CHECK.get(vid) if split == "val" else None),
            "q_meta": {r.get("question"): {"obj": r.get("layer"), "align": align(r.get("align_tq"))}
                       for r in j.get("results") or [] if r.get("question")},
            **scene2d_checks(j)}


LA = ZoneInfo("America/Los_Angeles")


def version_of(split: str, vid: str, is3d: bool, csv: Path) -> dict:
    """Version shown next to the timestamp: the scene json's own "version" field when it has one
    ({pipeline, commit, scene2d, scene_fit, objects, presence, refmeasure, time} -> "pipeline · commit"); else pieced
    together from the scene directory name, the pipeline tag in the result csv name (val_mh3_<tag>_...) and the json mtime."""
    d = (SCENE_DIRS if is3d else SCENE2D_DIRS).get(split)
    p = d / f"{vid}.json" if d else None
    ver = None
    if p is not None and p.exists():
        try:
            ver = json.load(open(p)).get("version")
        except (OSError, json.JSONDecodeError):
            ver = None
    if isinstance(ver, dict) and (ver.get("pipeline") or ver.get("commit")):
        return {"label": " · ".join(str(ver[k]) for k in ("pipeline", "commit") if ver.get(k)), "detail": ver}
    parts = []
    if p is not None and p.exists():
        parts.append(("拟合 " if is3d else "场景 ") + d.name)
    m = re.match(r"(?:val|test)_mh3_([A-Za-z0-9]+)", csv.name)
    if m:
        parts.append("流水线 " + m.group(1))
    elif split == "test":
        parts.append(csv.stem)
    detail = {}
    if p is not None and p.exists():
        detail["scene_mtime"] = datetime.fromtimestamp(p.stat().st_mtime, LA).strftime("%Y-%m-%d %H:%M %Z")
    return {"label": " · ".join(parts), "detail": detail}


def run_at(split: str, vid: str, csv: Path) -> dict:
    """Last time this video was run, in Los Angeles time: the mtime of its scene json (3D scene fit / 2D layered
    scene); only a video without a scene json falls back to the per-question csv."""
    scenes = [p for p in (d / f"{vid}.json" for d in (SCENE_DIRS.get(split), SCENE2D_DIRS.get(split)) if d) if p.exists()]
    ts = max(p.stat().st_mtime for p in scenes or [csv])
    t = datetime.fromtimestamp(ts, LA)
    return {"run_at": t.strftime("%Y-%m-%d %H:%M %Z"), "run_at_iso": t.isoformat(timespec="seconds")}


def video_meta(split: str, vid: str, make_thumb: bool):
    """(duration_s, width, height, thumb_rel_path) from the mp4; thumbnail from the first frame."""
    p = video_path(split, vid)
    if not p.exists():
        return None, None, None, None
    cap = cv2.VideoCapture(str(p))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    dur = n / fps if fps > 0 and n > 0 else None
    thumb = None
    if ok:
        out = THUMBS / f"{vid}.jpg"
        if make_thumb or not out.exists():
            th = max(1, round(frame.shape[0] * THUMB_W / frame.shape[1]))
            small = cv2.resize(frame, (THUMB_W, th), interpolation=cv2.INTER_AREA)
            for q in (72, 60, 48, 36, 25):
                ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok2 and len(buf) <= THUMB_MAX_BYTES:
                    break
            out.write_bytes(buf.tobytes())
        thumb = str(out.relative_to(WEB))
    return num(dur, 4), w, h, thumb


def fourcc(p: Path) -> str:
    cap = cv2.VideoCapture(str(p))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((code >> (8 * k)) & 0xFF) for k in range(4)).strip("\x00").lower()


def video_info(split: str, vid: str) -> dict:
    """Page-relative URL of the original mp4 (media/<split> are git-ignored symlinks to the video dirs of video_path()).

    Browsers only decode H.264 here. All validation mp4s are MPEG-4 Part 2 ('mp4v'); the test set has the same
    video_id at higher resolution, usually in H.264, so the page plays that copy instead and says so.
    "video" is None when no browser-playable copy exists; "orig" is always the original file.
    """
    orig = f"media/{split}/{video_path(split, vid).name}"
    codec = fourcc(video_path(split, vid))
    info = {"orig": orig, "codec": codec, "video": orig if codec in ("avc1", "h264") else None, "video_from": split}
    if info["video"] is None and split == "val" and video_path("test", vid).exists():
        if fourcc(video_path("test", vid)) in ("avc1", "h264"):
            info.update(video=f"media/test/{video_path('test', vid).name}", video_from="test")
    return info


def build_val(make_thumb: bool):
    df = pd.read_csv(VAL_CSV)
    df["mra"] = [mra_row(p, g) for p, g in zip(df["pred"], df["answer"])]
    ev = evaluate(df)
    print(f"val: n={ev['n']} macro={ev['average']:.4f} " + " ".join(f"{c}={ev[c]:.4f}" for c in CATEGORIES))
    videos = []
    verdicts, eval_src = eval_overview(sorted(df["video_id"].unique()))
    for vid, g in df.groupby("video_id", sort=True):
        dur, w, h, thumb = video_meta("val", vid, make_thumb)
        rep = eval_report(vid)
        ev_v = (verdicts.get(vid) or {}) | ({"report": rep} if rep else {})
        if rep:   # the per-video report wins over the overview row; the overview stays as the fallback
            ev_v |= {k: rep[k] for k in ("layer", "cause") if rep.get(k)} | {"from_report": True}
        first = g.iloc[0]
        is3d = str(first.dim) == "3"
        scene = scene_fit("val", vid) if is3d else None
        sc2 = None if is3d else scene2d("val", vid)
        qmeta = ((scene or sc2) or {}).get("q_meta") or {}
        qs = [{"cat": r.category, "type": r.inference_type, "q": r.question, "prior": text(r.prior),
               "depth": text(r.depth_info), "answer": num(r.answer), "geo": num(r.geo), "vlm": num(r.vlm),
               "pred": num(r.pred), "src": r.source, "mra": num(r.mra, 3), "s2d": num(getattr(r, "s2d", None)),
               "s2d_conf": num(getattr(r, "s2d_conf", None), 3), "fit": num(getattr(r, "fit", None)),
               "fit_conf": num(getattr(r, "fit_conf", None), 3),
               "unobs": bool(getattr(r, "unobservable", False) is True),
               "align": (qmeta.get(r.question) or {}).get("align")}
              for r in g.itertuples()]
        for q in qs:
            q["tq"], q["tjump"] = question_times(q["q"], qmeta.get(q["q"]), scene or sc2, is3d)
        videos.append({"id": vid, "source": first.video_source, "video_type": first.video_type, "fps": num(first.fps),
                       "dim": int(first.dim), "cats": sorted(set(g.category)), "duration": dur, "size": [w, h],
                       "thumb": thumb, **video_info("val", vid),
                       "scene": scene, "scene2d": sc2, "objtab": object_table("val", vid, is3d),
                       "version": version_of("val", vid, is3d, VAL_CSV),
                       **run_at("val", vid, VAL_CSV), "eval": ev_v or None,
                       "questions": qs})
    print(f"eval: {len(verdicts)} videos with a verdict from {eval_src}")
    return {"file": str(VAL_CSV.relative_to(ROOT)), "macro": num(ev["average"], 6),
            "per_cat": {c: num(ev[c], 6) for c in CATEGORIES}, "eval_source": eval_src, "videos": videos}


def build_test(make_thumb: bool):
    sub = pd.read_csv(TEST_CSV)
    ref = load_split("test").reset_index(drop=True)
    if len(sub) != len(ref):
        raise SystemExit(f"row count mismatch: {len(sub)} vs {len(ref)}")
    bad = (sub["video_id"].values != ref["video_id"].values) | (sub["question"].values != ref["question"].values)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} rows of {TEST_CSV.name} do not align with load_split('test')")
    ref["pred"] = sub["parsed_value"].values
    videos = []
    n_fit = n_2d = 0
    for vid, g in ref.groupby("video_id", sort=True):
        first = g.iloc[0]
        dur, w, h, thumb = video_meta("test", vid, make_thumb)
        is3d = str(first.dim) == "3"
        sc = scene_fit("test", vid) if is3d else None
        s2 = None if is3d else scene2d("test", vid)
        n_fit += sc is not None
        n_2d += s2 is not None
        ref_file = None
        if sc or s2:   # kept out of videos.json (568 videos): written per video and fetched when the card opens
            SCENES_TEST.mkdir(parents=True, exist_ok=True)
            out = SCENES_TEST / f"{vid}.json"
            out.write_text(json.dumps({"scene": sc, "scene2d": s2}, ensure_ascii=False, separators=(",", ":")))
            ref_file = str(out.relative_to(WEB))
        qs = [{"row": int(i) + 1, "cat": r.category, "type": r.inference_type, "q": r.question, "prior": text(r.prior),
               "depth": text(r.depth_info), "pred": num(r.pred)} for i, r in zip(g.index, g.itertuples())]
        for q in qs:
            q["tq"], q["tjump"] = question_times(q["q"], ((sc or s2) or {}).get("q_meta", {}).get(q["q"]), sc or s2, is3d)
        videos.append({"id": vid, "source": first.video_source, "video_type": first.video_type, "fps": num(first.fps),
                       "dim": int(first.dim), "cats": sorted(set(g.category)), "duration": dur, "size": [w, h],
                       "thumb": thumb, **video_info("test", vid), "scene": None, "scene2d": None, "scene_ref": ref_file,
                       "has_scene": sc is not None, "has_scene2d": s2 is not None, **run_at("test", vid, TEST_CSV),
                       "version": version_of("test", vid, is3d, TEST_CSV),
                       "questions": qs})
    n3d = sum(v["dim"] == 3 for v in videos)
    print(f"test: {len(ref)} questions, {len(videos)} videos, scene fit {n_fit}/{n3d} 3D videos, "
          f"2D layered scene {n_2d}/{len(videos) - n3d} 2D videos")
    return {"file": str(TEST_CSV.relative_to(ROOT)), "videos": videos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    ap.add_argument("--count-check", type=Path, default=COUNT_CHECK_VAL,
                    help="validation count_check results {video: {frames: [{t, n_video, n_scene, ...}], ...}} merged into "
                         "scenes that do not carry the field themselves (default: %(default)s when it exists)")
    args = ap.parse_args()
    if args.count_check and args.count_check.exists():
        COUNT_CHECK.update(json.load(open(args.count_check)))
        print(f"count_check: {len(COUNT_CHECK)} videos from {args.count_check}")
    THUMBS.mkdir(parents=True, exist_ok=True)
    data = {"generated": date.today().isoformat(), "val": build_val(args.rethumb), "test": build_test(args.rethumb)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
