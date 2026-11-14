"""Build web/data/videos.json (+ first-frame thumbnails) for evaluation.html from validation runs.

Reads only the runs' outputs; never runs the pipeline. The page lists one row per question; a question is the
unit <video_id>_q<k> (k = its 1-based order among the video's rows of the split table, docs/PER_QUESTION.md).

  python -B web/tools/build_videos.py --val-runs v6gpt --sceneprog v6val \
      --answers /nas/yumenghe/QuantiPhy-data/scratch/qp-code-3d-chain/answers_fix.json

Runs are listed in order of preference: each question is shown from the first run whose CSV has its row, and every
value of that question (pred, MRA, source, scene, overlays, tracks, scene graph) comes from that run alone. The page
labels each question with its version. Every version is independent (docs/WORKING_RULES.md): a run is read only from
the artifacts it wrote, in the format VERSION_FORMAT declares for it. A run missing an artifact of a question it has
is an error, never a substitute.

Inputs of a run <tag> (relative to the repo root):
  outputs/val_<tag>_answers.csv (or the CSV given as <tag>=<path>)
                                                       harness CSV (one row per question run, "row" = table row)
  outputs/harness/<tag>/answers/<key>.json             answer records ("version"."pipeline" = <tag>)
  outputs/scene_fit_eval/h_<tag>/agent/<key>.json      saved scenes (unit "video": one per prior group)
  outputs/scenegraph_<tag>/<key>.json                  scene graph cache
  outputs/tracks_<tag>/<key>.json                      track cache (visible intervals per object)
  outputs/objects_<tag>/<key>.json                     object inventory (unit "question": identity checked)
  web/data/runtime/<tag>/overlays.json                 overlays rendered for that version (images beside it)
A run tag is a version, optionally with the model backend ("v6gpt": v6 run with --model-backend gpt) and "_control";
its format is the version's. <key> is the video for unit "video" and the question's unit <video_id>_q<k> for unit "question"; a unit-question file
whose "unit" (or video) is not that question's raises. Answer records of unit "question" have schema "unit-1", with
the question's timing. Also docs/DEV20.md, the development question set -> val "dev"; each of its units must be in some run.

Overlays: the record web/data/runtime/<tag>/overlays.json lists every overlay of the version. Only the current version
(pipeline/VERSION) may render the ones it lacks, with the current code; building an older version needs every overlay
in its record, with its image on disk.

Scene program (versions whose VERSION_FORMAT has "sceneprog", --sceneprog BUILD --answers JSON): every question of the
first run also carries "sp" (sceneprog_view: both answers from the answer step's table, their MRA against the ground
truth with the baselines' rule, the renders under media/sceneprog/<BUILD>/), and data/comparison_ours.json is written
(write_comparison_ours).

Layout: {"generated", "val": {"runs": [{"version", "unit", "csv"}], "dev", "videos", "questions", "sceneprog"}}. Each question
names its "version". A video carries "runs": {version: {"run_at", ...}}; with unit "video" (the run fits one scene
per video and prior group) that entry also holds the groups' scenes, the scene graph and tracks, and each question
names its group. With unit "question" each question carries its own scene / scene graph / tracks / fit.

The page plays the original videos from media/val (git-ignored symlinks, see video_info).

Also regenerates the class-size table of pipeline.html (scene program section) from pipeline/class_sizes.py (write_class_table).

Note: the "v0" keys below (and o.v0 / p.v0 in videos.js) are the initial-velocity field of the scene json written by
the pipeline, not a version label; do not rename or remove them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.data import load_split, video_path  # noqa: E402
from pipeline.metrics import mra_row  # noqa: E402
from pipeline.parse import parse_question  # noqa: E402

OUT = WEB / "data" / "videos.json"
THUMBS = WEB / "data" / "thumbs"
THUMB_W = 240
THUMB_MAX_BYTES = 30_000
CURRENT = (ROOT / "pipeline" / "VERSION").read_text().strip()

# The artifact format of each version. A version not listed here cannot be built: add its entry first.
#   motion   key of the physics-law motion record of a scene object ("ballistic" in v1, "physics" from v2 on)
#   paths    object keys whose motion is sampled for the 3D viewer (pipeline.scene_state3d.position); v1 "ballistic"
#            motion is not executable by scene_state3d, so v1 balls are drawn from p0 / v0 / a only
#   unit     processing unit of the run: "video" (one scene per video and prior group) or "question"
#   decide   how a scene value becomes the answer: "conf" (a numeric confidence: CSV / scene result "conf"),
#            "checks" (named pass/fail checks: answer record "checks" / "failed" / "status", CSV "failed"; no conf) or
#            "agree" (v5, docs/ANSWER_AGREEMENT.md section 5: the named checks, plus every measurement of the asked
#            quantity in the scene result -- "measurements", "chosen", "selection" -- the checks "agree" {chi2, dof,
#            bound, values} and "constraints" {failed_ids}, and the answer's "constraint_residuals" with sigma_stated /
#            sigma_pred / bound / pass / testable; each scene object has a "motion_law")
#   steps    the answer record's timing steps, in pipeline order (unit "question" only)
#   identity the track cache has "meta"."identity" {name: {"candidates", "chosen", "untracked"}} (the track identity
#            judge: static / verify / class_size / presence / border); each scene object has "dim_bounds"
#            {dimension: [lo, hi | null]} for exactly its "bound" dimensions; a prior-scaled scene's "scale" names its
#            source, and a prior-extent scale its "extent" (the frames and depth observations it was measured on)
#   gravity  the scene has "gravity" {g, sigma, source, rule, up_source, evidence, checks, failed} (rule B2) and a
#            prior-scaled scene's "scale" its named "checks" / "failed"; a measurement read from a ballistic piece has
#            "inputs"."law" {g, source, up}; each ballistic piece of the gravity evidence has "admissible" {pass, reason}
#            and an inferred gravity its "earth_test" {pass, reason}
#   release  every scene graph edge with "t_release" has its "release_rule" (release.last_contact / free_flight), an
#            airborne one its "release_why"; the graph has "untracked" {name: why}
#   refmeasure the reference measurement record outputs/refmeasure_v<refmeasure.VERSION>_<tag>/<unit>.json (when the
#            unit has one) has "checks" / "failed" per "<object>|<quantity>" entry
#   kept     the answer record has "kept_state" {iter, scenegraph, ...}: the scene graph the answer used (the agent
#            round kept, 0 = the initial fit) and "iterations" (the agent rounds with their tool results); a scene graph
#            edge rewritten by scene_fit carries "was" {rel, to} and "rule"; every constraint residual has "stated"
#            {text, digits, unit, sigma, source}; an identity class_size entry has "dims" {candidate, ref}
#   selftest every scene graph edge has the builder's "test" and a merged one "absorbed" [{rel, to, t_start, t_end, test}];
#            the graph has "unknown" [{from, rel, to, t_start, t_end, test, reason}] and "self_test" {tested, fail,
#            merged, unknown} (rule builder.self_test); each scene object has "segfit" [piece] | null with "point" /
#            "sigma" / "bound" / "rms" / "run" / "pass" (rule segfit.point), a physics record "events_graph" and a
#            motion law "events" / "unknown" (rule events.graph)
#   horizon  the scene has "focal" {votes [{object, span, hop, f, sigma, law, n, rms, used, why}], f_votes, sigma,
#            n_votes, f, source} (rule focal.gravity_hop) and "support_horizon" {pass, reason, phi_plane_deg,
#            phi_horizon_deg, sigma_deg, z, row_plane, row_horizon, horizon {...}, repaired} (rule support.horizon);
#            camera.support_plane has "pitch_source" / "depth_plane"; the answer record has "stop" {reason, round, why,
#            tried} and each agent round "newly_failed" {check: reason}
#   camera   the scene camera has "f_sigma" / "f_sigma_how" (rule C3a, the scene's "f_source" names its source) and
#            "focal"."consistent" [{source, f_obj, sigma_obj, f, sigma, bound, pass, reason, how, class_size?}]; the scene
#            has "support_plane" {pass, failed, reason, tests {name: {pass, reason, ..}}} (pass null, no tests: no
#            support plane; rule D9); a direct
#            measurement with a state span has "window" {t0, t1, state, clipped_by {start, end}, window} (rule
#            window.state), a direct size "support_depth" {applies, ..} and, when it applies, "cross_checks"
#            {"direct.depth_agree"} (rule C12); each agent round past the accept test has "checks_fixed" /
#            "checks_broken" (rule accept.full_checks) and a rebuilding round "rebuilt" {steps, failed, self_test}
#   roll     the scene has "camera_roll" {pass, reason, roll_source, roll_fit_deg?, roll_fit_how?, roll_image_deg?,
#            sigma_deg?, z?, image {status, why, n_lines, n_frames, ..}?, refit {was, now}?} (rule camera.roll); a
#            direct window has "law" / "check" (rule window.driven: law null, check "window.smooth", inputs "driven" /
#            "noise_px" / "noise_how"); an agent round has "repair_candidates" [{for, call, class}] and
#            "repair_candidates_dropped" [{call, why}] (rule repair.window_checks); the unit's extents record
#            outputs/extents2d_<tag>/<unit>.json, when it exists, has "format" 3 with per-frame "extent" /
#            "speed_px" / "overlay" (rule extent.overlay)
#   static   the scene has "extents" {object: {dim: {sigma_mask, sigma_mask_how, at: {label: {value, sigma, frames_used,
#            frames_rejected [{frame, speed_px, extent, why}], how}}}}} with "objects"[name]."extent" the same record or
#            null (rule extent.static), "camera_sources" [{object, frames, t, rule}], every focal.consistent row
#            "independent" (rule focal.independent), "camera_one" {f, sigma, how, votes [{source, object, f, sigma,
#            used, why}]} or null (rule camera.one) and "depth_bindings" {given, bound [{name, node, as, used {by,
#            why}}], unbound [{name, reason}]} (rule C14); a direct speed measurement may have "inputs"."centred"
#            {from, extremum, stop} (rule window.centred); the harness CSV has "unit_set"
#   fusion   the scene's "focal" has "fusion" {pass, reason, f, sigma, n_sources, camera_f, camera_sigma, sources
#            [{object, kind, f, sigma, n_votes, camera, how, verdict (used / outlier / disagree), why, bounded?}]} (rule
#            focal.fusion; a fail adds the measurement check "focal"); a measurement's sigma text is "inputs"."sigma_how"
#            with "inputs"."chain_piece" {piece, events, window_short, events_near_t} (rule fit.sigma_chain) or, when
#            the fit has no chain piece, the text "inputs"."sigma"; "inputs"."sigma_events" {one_sided, event_unstable}
#   fused    a scene with "f_source" "camera.fused" has "focal"."fused" {f, sigma, n_sources, how, was {f, source}}
#            (rule C1a) and no other scene has it; "focal"."agency_unknown" [{object, why}] when present; a motion law
#            may have "agency" "untestable" with "why" (rule agency.unknown); every fit measurement with a piece has
#            "inputs"."chain_piece" with "chain" (bool) and "inputs"."piece" equal to its "piece" (rule C1b); a
#            constraint residual with "pass" null has "untestable" {rule, sigma_pred, source, reason} and
#            "checks"."constraints" has "untestable_ids" / "untestable" (rule G13); a failed unit's answer record has
#            "error" (StageError: objects.llm_invalid / scenegraph.llm_invalid ...) and a timing prefix; a scene's
#            plausible parts may have "untestable" [reason] (critic.agency_unknown)
#   impulse  every physics record has "impulse" [{t, event, agent, source, verdict (kick / agent.unobserved / rejected:
#            .. / chain ends / no_law), why, jump_px_s?, sigma_px_s?, jump_how?}] equal to its motion law's (rule
#            law.impulse), "onset" {phase: {t, sigma, how}} and "incline_law" [{support, t_on, sigma_t_on, accel,
#            sigma_accel, accel_source, accel_geometry, kappa, shape, theta_deg, law}] (rule support.incline); the scene
#            has "support_incline" {support: {method, theta_deg, sigma_deg, length_m, inclined, up {level, angle_deg,
#            sigma_deg, source, why}, why}} and "incline_up"; a measurement may have "kinematic_onset" {pass, t_on,
#            sigma_t_on, v, integral, sigma, bound, reason} (rule kinematic.onset); the scene graph cache has "format" 2:
#            every edge a "source" (agent / per-frame / grid) and no "conf", "conflicts" [{name, from, source, t_start,
#            t_end, edges, why}] (rule scenegraph.overlap); the presence record
#            outputs/objects_<tag>_presence_v4/<unit>.json has {name: {present, rule, votes, zoom_votes?}} and no
#            confidence (rule decisions.no_conf)
#   contact  the scene has "scale_contact" ({} without a prior scale; {name, object, pass, reason, events [{t, other,
#            kind, source, Z_scale, sigma_scale, Z_other, sigma_other, n_frames, gap_s, z, pass, why}], refit?, after?},
#            rule scale.contact; with "launch" no refit, see there),
#            "main_plane" ({} unless the plane is the main support's own, rule support.main; refit {was, now} by rule
#            support.level_refit) and support_plane.tests.level {angle_deg, sigma_deg (inf allowed), sigma_parts_deg};
#            every scene object has "fit_object" {name, pass, s2, dof, bound, chi2, n_fits, rows, reason} (rule
#            fit.object); a constraint residual of a failed object fit has "inadmissible" {rule, reason} and
#            "checks"."constraints" has "inadmissible_ids" / "inadmissible"; a focal.fusion source may have verdict
#            "excluded" or "no sigma" (listed, not counted in n_sources); a scene-graph edge or unknown span may have
#            "event" "flight_end" (zero length allowed, rule scenegraph.flight_end); a presence record may have
#            "reason", "zoom_errors" / "errors" [{where, type, msg}] and zoom votes true / false / null (unknown)
#            (rules presence.split_vote, errors.recorded); a main-plane roll refit has no "rms"
#   interval a held_by -> airborne release has "t_release" [lo, hi], "release_rule" {rule "release.interval", lo, hi} on
#            both edges and "release_why" {why, lo, hi, last_contact, first_no_contact, bounce, frames} on the airborne
#            edge (rule release.interval); a physics event's "t" may be that interval, with "t_fit_release"; a motion
#            law may have "release_fit" {name "release.fit", t, lo, hi, t_stop, sigma_px, tried, why, ..} (rule
#            window.release); every scale_contact event has "reach" ({point, H, H_how, sigma_reach} or null) and
#            "sigma_plane_depth", a contact refit has "rescaled" (rules scale.contact_reach, scale.contact_rescale);
#            the scene has "depth_rows" {object: {name "direct.depth_agree", pass, z_mask, z_plane, bound, n_rows,
#            n_matched, dropped, reason}}; every fit_object has "stated" [{kind, residual, sigma_stated, sigma_pred,
#            z}], "box_noise_px", "box_bias_px", "box_noise_test" (null without box rows) and "state_mismatch",
#            recorded facts (rule fit.object_noise); every presence verdict has "reason"
#   span     every track has "ends" {start, end: {frame, reason (out_of_frame / occluded / presence_no / clip_end), why}}
#            on its first and last boxed frame and "overlap_same_class" [frames] (rule track.visible_span); every scene
#            object has "size.full_extent" (null unless an articulated agent sizes it) {name, pass, reason, frames_used,
#            frames_excluded {overlap, shrink, border, no_depth}, extent_px, extent_norm, ref_norm, depth, ..}, so has a
#            direct size of that object (rule size.full_extent); a result after an accepted kick has "window_impulse"
#            {name, pass (the direct minimal window's), reason, onset, agent, verdict, ramp_end, first_frame, piece,
#            window, speeds}, its fit / direct measurements "window_impulse" and, when testable, the check
#            "window.impulse" {pass, reason, onset, ramp_end, piece, window}; an inadmissible direct may carry it too
#            (rule window.impulse); fit_object "box_noise_px" / "box_bias_px" have "sides" ([x1, y1, x2, y2] or a
#            pendulum's [u, v, d]), "box_noise_test" may be untestable {pass null, dof, why}, "state_mismatch" is
#            never null; a direct "direct.depth_agree" copied from the scene's depth_rows has "see" "depth_rows" with
#            n_rows / n_matched / dropped (rule fit.object.record); the scene has no "caches"."objects_mode"
#   launch   an "on S" -> airborne boundary may carry "t_launch" [lo, hi] and "launch_rule" {rule "state.launch",
#            support} on both edges, the airborne edge "launch_why" {why, last_contact [frame, t], first_no_contact
#            [frame, t], n_off, n_frames, testable, untestable_why, bound_px, frames}, an inserted edge "event" "launch"
#            (rule state.launch); a label-only edge test may be the failed check "state.grounded" / "held_by.agent_box"
#            (in "absorbed" / "unknown") and a tested edge carries it under "values" (rule state.label_only); the scene
#            camera has "plane_fusion" {name, pass, z, h, sigma, k, h_depth_model, sources [{source, h, sigma, how,
#            other?, n_events?, t?}], rescaled?, reason} (null in a scene without one; mirrored in
#            camera.support_plane.plane_fusion), "plane_sigma" {extrapolation {obj: {row, Z_row, mask_depth, lever_m,
#            ratio, phi_frame_deg, rel, beyond, odd_rel, plane_rel, how}}, odd {obj: rel}} or null, a measurement's
#            sigma_parts "plane.fusion" (rule plane.fusion; no "scale.contact_refit", no scale_contact "refit", no
#            camera.support_plane.contact_refit; scale_contact "after" exactly when the fusion rescaled, every event
#            "sigma_line" / "sigma_plane_angles" / "extrapolation"); a depth_rows record has "odd" (null unless it
#            fails: "plane" keeps the rows, "silhouette" drops them) and, when the ground was sampled, "ground" {z_ground,
#            n, pass, bound, reason} (rule depth_rows.odd); dim_source may be "class"; every scene object has
#            "dims.agent_lateral" {name, pass, <dim>_source, box_w_px_median, box_w_m, fits_box, reason} and
#            "dims.agent_height" {name, pass, pitch_deg, ray_angle_deg, threshold_deg, H_source, plane, reason}, both
#            null unless the object is an articulated agent (rules dims.agent_lateral / dims.agent_height), and a
#            size.full_extent may exclude every frame by "height"; a physics chain's phases may be "driven" (no law),
#            with "impulse_span" {name, onset, verdict, ramp_end, first_frame, t_stop, n_driven_frames, speeds, how,
#            untestable, applied, free_piece} (its motion law's copy without applied / free_piece) and the onset event
#            of events_graph its "impulse_span" {onset, ramp_end, n_driven_frames, how} (rule law.impulse_span)
#   classref the object inventory has format 7 (rule C15 scale.class_reference, part 1): "class_table" (pipeline.class_sizes
#            TABLE_VERSION), "rendered", "references" {status}, "reference_rejected" [{name, class, reason, why}], every
#            object "size_class", "size_reference" (pipeline.class_sizes.size_reference of its class, or null) and
#            "self_reference", a reference object "extra" {role "reference", reason}; the scene has
#            "scale.class_reference" {name, mode s / f, votes [..], used, skipped [{name, reason, why}], carries_scale,
#            stated, value, sigma, pass, reason} (mirrored in "scale"."class_reference" of a prior-scaled scene, its
#            class-reference rows in focal.fusion.sources with verdict "cross-check" for a depth-stated one), every
#            object "role" and "size_reference" (role reference: dim_source class for every dim), camera["scale.source"]
#            and results[i]["scale.source"] (prior / depth_info / class / none), camera keys f_source / focal_spread /
#            support_plane / support_rejected / s / metric / plane_fusion always present (null with depth statements),
#            depth_rows[obj] n_matched / odd / ground / plane_sigma_added always present (rule record.one_shape), a
#            physics chain's "gate" {name physics.gate, pass, rms_chain, rms_kinematic, frames, bound, rms_chain_law,
#            frames_law, rms_kinematic_all, frames_kinematic_all, reason} (rule physics.gate_frames), a result's
#            window_impulse "measurement" "direct" (rule window_impulse.measurement) and a prior scale's "full_extent"
#            {agent: {variant, depth, pass, frames_used, frames_excluded}} (rule size.full_extent_nominal)
#   sizeonly a reference object is a size read: the scene's "references" and the answer record's plausible "excluded"
#            (rule C15c reference.size_only); a partial record names its planning_failed round
#   undetermined (batch 24) the camera has "focal.undetermined" (null, or {name, pass, reason, stage_a_f, stage_a_sigma,
#            stage_a_how, kept false, f, sigma, sources_used, fusion_pass, sources [{object, kind, f, sigma, verdict,
#            why}], depth_model {f, sigma, how, per_frame}, repair {class_skipped [{name, class, reason, why}],
#            depth_model}, refit_fusion {pass, f, sigma, reason}}; a passing record sets the scene's "f_source"
#            "focal.fusion", rule C17) and "prior.untracked" (reason | null; every result of the unit then fails the check
#            "prior.untracked" with that reason and the agent offers track_reselect on the prior's object, rule D24);
#            plausible "excluded" has exactly "reference.size_only" / "structure.static" / "buoyant", with an optional
#            "excluded_why" (rules D25 critic.structure_static / critic.buoyant); a direct measurement with a value has
#            "plausible_bounds" {lo, hi, prior_m, pass} (rule D25b); the ground record outputs/ground_<tag>/<unit>.json
#            has "version" 4 with "floor_frames" [{frame, coverage_raw, coverage_minus_objects, objects_covered, kept,
#            reason}] and each "floor" entry coverage_raw / coverage_minus_objects (rule C19 support.full_frame); a
#            direct-only scene (pipeline.state.direct_only: stage A failed, objects {}, no camera) is read as an
#            unresolved question whose results fail "fit" (rule D24 scene.direct_only)
#   unobserved (batch 26) the graph has "unobserved" phases [{from, state, t_start, t_end, rule, visibility, cut}]
#            (rule A12 state.occluded_not_airborne) and edges may carry launch_moved / launch_removed (rule A14
#            graph_edit.launch_pair); choose records carry choice_frame {frame, why, shared, shared_status, rule} (E20),
#            rule / rejected / accepted_frame / binding (E17 choose.track_binding, = identity binding); a driven span and
#            a result's window_impulse have "kick_ramp" {pair_speeds, ramp_end, why} (B29); a pendulum has
#            "swing_period" and may have no period (B30); windows and pieces carry frame indices beside their seconds
#            (D31 time.one_grid); agent rounds carry the replayed constraint contract residuals and untestable ids (D30),
#            graph_edit.unobserved_kept on results / rejected calls (A16); a measurement's constraint_residuals hold the
#            contained rows (D30)
#   rendered (batch 27) inventory format 9: "rendered" is {verdict, answer, cues [{kind, where, why}], cues_rejected
#            [{.., reason}], why, reason null | rendered.unsupported} (C15d); the scene has "unobserved.no_new_law" (B31),
#            "support.undetermined_soft" (also on the plane and on each softened chain's motion law, C23), camera
#            "focal.refit_fixed_point" (C22; class skip reason focal.undetermined); a result has "answer.span" (B31); a
#            hanging object's law is pendulum / held_static with "pendulum.oscillates" (A15); a rejected physics record
#            may hold only the plan; a round has "visibility.support_plane" (D32); forward.near is the fit's own depth
#   up      (batch 28) the scene has "gravity.up" {name, up, up_source, roll {deg, source}, pitch {deg, source},
#            skipped [{source, why}], why} (C23b gravity.up_not_from_soft_plane): a soft support plane never supplies
#            the up; gravity.up_source carries the same string, also on a scene without a metric scale
#   anchored (batch 29) a track has "mask" {components {dropped [{frame, area, box, why}], leak [frames], rule}} or null
#            (E21 track.mask_one_component); a class vote has "anchored" {t, Z, source "depth_info", f [], readings} or
#            null, an anchored vote has extent null, an anchored skip may carry "anchored_readings", skip reasons
#            dim.truncated / dim.occluded, and the refit's first pass may skip a class object before its fit (C15e
#            focal.anchored_class_vote); an overhead plan may be "static_above" with "plan.overhead_seen_from_below",
#            listed at the scene's top level too, answer method fit-box-clearance (D34)
STEPS_V4 = ("objects", "scenegraph_grid", "track", "scenegraph_pf", "observe", "scene", "agent", "answer")
STEPS_V5 = ("objects", "scenegraph_grid", "observe", "track", "measure", "scenegraph_pf", "ground", "scene", "agent",
            "answer")
VERSION_FORMAT = {
    "v1": {"motion": "ballistic", "paths": ("segments",), "unit": "video", "decide": "conf"},
    "v2": {"motion": "physics", "paths": ("physics", "segments"), "unit": "video", "decide": "conf"},
    "v3": {"motion": "physics", "paths": ("physics", "segments"), "unit": "video", "decide": "conf"},
    "v4": {"motion": "physics", "paths": ("physics", "segments"), "unit": "question", "decide": "checks",
           "steps": STEPS_V4},
    "v5": {"motion": "physics", "paths": ("physics", "segments"), "unit": "question", "decide": "agree",
           "steps": STEPS_V5, "identity": True, "gravity": True, "release": True, "refmeasure": True, "kept": True,
           "pendulum": True, "selftest": True, "horizon": True, "camera": True, "roll": True,
           "static": True, "fusion": True, "fused": True, "impulse": True, "contact": True, "interval": True,
           "span": True, "launch": True, "classref": True, "sizeonly": True,
           "undetermined": True, "contained": True, "unobserved": True, "rendered": True, "up": True,
           "anchored": True},
}
# v6: the v5 harness (batch 29 plus the kept parts of 2026-11-05) as the perception below the scene-program chain;
# its harness artifacts have the v5 format, and every question also carries its scene-program products
# (outputs/sceneprog/<build tag>/<unit> and the answer step's table, see sceneprog_view)
VERSION_FORMAT["v6"] = {**VERSION_FORMAT["v5"], "sceneprog": True}
CHECK_SOURCES = ("scene", "measurement")   # sources of a resolved answer under "checks"; unresolved: "unresolved"
MOTION_KEYS = ("ballistic", "physics")
FMT: dict = {}     # VERSION_FORMAT entry of the version being built
DIRS: dict = {}    # artifact dirs of the version being built
AT = {"where": ""}  # "<version> <video or unit>" being read, for error messages


TAG_RE = re.compile(r"(v\d+)(gpt)?(_control)?")


def base_version(tag: str) -> str:
    """The version of a run tag: "v5gpt" -> "v5" (a backend suffix), "v4_control" -> "v4"."""
    m = TAG_RE.fullmatch(tag)
    if not m:
        raise ValueError(f"{tag}: not a run tag <version>[gpt][_control]")
    return m.group(1)


def version_format(tag: str) -> dict:
    base = base_version(tag)
    if base not in VERSION_FORMAT:
        raise ValueError(f"{tag}: no VERSION_FORMAT entry for {base}; declare its artifact format first")
    return VERSION_FORMAT[base]


def run_label(tag: str, pipeline: str) -> bool:
    """Is `pipeline` (an answer record's version.pipeline) the label of run `tag`? The harness writes the tag, with
    "-dirty" when the run's working tree had uncommitted changes."""
    t = tag.removesuffix("_control")
    return pipeline in (t, f"{t}-dirty")


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


def req(d: dict, key: str, what: str):
    """d[key], a field every artifact of the version's format has; its absence is a format mismatch, not a default."""
    if key not in d:
        raise ValueError(f"{AT['where']}: {what} has no {key!r} field")
    return d[key]


def read_json(p: Path, what: str):
    if not p.exists():
        raise ValueError(f"{what} missing: {p.relative_to(ROOT)}")
    return json.loads(p.read_text())


SUPPORT_RELS = ("on", "in", "held_by", "attached_to", "above", "airborne")   # contact edges are not support


# the field naming the video in each per-unit artifact
VIDEO_FIELD = {"scene graph": "video", "track cache": "video_id", "object inventory": "video_id",
               "question scene": "video_id", "answer record": "video_id"}


def unit_json(kind: str, key: str, what: str):
    """The artifact DIRS[kind]/<key>.json; for unit "question" its "unit" and video fields must name that question."""
    j = read_json(DIRS[kind] / f"{key}.json", what)
    if FMT["unit"] == "question":
        vid = key.rsplit("_q", 1)[0]
        if req(j, "unit", what) != key or req(j, VIDEO_FIELD[what], what) != vid:
            raise ValueError(f"{AT['where']}: {what} {DIRS[kind].name}/{key}.json is for unit {j.get('unit')!r} "
                             f"/ video {j.get(VIDEO_FIELD[what])!r}, not {key}")
    return j


def graph_references(g: dict) -> list:
    """The scene graph's references (format 3, rule reference.size_only): [{"name", "class", "frames"}], no state, edge
    or event; a reference named by a node, an edge, an unknown span or a per-frame phase raises."""
    refs = []
    for r in req(g, "references", "scene graph cache"):
        w = f"scene graph reference {r.get('name')!r}"
        if set(r) != {"name", "class", "role", "frames"} or r["role"] != "reference":
            raise ValueError(f"{AT['where']}: {w} keys {sorted(r)} role {r.get('role')!r}")
        refs.append({"name": text(r["name"]), "class": text(r["class"]), "frames": frame_runs([int(f) for f in r["frames"]])})
    names = {r["name"] for r in refs}
    used = {text(n.get("id")) for n in req(g, "nodes", "scene graph cache")} | set(map(text, req(g, "objects", "scene graph cache"))) \
        | {text(e.get(k)) for e in req(g, "edges", "scene graph cache") + req(g, "unknown", "scene graph cache") for k in ("from", "to")}
    if names & used:
        raise ValueError(f"{AT['where']}: scene graph references {sorted(names & used)} also in its nodes / objects / edges")
    return refs


def frame_runs(fs: list[int]) -> str:
    """Sorted frames as runs: [0, 1, 2, 5] -> "0-2, 5"."""
    runs, fs = [], sorted(fs)
    for f in fs:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


def sgraph(key: str, rec: dict | None = None, state: dict | None = None):
    """The run's scene graph cache outputs/scenegraph_<tag>/<key>.json as written (with FMT "kept": the graph the
    answer used, the answer record's kept_state.scenegraph, and "kept" its agent round), for the tree on the page:
    {"nodes": [{"id", "type": object | support | holder}], "edges": [{"from", "to" (null: airborne), "rel", "t0", "t1"}]}
    (t0 / t1 null = the whole video; contact edges dropped); with FMT "release" also "releases" [{"obj", "t", "rule",
    "why"}] (one per held -> airborne release) and "untracked" [[name, why]]; with FMT "kept" "rewritten" [{"obj", "rel",
    "to", "t0", "t1", "was", "rule"}] (edges of the scene's own graph rewritten by scene_fit). None when it has none."""
    if FMT.get("kept"):
        ks = req(rec, "kept_state", "answer record")
        g = read_json(ROOT / text(req(ks, "scenegraph", "kept_state")), "kept scene graph")
        if req(g, "unit", "kept scene graph") != key:
            raise ValueError(f"{AT['where']}: kept scene graph is for unit {g['unit']!r}")
    else:
        g = unit_json("sgraph", key, "scene graph")
    if FMT.get("impulse"):
        fmt = 3 if FMT.get("sizeonly") else 2
        if req(g, "format", "scene graph") != fmt:
            raise ValueError(f"{AT['where']}: scene graph format {g['format']!r}, expected {fmt}")
        for e in req(g, "edges", "scene graph cache"):
            if "conf" in e or req(e, "source", "edge") not in EDGE_SOURCES:
                raise ValueError(f"{AT['where']}: edge {e.get('from')!r} {e.get('rel')!r} source {e.get('source')!r}"
                                 f"{' with a conf' if 'conf' in e else ''}")
    edges = [{"from": text(req(e, "from", "edge")), "to": text(req(e, "to", "edge")) or None,   # null "to": airborne
              "rel": text(e["rel"]), "t0": num(req(e, "t_start", "edge"), 4), "t1": num(req(e, "t_end", "edge"), 4)}
             | ({"source": text(e["source"])} if FMT.get("impulse") else {})
             | ({"event": edge_event(e, f"edge {e.get('from')!r} {e.get('rel')!r}")} if FMT.get("contact") and "event" in e else {})
             | ({"test": test_view(e["test"], f"edge {e.get('from')!r} {e.get('rel')!r}")} if "test" in e else {})
             | ({"containment": containment_view(e)} if FMT.get("contained") and "containment" in e else {})
             | ({k: launch_edit_view(e, k) for k in ("launch_moved", "launch_removed") if k in e}
                if FMT.get("unobserved") else {})
             | ({"absorbed": [span_edge(x, f"edge {e.get('from')!r} {e.get('rel')!r} absorbed") for x in e["absorbed"]]}
                if FMT.get("selftest") and e.get("absorbed") else {})
             for e in req(g, "edges", "scene graph cache") if req(e, "rel", "edge") in SUPPORT_RELS]
    nodes = [{"id": text(req(n, "id", "scene graph node")), "type": text(req(n, "type", "scene graph node"))}
             for n in req(g, "nodes", "scene graph cache")]
    out = {"nodes": nodes, "edges": edges} if edges else {}
    if FMT.get("sizeonly"):
        refs = graph_references(g)
        if refs:
            out["references"] = refs
    if FMT.get("release"):
        rel = {}     # (object, t_release) -> {"obj", "t", "rule", "why"}: the held_by and the airborne edge of one release
        for e in req(g, "edges", "scene graph cache"):
            if "t_release" not in e:
                continue
            w = f"edge {e.get('from')!r} {e.get('rel')!r}"
            t = [num(x, 4) for x in req(e, "t_release", w)]
            if FMT.get("interval"):
                rr = req(e, "release_rule", w)
                if not isinstance(rr, dict) or req(rr, "rule", f"{w} release_rule") != "release.interval" or \
                        len(t) != 2 or None in t or t[0] > t[1]:
                    raise ValueError(f"{AT['where']}: {w} release_rule {rr!r} t_release {t}")
                rule = {"rule": "release.interval", "lo": text(req(rr, "lo", f"{w} release_rule")),
                        "hi": text(req(rr, "hi", f"{w} release_rule"))}
            else:
                rule = text(req(e, "release_rule", w))
                if rule not in ("release.last_contact", "release.free_flight"):
                    raise ValueError(f"{AT['where']}: {w} release_rule {rule!r}")
            r = rel.setdefault((text(e["from"]), tuple(t)), {"obj": text(e["from"]), "t": t, "rule": rule, "why": None})
            if r["rule"] != rule:
                raise ValueError(f"{AT['where']}: {w} release_rule {rule!r} differs from its other edge's {r['rule']!r}")
            if e["rel"] == "airborne":
                if FMT.get("interval"):
                    rw, ww = req(e, "release_why", w), f"{w} release_why"
                    if [num(req(rw, "lo", ww), 4), num(req(rw, "hi", ww), 4)] != t:
                        raise ValueError(f"{AT['where']}: {ww} [lo, hi] differs from t_release {t}")
                    r["why"] = text(req(rw, "why", ww))
                    r["last_contact"] = num(req(rw, "last_contact", ww), 4)
                    r["first_no_contact"] = num(req(rw, "first_no_contact", ww), 4)
                    r["bounce"] = num(req(rw, "bounce", ww), 4)
                    req(rw, "frames", ww)
                else:
                    r["why"] = text(req(e, "release_why", w))
        if FMT.get("interval") and any(r["why"] is None for r in rel.values()):
            raise ValueError(f"{AT['where']}: a release interval without its airborne edge's release_why")
        if FMT.get("launch"):
            la = launch_view(g)
            if la:
                out["launches"] = la
        unt = req(g, "untracked", "scene graph cache")
        if rel:
            out["releases"] = list(rel.values())
        if unt:
            out["untracked"] = [[text(k), text(v)] for k, v in unt.items()]
    if FMT.get("selftest"):
        st = req(g, "self_test", "scene graph cache")
        out["self_test"] = {k: int(req(st, k, "self_test")) for k in ("tested", "fail", "merged", "unknown")}
        unk = [span_edge(x, "unknown span") | {"obj": text(req(x, "from", "unknown span")),
                                                 "reason": text(req(x, "reason", "unknown span"))}
               | ({"event": edge_event(x, "unknown span")} if FMT.get("contact") and "event" in x else {})
               for x in req(g, "unknown", "scene graph cache")]
        if unk:
            out["unknown"] = unk
    if FMT.get("unobserved"):
        uo = [unobserved_view(x) for x in req(g, "unobserved", "scene graph cache")]
        if uo:
            out["unobserved"] = uo
    if FMT.get("impulse"):
        cf = [conflict_view(x) for x in req(g, "conflicts", "scene graph cache")]
        if cf:
            out["conflicts"] = cf
    if FMT.get("kept") and state is not None:
        rw = []
        for e in req(req(state, "scene_graph", "scene"), "edges", "scene graph"):
            if "was" not in e:
                continue
            w = f"scene graph edge {e.get('from')!r} {e.get('rel')!r}"
            was = req(e, "was", w)
            rw.append({"obj": text(req(e, "from", w)), "rel": text(req(e, "rel", w)), "to": text(e.get("to")) or None,
                       "t0": num(req(e, "t_start", w), 4), "t1": num(req(e, "t_end", w), 4),
                       "was": {"rel": text(req(was, "rel", f"{w} was")), "to": text(was.get("to")) or None},
                       "rule": text(req(e, "rule", w))})
        if rw:
            out["rewritten"] = rw
    if out and "nodes" not in out:
        out |= {"nodes": nodes, "edges": []}
    if out and FMT.get("kept"):
        out["kept"] = int(req(rec["kept_state"], "iter", "kept_state"))
    return out or None


UNOBSERVED_WHYS = ("outside_track", "track_gap", "border", "sliver", "occluded", "island")   # scenegraph.UNOBSERVED_WHY


def unobserved_view(x: dict) -> dict:
    """Rule A12 state.occluded_not_airborne: a phase where the object is not fully seen holds no state (not an edge):
    {"obj", "t0", "t1", "frames" [f0, f1], "n", "why", "runs" {why: [[a, b]]}, "detail", "cut" [{rel, to, t0, t1}]}."""
    w = "scene graph unobserved"
    if set(x) != {"from", "state", "t_start", "t_end", "rule", "visibility", "cut"} or x["state"] != "unobserved" \
            or x["rule"] != "state.occluded_not_airborne":
        raise ValueError(f"{AT['where']}: {w} {sorted(x)} state {x.get('state')!r} rule {x.get('rule')!r}")
    v = x["visibility"]
    if set(v) != {"frames_unobserved", "n", "why", "frames", "detail"} or len(v["frames_unobserved"]) != 2 \
            or any(k not in UNOBSERVED_WHYS for k in v["frames"]):
        raise ValueError(f"{AT['where']}: {w} visibility {v!r}")
    cut = []
    for c in x["cut"]:
        if set(c) != {"rel", "to", "t_start", "t_end"}:
            raise ValueError(f"{AT['where']}: {w} cut {c!r}")
        cut.append({"rel": text(c["rel"]), "to": text(c["to"]) or None, "t0": num(c["t_start"], 4), "t1": num(c["t_end"], 4)})
    return {"obj": text(x["from"]), "t0": num(x["t_start"], 4), "t1": num(x["t_end"], 4),
            "frames": [int(f) for f in v["frames_unobserved"]], "n": int(v["n"]), "why": text(v["why"]),
            "runs": {k: [[int(a), int(b)] for a, b in r] for k, r in v["frames"].items()},
            "detail": [text(d) for d in v["detail"]], "cut": cut}


LAUNCH_EDIT_KEYS = {"launch_moved": {"rule", "by", "round", "why", "from", "to"},
                    "launch_removed": {"rule", "by", "round", "why", "t_launch", "support"}}
LAUNCH_EDIT_BY = ("agent", "builder.self_test", "state.occluded_not_airborne")


def launch_edit_view(e: dict, k: str) -> dict:
    """Rule A14 graph_edit.launch_pair: a launch pair re-timed with its airborne start (launch_moved) or cleared from
    the surviving halves when the airborne half went (launch_removed)."""
    x, w = e[k], f"edge {e.get('from')!r} {e.get('rel')!r} {k}"
    if set(x) != LAUNCH_EDIT_KEYS[k] or x["rule"] != "graph_edit.launch_pair" or (k == "launch_removed" and x["by"] not in LAUNCH_EDIT_BY):
        raise ValueError(f"{AT['where']}: {w} {x!r}")
    if k == "launch_removed" and any(f in e for f in ("t_launch", "launch_rule", "launch_why")):
        raise ValueError(f"{AT['where']}: {w} on an edge that keeps its launch fields")
    out = {"by": text(x["by"]), "round": x["round"], "why": text(x["why"])}
    if k == "launch_moved":
        return out | {"from": [num(t, 4) for t in x["from"]], "to": [num(t, 4) for t in x["to"]]}
    return out | {"t_launch": [num(t, 4) for t in x["t_launch"]], "support": text(x["support"])}


CONTAINMENT_KEYS = {"rule", "verdict", "inside_frac", "bound", "frames", "level", "r_px", "why"}
CONTAINMENT_OPTIONAL = {"outside_frames", "longest_outside_run", "rows", "hull", "corrected"}
CONTAINMENT_VERDICTS = ("in", "not_in", "untestable")
CONTAINMENT_LEVELS = ("hull", "mask", "track")


def containment_view(e: dict, hull: bool = False) -> dict:
    """Rule A13 (per-frame graph): an "on" / "in" edge between two objects decided by their masks -- verdict, level
    (hull: the extents2d hulls; mask: the SAM 2 masks, after the hulls said in; track: no track), median inside
    fraction against its bound, frames with evidence, and the label it corrected; "in" -> rel in, "not_in" -> rel on."""
    w = f"edge {e.get('from')!r} {e.get('rel')!r} containment"
    c = e if hull else e["containment"]
    if not CONTAINMENT_KEYS <= set(c) or set(c) - CONTAINMENT_KEYS - CONTAINMENT_OPTIONAL or c["rule"] != "contained.same_depth" or \
            c["verdict"] not in CONTAINMENT_VERDICTS or c["level"] not in CONTAINMENT_LEVELS:
        raise ValueError(f"{AT['where']}: {w} keys {sorted(c)} verdict {c.get('verdict')!r} level {c.get('level')!r}")
    if not hull:
        rel = e.get("rel")
        if (c["verdict"] == "in" and rel != "in") or (c["verdict"] == "not_in" and rel != "on") or \
                ("corrected" in c and (rel != "in" or c["corrected"].get("from_rel") == rel)):
            raise ValueError(f"{AT['where']}: {w} verdict {c['verdict']!r} on a {rel!r} edge, corrected {c.get('corrected')!r}")
    out = {"verdict": c["verdict"], "level": c["level"], "inside_frac": num(c["inside_frac"], 3), "bound": num(c["bound"], 3),
           "frames": int(c["frames"]), "r_px": num(c["r_px"], 3), "why": text(c["why"]),
           "outside": int(c["longest_outside_run"]) if c.get("longest_outside_run") is not None else None}
    if "hull" in c and not hull:
        if c["level"] != "mask":
            raise ValueError(f"{AT['where']}: {w} a hull record on level {c['level']!r}")
        out["hull"] = containment_view(c["hull"], hull=True)
    if "corrected" in c:
        cr = c["corrected"]
        out["corrected"] = {"from_rel": text(req(cr, "from_rel", f"{w} corrected")), "source": text(req(cr, "source", f"{w} corrected"))}
    return out


TEST_RESULTS = ("pass", "fail", "untestable")
EDGE_EVENTS = ("flight_end",)   # pipeline.scenegraph: an edge that is an event, kept at zero length (scenegraph.flight_end)


def edge_event(e: dict, what: str) -> str:
    """The event an edge of the graph stands for (rule scenegraph.flight_end; with FMT "launch" also "launch", an
    airborne edge inserted by rule state.launch, which then carries t_launch): its "event" field, a known name."""
    ev = text(e["event"])
    if ev == "launch" and FMT.get("launch"):
        if e.get("rel") != "airborne" or "t_launch" not in e:
            raise ValueError(f"{AT['where']}: {what} event 'launch' on a {e.get('rel')!r} edge without t_launch")
        return ev
    if ev not in EDGE_EVENTS:
        raise ValueError(f"{AT['where']}: {what} event {ev!r}")
    return ev


def launch_view(g: dict) -> list[dict]:
    """Every launch off a support's end (rule state.launch): the "on S" edge and the airborne edge after it share
    t_launch [lo, hi] and launch_rule {rule "state.launch", support S}; the airborne edge's launch_why gives the last
    contact and first no-contact frames (= [lo, hi]), the off-support frames and whether the flight is testable:
    [{"obj", "t", "support", "why", "last_contact", "first_no_contact", "n_off", "n_frames", "testable",
    "untestable_why", "bound_px", "inserted"}]."""
    la = {}
    for e in req(g, "edges", "scene graph cache"):
        if "t_launch" not in e and "launch_rule" not in e and "launch_why" not in e:
            continue
        w = f"edge {e.get('from')!r} {e.get('rel')!r}"
        t = [num(x, 4) for x in req(e, "t_launch", w)]
        lr = req(e, "launch_rule", w)
        if len(t) != 2 or None in t or t[0] > t[1] or not isinstance(lr, dict) or set(lr) != {"rule", "support"} \
                or lr["rule"] != "state.launch":
            raise ValueError(f"{AT['where']}: {w} t_launch {t} launch_rule {lr!r}")
        rel = text(req(e, "rel", w))
        if (rel == "on") != (text(e.get("to")) == text(lr["support"])) or rel not in ("on", "airborne") \
                or (rel == "airborne") != ("launch_why" in e):
            raise ValueError(f"{AT['where']}: {w} launch_rule support {lr['support']!r} on a {rel!r} edge "
                             f"{'with' if 'launch_why' in e else 'without'} launch_why")
        r = la.setdefault((text(e["from"]), tuple(t)), {"obj": text(e["from"]), "t": t, "support": text(lr["support"]),
                                                       "edges": set(), "why": None})
        if r["support"] != text(lr["support"]) or rel in r["edges"]:
            raise ValueError(f"{AT['where']}: {w} launch {t} support {lr['support']!r} differs from its other edge's")
        r["edges"].add(rel)
        if rel == "on":
            r["on_end"] = num(req(e, "t_end", w), 4)
        if rel == "airborne":
            lw, ww = req(e, "launch_why", w), f"{w} launch_why"
            lc, fn = req(lw, "last_contact", ww), req(lw, "first_no_contact", ww)
            if [num(lc[1], 4), num(fn[1], 4)] != t:
                raise ValueError(f"{AT['where']}: {ww} last_contact {lc} / first_no_contact {fn} differ from t_launch {t}")
            tb = req(lw, "testable", ww)
            if not isinstance(tb, bool) or (tb == bool(req(lw, "untestable_why", ww))):
                raise ValueError(f"{AT['where']}: {ww} testable {tb!r} with untestable_why {lw.get('untestable_why')!r}")
            for f in req(lw, "frames", ww):
                if len(f) != 3 or not isinstance(f[2], bool):
                    raise ValueError(f"{AT['where']}: {ww} frame {f!r}")
            r |= {"why": text(req(lw, "why", ww)), "last_contact": [int(lc[0]), num(lc[1], 4)],
                  "first_no_contact": [int(fn[0]), num(fn[1], 4)], "n_off": int(req(lw, "n_off", ww)),
                  "n_frames": int(req(lw, "n_frames", ww)), "testable": tb,
                  "untestable_why": text(lw["untestable_why"]) or None, "bound_px": num(req(lw, "bound_px", ww), 3),
                  "inserted": e.get("event") == "launch"}
    out = []
    for r in la.values():
        r["replaced"] = None
        if r["edges"] == {"on"}:
            # the airborne half was overwritten by an agent edit (scenegraph_set_relation): the agent's edge of the same
            # object starts where the "on" half ends; the launch record then has no flight of its own
            rep = [e for e in g["edges"] if text(e.get("from")) == r["obj"] and e.get("source") == "agent"
                   and num(e.get("t_start"), 4) == r["on_end"] and "t_launch" not in e and "launch_rule" not in e]
            if len(rep) != 1:
                raise ValueError(f"{AT['where']}: launch of {r['obj']!r} at {r['t']} on edges ['on'] and no agent edge from {r['on_end']}")
            e = rep[0]
            r |= {"replaced": {"rel": text(e["rel"]), "to": text(e.get("to")) or None, "t0": num(e["t_start"], 4),
                               "t1": num(req(e, "t_end", "edge"), 4)},
                  "last_contact": None, "first_no_contact": None, "n_off": None, "n_frames": None, "testable": None,
                  "untestable_why": None, "bound_px": None, "inserted": False}
        elif r["why"] is None or (r["edges"] != {"on", "airborne"} and not r["inserted"]):
            raise ValueError(f"{AT['where']}: launch of {r['obj']!r} at {r['t']} on edges {sorted(r['edges'])}")
        del r["edges"]
        r.pop("on_end", None)
        out.append(r)
    return out
EDGE_SOURCES = ("agent", "per-frame", "grid")   # pipeline.scenegraph.EDGE_SOURCES


def conflict_view(x: dict) -> dict:
    """An overlap of two same-source state edges of one object (rule scenegraph.overlap): {"obj", "source", "t0", "t1"
    (null: open), "edges": [{"rel", "to", "t0", "t1"} x 2], "why"}."""
    w = "scene graph conflict"
    if req(x, "name", w) != "scenegraph.overlap":
        raise ValueError(f"{AT['where']}: {w} name {x['name']!r}")
    es = req(x, "edges", w)
    if len(es) != 2:
        raise ValueError(f"{AT['where']}: {w} has {len(es)} edges")
    return {"obj": text(req(x, "from", w)), "source": text(req(x, "source", w)), "t0": num(req(x, "t_start", w), 4),
            "t1": num(req(x, "t_end", w), 4),
            "edges": [{"rel": text(req(e, "rel", w)), "to": text(e.get("to")) or None, "t0": num(req(e, "t_start", w), 4),
                       "t1": num(req(e, "t_end", w), 4)} for e in es], "why": text(req(x, "why", w))}


def span_edge(x: dict, what: str) -> dict:
    """A tested span of a scene graph (an absorbed or unknown edge): {"rel", "to", "t0", "t1", "test"}."""
    return {"rel": text(req(x, "rel", what)), "to": text(x.get("to")) or None, "t0": num(req(x, "t_start", what), 4),
            "t1": num(req(x, "t_end", what), 4), "test": test_view(req(x, "test", what), what)}


LABEL_ONLY = ("state.grounded", "held_by.agent_box")   # rule state.label_only (pipeline.scenegraph)


def test_view(t: dict, what: str) -> dict:
    """The geometric test of an agent state edit (rule graph_edit.retest): {"name", "result", "why"}; with FMT
    "launch" a label-only check that did not fail is kept under the test's values (rule state.label_only): "label"
    [{"name", "result", "why"}]; a failed one is the test itself, its old test under its values."""
    r = text(req(t, "result", f"{what} test"))
    if r not in TEST_RESULTS:
        raise ValueError(f"{AT['where']}: {what} test result {r!r}")
    out = {"name": text(req(t, "name", f"{what} test")), "result": r, "why": text(req(t, "why", f"{what} test"))}
    if FMT.get("launch"):
        vals = t.get("values") or {}
        if out["name"] in LABEL_ONLY and r != "fail":
            raise ValueError(f"{AT['where']}: {what} test {out['name']} result {r!r} is not the edge's test")
        lab = []
        for k in LABEL_ONLY:
            if k not in vals:
                continue
            x, wx = vals[k], f"{what} test values {k}"
            if req(x, "name", wx) != k or req(x, "result", wx) not in TEST_RESULTS or out["name"] in LABEL_ONLY:
                raise ValueError(f"{AT['where']}: {wx} name {x.get('name')!r} result {x.get('result')!r}")
            if x["result"] == "fail":
                raise ValueError(f"{AT['where']}: {wx} failed but is not the edge's test")
            lab.append({"name": k, "result": x["result"], "why": text(req(x, "why", wx))})
        if lab:
            out["label"] = lab
    return out


def track_intervals(key: str):
    """{track name: [[t0, t1], ...]} — the frames where the track cache outputs/tracks_<tag>/<key>.json has a box,
    merged into runs of consecutive frames, in seconds (frame / fps of the cache). None when it has no track."""
    j = unit_json("tracks", key, "track cache")
    fps = j["info"]["fps"]
    out = {}
    for name, tr in req(j, "tracks", "track cache").items():
        fr = sorted(int(k) for k, b in req(tr, "boxes", f"track {name!r}").items() if b is not None)
        runs = []
        for f in fr:
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        out[name] = [[num(a / fps, 5), num(b / fps, 5)] for a, b in runs]
    return out or None


TRACK_END_REASONS = ("out_of_frame", "occluded", "presence_no", "clip_end")


def track_ends(key: str):
    """Each track's span with the reason it starts and ends there and the frames it overlaps another instance of its
    class (rule track.visible_span): {"fps", "tracks": {name: {"ends": {"start" / "end": {frame, reason, why}},
    "overlap_same_class": [frames]}}}; the start / end frames are the track's first / last boxed frame. None when it has no track."""
    j = unit_json("tracks", key, "track cache")
    out = {}
    for name, tr in req(j, "tracks", "track cache").items():
        w = f"track {name!r}"
        ends, ov = req(tr, "ends", w), req(tr, "overlap_same_class", w)
        if ends is None or ov is None:
            raise ValueError(f"{AT['where']}: {w} ends {ends!r} / overlap_same_class {ov!r} not recorded")
        fr = sorted(int(k) for k, b in req(tr, "boxes", w).items() if b is not None)
        row = {}
        for side, f in (("start", fr[0]), ("end", fr[-1])):
            e = req(ends, side, f"{w} ends")
            r = text(req(e, "reason", f"{w} ends {side}"))
            if r not in TRACK_END_REASONS or int(req(e, "frame", f"{w} ends {side}")) != f:
                raise ValueError(f"{AT['where']}: {w} {side} frame {e.get('frame')} reason {r!r}; boxed frames {fr[0]}-{fr[-1]}")
            row.setdefault("ends", {})[side] = {"frame": f, "reason": r, "why": text(req(e, "why", f"{w} ends {side}"))}
        row["overlap_same_class"] = sorted(int(x) for x in ov)
        if FMT.get("sizeonly"):
            # rule reference.size_only: a reference track has role "reference" and its frame_set, boxes only inside it
            role, fs = req(tr, "role", w), req(tr, "frame_set", w)
            if (role is None) != (fs is None) or role not in (None, "reference") or \
                    (fs is not None and not set(fr) <= {int(x) for x in fs}):
                raise ValueError(f"{AT['where']}: {w} role {role!r} frame_set {'set' if fs else fs!r} boxes outside it")
            row["frame_set"] = frame_runs([int(x) for x in fs]) if fs is not None else None
        if FMT.get("anchored"):
            row["mask"] = mask_view(req(tr, "mask", w), set(fr), w)
        out[name] = row
    return {"fps": num(j["info"]["fps"], 6), "tracks": out} if out else None


def mask_view(m, boxed: set, w: str):
    """Rule E21 track.mask_one_component: the mask components a track's frames dropped (detached from the object's
    reference box) and its leak frames (an edge out and back within 6 frames). None for a track without masks."""
    if m is None:
        return None
    c = m.get("components") if isinstance(m, dict) and set(m) == {"components"} else None
    if not isinstance(c, dict) or set(c) != {"dropped", "leak", "rule"} or not str(c["rule"]).startswith("track.mask_one_component"):
        raise ValueError(f"{AT['where']}: {w} mask {m!r}")
    dropped = []
    for d in c["dropped"]:
        if not isinstance(d, dict) or set(d) != {"frame", "area", "box", "why"} or len(d["box"]) != 4 \
                or int(d["frame"]) not in boxed or not int(d["area"]) > 0:
            raise ValueError(f"{AT['where']}: {w} mask dropped {d!r}")
        dropped.append({"frame": int(d["frame"]), "area": int(d["area"]), "box": [num(x, 1) for x in d["box"]],
                        "why": text(d["why"])})
    leak = sorted(int(x) for x in c["leak"])
    if not set(leak) <= boxed:
        raise ValueError(f"{AT['where']}: {w} mask leak frames {leak} outside its boxes")
    return {"dropped": dropped, "leak": leak}


def full_extent_view(fe: dict | None, what: str):
    """An articulated agent's upright, unmerged extent (rule size.full_extent): pass (None: untestable), frames used,
    frames excluded by overlap / shrink / border / no_depth, the median box height over the used frames against all
    clear frames, and the depth-normalised extent h z against its reference. None for an object it does not size."""
    if fe is None:
        return None
    w = f"{what} size.full_extent"
    if req(fe, "name", w) != "size.full_extent":
        raise ValueError(f"{AT['where']}: {w} name {fe['name']!r}")
    ok = req(fe, "pass", w)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass {ok!r}")
    ex = req(fe, "frames_excluded", w)
    keys = ("overlap", "shrink", "border", "no_depth")
    # rule dims.agent_height: a height not observed excludes every frame by "height", untestable, no frame used
    height = FMT.get("launch") and "height" in ex
    if set(ex) != set(keys) | ({"height"} if height else set()) or \
            (height and (ok is not None or req(fe, "frames_used", w) or not ex["height"])):
        raise ValueError(f"{AT['where']}: {w} frames_excluded keys {sorted(ex)} with pass {ok!r}")
    if height:
        keys += ("height",)
    return {"pass": ok, "reason": text(req(fe, "reason", w)), "used": [int(x) for x in req(fe, "frames_used", w)],
            "excluded": {k: [int(x) for x in ex[k]] for k in keys},
            "extent_px": num(req(fe, "extent_px", w), 4), "extent_px_all": num(req(fe, "extent_px_all", w), 4),
            "extent_norm": num(req(fe, "extent_norm", w), 5), "ref_norm": num(req(fe, "ref_norm", w), 5),
            "depth": text(req(fe, "depth", w))}


KICK_RAMP_END_KEYS = {"frame", "frames", "t", "v", "margin", "sigma", "bound", "against", "pass"}


def kick_ramp_view(x: dict, what: str) -> dict:
    """Rule B29 kick_ramp.significant: every frame-pair speed with its propagated sigma, and the ramp end -- the first
    pair speed maximum that exceeds a following pair by more than 3 sigma (margin vs bound, the pair it beats);
    untestable when no maximum is significant (ramp_end pass false)."""
    w = f"{what} kick_ramp"
    if set(x) != {"pair_speeds", "ramp_end", "why"}:
        raise ValueError(f"{AT['where']}: {w} keys {sorted(x)}")
    pairs = []
    for p in x["pair_speeds"]:
        if set(p) != {"frames", "v", "sigma"} or len(p["frames"]) != 2:
            raise ValueError(f"{AT['where']}: {w} pair {p!r}")
        pairs.append({"frames": [int(f) for f in p["frames"]], "v": num(p["v"], 4), "sigma": num(p["sigma"], 3)})
    re_ = x["ramp_end"]
    if set(re_) != KICK_RAMP_END_KEYS or not isinstance(re_["pass"], bool):
        raise ValueError(f"{AT['where']}: {w} ramp_end {re_!r}")
    if re_["pass"]:
        if re_["frames"] not in [p["frames"] for p in pairs] or re_["frame"] != re_["frames"][1] \
                or not float(re_["margin"]) > float(re_["bound"]):
            raise ValueError(f"{AT['where']}: {w} ramp_end {re_!r} is not a significant pair of its pair speeds")
    ints = lambda v: [int(f) for f in v] if v is not None else None
    return {"pairs": pairs, "why": text(x["why"]),
            "end": {"pass": re_["pass"], "frame": re_["frame"], "frames": ints(re_["frames"]), "t": num(re_["t"], 5),
                    "v": num(re_["v"], 4), "margin": num(re_["margin"], 4), "sigma": num(re_["sigma"], 3),
                    "bound": num(re_["bound"], 4), "against": ints(re_["against"])}}


def frames_pair(x: dict, k: str, w: str, n: int = 2):
    """A frame-index field beside a seconds field (rule D31 time.one_grid): [first, last] (an end may be null)."""
    v = req(x, k, w)
    if v is None:
        return None
    if len(v) != n or any(f is not None and not isinstance(f, int) for f in v):
        raise ValueError(f"{AT['where']}: {w} {k} {v!r}")
    return list(v)


def impulse_window_view(x: dict, what: str, result: bool = False) -> dict:
    """Rule window.impulse: the kick's ramp end, the piece after it, and the window read inside it; a result's record
    (result=True) also has the agent, the verdict, the first observed frame and the frame-pair speeds [[t, m/s]]."""
    w = f"{what} window_impulse"
    if req(x, "name", w) != "window.impulse":
        raise ValueError(f"{AT['where']}: {w} name {x['name']!r}")
    ok = req(x, "pass", w)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass {ok!r}")
    t4 = lambda v: [num(a, 5) for a in v] if v is not None else None
    out = {"pass": ok, "reason": text(req(x, "reason", w)), "onset": num(req(x, "onset", w), 5),
           "ramp_end": num(req(x, "ramp_end", w), 5), "piece": t4(req(x, "piece", w)), "window": t4(req(x, "window", w))}
    if FMT.get("unobserved"):
        out |= {"piece_frames": frames_pair(x, "piece_frames", w), "window_frames": frames_pair(x, "window_frames", w)}
    if result:
        # rule window_impulse.measurement: the result-level record is the direct measurement's verdict
        if FMT.get("classref") and req(x, "measurement", w) != "direct":
            raise ValueError(f"{AT['where']}: {w} measurement {x['measurement']!r}, expected 'direct'")
        out |= {"agent": text(req(x, "agent", w)) or None, "verdict": text(req(x, "verdict", w)),
                "first_frame": num(req(x, "first_frame", w), 5),
                "speeds": [[num(t, 5), num(v, 4)] for t, v in req(x, "speeds", w)]}
        if FMT.get("unobserved"):
            out["kick_ramp"] = kick_ramp_view(req(x, "kick_ramp", w), w) if x.get("kick_ramp") is not None else None
    return out


PATH_DT = 1 / 24     # sampling step (s) of non-parabolic object paths for the 3D viewer


def sampled_path(o: dict):
    """[[t, x, y, z], ...] of an object's centre (pipeline.scene_state3d.position), over the times its fit covers
    (to the last visibility time, at least 2 s)."""
    from pipeline.scene_state3d import position
    ends = [iv[1] for k in ("visible", "out_of_frame") for iv in req(o, k, "scene object")]
    T = max(ends + [2.0])
    n = int(round(T / PATH_DT))
    return [[num(i * PATH_DT, 4)] + [num(float(x), 5) for x in position(o, i * PATH_DT)] for i in range(n + 1)]


def fitted_physics(name: str, o: dict):
    """The object's physics record when its chain was fitted; None without one or when scene_fit rejected the chain
    (a "rejected" record keeps the plan -- model, phases, why, impulse_span -- and no fit)."""
    ph = req(o, "physics", f"scene object {name!r}")
    if ph is not None and FMT.get("rendered") and "rejected" in ph and "events" not in ph:
        if not set(ph) <= {"model", "phases", "why", "rejected", "impulse_span"} or o.get("mode") == "physics":
            raise ValueError(f"{AT['where']}: object {name!r} rejected physics record keys {sorted(ph)} mode {o.get('mode')!r}")
        return None
    return ph


def motion_events(name: str, o: dict) -> list:
    """An object's motion events ([t, kind] list) from the version's physics-law motion record (FMT["motion"])."""
    other = [k for k in MOTION_KEYS if k != FMT["motion"] and o.get(k)]
    if other:
        raise ValueError(f"object {name!r} has a {other[0]!r} motion record; this version's format is {FMT['motion']!r}")
    record = req(o, FMT["motion"], f"object {name!r}")    # null: no physics-law motion
    if record is not None and FMT.get("rendered") and "rejected" in record:
        # a rejected chain (scene_fit fit_physics: fit failed / no law after the kick: the plan only; box rms above the
        # gate: the full record) is not the object's motion
        if o.get("mode") == "physics":
            raise ValueError(f"{AT['where']}: object {name!r} rejected physics record with mode physics")
        if fitted_physics(name, o) is None:
            return []
    return [] if record is None else list(req(record, "events", f"object {name!r} motion record"))


def motion_law(name: str, o: dict):
    """The motion law of a scene object (v5 "motion_law"): {"model", "rotation": {"rel", "target", "pieces",
    "circle_candidate"} | None, "phases": [[t, state, moving]] | None}; None: no physics-law motion record."""
    law = req(o, "motion_law", f"scene object {name!r}")
    if law is None:
        return None
    rot = law.get("rotation")
    if rot is not None:
        rot = {"rel": text(req(rot, "rel", "rotation law")), "target": text(req(rot, "target", "rotation law")) or None,
               "pieces": list(req(rot, "pieces", "rotation law")),
               "circle_candidate": bool(req(rot, "circle_candidate", "rotation law"))}
    phases = [[num(p[0], 4), text(p[1]), bool(p[2])] for p in law["phases"]] if law.get("phases") else None
    out = {"model": text(req(law, "model", f"object {name!r} motion law")), "rotation": rot, "phases": phases}
    if FMT.get("fused") and law.get("agency") is not None:
        if law["agency"] != "untestable":
            raise ValueError(f"{AT['where']}: object {name!r} motion law agency {law['agency']!r}")
        out["agency"] = {"why": text(req(law, "why", f"object {name!r} motion law"))}
    if FMT.get("interval") and "release_fit" in law:
        rf, wr = law["release_fit"], f"object {name!r} motion_law release_fit"
        if req(rf, "name", wr) != "release.fit":
            raise ValueError(f"{AT['where']}: {wr} name {rf['name']!r}")
        # scene_fit.release_fit: an untestable record (too few test frames) stops before the noise and the trials
        untestable = text(req(rf, "why", wr)).startswith("untestable:")
        if untestable and (req(rf, "t", wr) is not None or "sigma_px" in rf or "tried" in rf):
            raise ValueError(f"{AT['where']}: {wr} untestable with t / sigma_px / tried")
        out["release_fit"] = {"t": num(req(rf, "t", wr), 4), "lo": num(req(rf, "lo", wr), 4), "hi": num(req(rf, "hi", wr), 4),
                              "t_stop": num(req(rf, "t_stop", wr), 4),
                              "sigma_px": None if untestable else [num(x, 3) for x in req(rf, "sigma_px", wr)],
                              "tried": None if untestable else [[num(a, 4), bool(b), int(c)] for a, b, c in req(rf, "tried", wr)],
                              "why": text(req(rf, "why", wr))}
        if FMT.get("unobserved"):
            # rule D31 time.one_grid: the release frame beside its time (no frame when no frame holds)
            if (rf.get("t") is None) != ("frame" not in rf):
                raise ValueError(f"{AT['where']}: {wr} t {rf.get('t')!r} with frame {rf.get('frame', 'absent')!r}")
            out["release_fit"]["frame"] = rf.get("frame")
    if FMT.get("rendered") and out["model"] in ("pendulum", "held_static"):
        # rule A15 pendulum.oscillates: a hanging object is a pendulum only when its track oscillates, else held static
        # at its attachment (attached_to: the attachment's name, or null for a string)
        keys = {"model", "why", "pendulum.oscillates"} | ({"attached_to"} if out["model"] == "held_static" else set())
        po = law.get("pendulum.oscillates") or {}
        if set(law) != keys or set(po) != {"pass", "period_from", "why"} \
                or po["pass"] is not (out["model"] == "pendulum") or (po["period_from"] is None) == po["pass"] \
                or po["period_from"] not in (None, "d", "u", "d+u"):
            raise ValueError(f"{AT['where']}: object {name!r} {out['model']} motion law {law!r}")
        out |= {"why": text(law["why"]), "oscillates": {"pass": po["pass"], "period_from": po["period_from"], "why": text(po["why"])}}
        if out["model"] == "held_static":
            out["attached_to"] = text(law["attached_to"]) or None
    elif FMT.get("contained") and out["model"] == "pendulum":
        # a held, hanging object's pendulum law (scene_fit.motion_law, plan held): no phases, no unknown span
        if set(law) != {"model", "why"}:
            raise ValueError(f"{AT['where']}: object {name!r} pendulum motion law keys {sorted(law)}")
        out["why"] = text(law["why"])
    elif FMT.get("sceneprog") and set(law) == {"model", "why"}:
        # a law with its reason only (scene_fit.motion_law: "scene graph: does not move"), no phases, no unknown span
        out["why"] = text(law["why"])
    elif FMT.get("selftest") and rot is None:
        out["unknown"] = [[num(a, 4), None if b >= OPEN_END else num(b, 4)]
                          for a, b in req(law, "unknown", f"object {name!r} motion law")] or None
    return out


OPEN_END = 1e8      # the scene json writes an open span end as +-1e9


def segfit_view(name: str, o: dict):
    """The motion-law test of each piece (rule segfit.point): {"model", "t0", "t1" (null: open), "point", "sigma",
    "bound", "rms", "run", "n_test", "pass" (null: untestable)}; None when the object has no segfit record."""
    sf = req(o, "segfit", f"scene object {name!r}")
    if sf is None:
        return None
    out = []
    for p in sf:
        w = f"object {name!r} segfit piece"
        t0, t1 = float(req(p, "t0", w)), float(req(p, "t1", w))
        rms = req(p, "rms", w)
        out.append({"model": text(req(p, "model", w)), "t0": None if t0 <= -OPEN_END else num(t0, 4),
                    "t1": None if t1 >= OPEN_END else num(t1, 4), "point": text(req(p, "point", w)),
                    "sigma": [num(x, 3) for x in req(p, "sigma", w)], "bound": [num(x, 3) for x in req(p, "bound", w)],
                    "rms": None if rms is None else num(rms, 3), "run": int(req(p, "run", w)),
                    "n_test": int(req(p, "n_test", w)), "pass": req(p, "pass", w)})
    return out


def events_view(name: str, o: dict):
    """The boundaries of the physics chain, each a scene-graph event (rule events.graph): [{"phase", "event", "t",
    "window", "precision", "t_fit", "kick"}]; None without a physics record."""
    ph = fitted_physics(name, o)
    if ph is None:
        return None
    w = f"object {name!r} physics events_graph"

    def t_of(e):     # a time, or (FMT "interval") a release interval [lo, hi]
        t = e.get("t")
        if isinstance(t, list):
            if not FMT.get("interval") or len(t) != 2:
                raise ValueError(f"{AT['where']}: {w} event {e.get('event')!r} t = {t!r}")
            return [num(x, 4) for x in t]
        return num(t, 4) if t is not None else None

    return [{"phase": int(req(e, "phase", w)), "event": text(req(e, "event", w)), "t": t_of(e),
             "window": [num(x, 4) for x in e["window"]] if e.get("window") else None,
             "precision": text(e.get("precision")) or None, "t_fit": num(req(e, "t_fit", w), 4), "kick": bool(req(e, "kick", w))}
            | ({"t_fit_release": num(e["t_fit_release"], 4)} if FMT.get("interval") and "t_fit_release" in e else {})
            for e in req(ph, "events_graph", w)] or None


def align(a):
    """t_q alignment check of one answer (scene result quality "align_tq"): {"iou", "dc", "t", "frame"} | None."""
    if a is None:
        return None
    return {"iou": num(req(a, "iou", "align_tq"), 6), "dc": num(req(a, "dcenter_rel", "align_tq"), 6),
            "t": num(req(a, "t", "align_tq"), 4),
            "frame": req(a, "frame", "align_tq")}


def question_times(q: str, alignment: dict | None):
    """t_q of a question for the page: the times it names (pipeline.parse), and the time to jump to — the named time
    (midpoint of a [t1, t2] window); for a question that names no time, the frame its alignment check used."""
    times = [float(t) for t in parse_question(q).times]
    jump = (sum(times[:2]) / len(times[:2])) if times else (alignment or {}).get("t")
    return [num(t, 4) for t in times], num(jump, 4)


def fit_row(r: dict) -> dict:
    """One scene result (scene json "results" entry) as shown on the page."""
    qual = req(r, "quality", "scene result")
    out = {"q": req(r, "question", "scene result"), "obj": req(r, "object", "scene result"),
           "pred": num(req(r, "pred", "scene result")), "method": req(r, "method", "scene result"),
           "quality": {k: (num(v, 3) if isinstance(v, float) else v) for k, v in qual.items() if k != "conf"} or None}
    if FMT["decide"] == "conf":
        req(qual, "conf", "scene result quality")    # read as the format's field, not shown (decisions.no_conf)
    elif "conf" in qual:
        raise ValueError(f"{AT['where']}: scene result quality has a 'conf' field; this version decides by named checks")
    return out


def answer_checks(answer: dict, row: dict) -> dict:
    """The named pass/fail checks of a unit's answer (answer record "answer"), for versions that decide by checks:
    {"checks": [{"name", "pass", "reason"}], "failed": [names], "scene_value" (unresolved only, when kept)}.
    The CSV row's status / source / failed / pred must agree with the record."""
    if "conf" in answer or "conf" in row:
        raise ValueError(f"{AT['where']}: answer has a 'conf' field; this version decides by named checks")
    checks = req(answer, "checks", "answer")
    failed = list(req(answer, "failed", "answer"))
    status = req(answer, "status", "answer")
    out = []
    for name, c in checks.items():
        ok = req(c, "pass", f"check {name!r}")
        if not isinstance(ok, bool):
            raise ValueError(f"{AT['where']}: check {name!r} pass = {ok!r}")
        out.append({"name": name, "pass": ok, "reason": text(req(c, "reason", f"check {name!r}"))}
                   | ({"which": text(c["which"])} if c.get("which") else {}))
    if failed != [c["name"] for c in out if not c["pass"]]:
        raise ValueError(f"{AT['where']}: failed {failed} differs from the checks that did not pass")
    csv_failed = [x for x in text(req(row, "failed", "harness CSV row")).split(";") if x]
    if csv_failed != failed:
        raise ValueError(f"{AT['where']}: CSV failed {csv_failed} differs from the answer record's {failed}")
    if text(row["status"]) != status or text(row["source"]) != text(req(answer, "source", "answer")):
        raise ValueError(f"{AT['where']}: CSV status / source differs from the answer record")
    pred = num(row["pred"])
    if status == "resolved":
        if failed or pred is None or row["source"] not in CHECK_SOURCES:
            raise ValueError(f"{AT['where']}: resolved answer with failed {failed}, pred {pred}, source {row['source']!r}")
    elif status == "unresolved":
        if not failed or pred is not None or row["source"] != "unresolved":
            raise ValueError(f"{AT['where']}: unresolved answer with failed {failed}, pred {pred}, source {row['source']!r}")
    else:
        raise ValueError(f"{AT['where']}: answer status {status!r}")
    res = {"checks": out, "failed": failed}
    if status == "unresolved" and answer.get("scene_value") is not None:   # kept for inspection, not the answer
        res["scene_value"] = num(answer["scene_value"])
    return res


CHECK_FRAMES = {"window.impulse": ("window_frames", "piece_frames"), "window.release": ("window_frames", "flight_frames")}


def measurement_row(m: dict) -> dict:
    """One measurement of the asked quantity (scene result "measurements", docs/ANSWER_AGREEMENT.md section 5)."""
    what = f"measurement {m.get('name')!r}"
    adm = req(m, "admissible", what)
    ok = req(adm, "pass", f"{what} admissible")
    if not isinstance(ok, bool):
        raise ValueError(f"{AT['where']}: {what} admissible pass = {ok!r}")
    checks = req(m, "checks", what)
    failed = list(req(m, "failed", what))
    for k in failed:
        if k not in checks or checks[k]["pass"] is not False:
            raise ValueError(f"{AT['where']}: {what} lists {k!r} as failed, its checks do not")
    parts = req(m, "sigma_parts", what) or {}
    if FMT.get("launch") and "scale.contact_refit" in parts:
        raise ValueError(f"{AT['where']}: {what} sigma_parts has scale.contact_refit (rule plane.fusion replaced it)")
    terms = {k: num(v, 3) for k, v in parts.items() if k in ("track", "depth", "align", "focal", "gravity", "fit_se", "stated", "plane.fusion")}
    lst = lambda v: [num(x, 3) for x in v] if isinstance(v, list) else ([num(v, 3)] if v is not None else None)
    dof = parts.get("dof")
    inputs = req(m, "inputs", what) or {}
    frames = {}
    if FMT.get("unobserved"):
        # rule D31 time.one_grid: frame indices beside the seconds of the window and the piece
        if (not FMT.get("sceneprog") and bool(inputs.get("window")) != ("frames" in inputs)) or \
                ("piece" in inputs) != ("piece_frames" in inputs):
            raise ValueError(f"{AT['where']}: {what} inputs window / frames, piece / piece_frames {sorted(inputs)}")
        cp = inputs.get("chain_piece")
        if cp is not None and req(cp, "piece_frames", f"{what} chain_piece") != inputs.get("piece_frames"):
            raise ValueError(f"{AT['where']}: {what} chain_piece piece_frames {cp['piece_frames']} vs {inputs.get('piece_frames')}")
        if (inputs.get("piece") is None) != (inputs.get("piece_frames") is None):
            raise ValueError(f"{AT['where']}: {what} piece {inputs.get('piece')} with piece_frames {inputs.get('piece_frames')}")
        frames = {"window_frames": frames_pair(inputs, "frames", what) if inputs.get("window") and "frames" in inputs else None,
                  "piece_frames": frames_pair(inputs, "piece_frames", what) if inputs.get("piece_frames") is not None else None}
    if FMT.get("fused") and inputs.get("piece") is not None:
        cp = req(inputs, "chain_piece", what)
        if [num(x, 4) for x in inputs["piece"]] != [num(x, 4) for x in req(cp, "piece", f"{what} chain_piece")]:
            raise ValueError(f"{AT['where']}: {what} piece {inputs['piece']} differs from its chain_piece {cp['piece']}")
    return {"name": text(req(m, "name", what)), "method": text(req(m, "method", what)),
            "value": num(req(m, "value", what)), "unit": text(req(m, "unit", what)), "sigma": num(req(m, "sigma", what), 3),
            "terms": terms or None, "scale": lst(parts.get("scale")), "q": lst(parts.get("q")),
            "dof": [int(x) for x in dof] if isinstance(dof, list) else ([int(dof)] if dof is not None else None),
            "shared": list(req(m, "shared", what)), "admissible": ok, "why": text(req(adm, "reason", f"{what} admissible")),
            "failed": [{"name": k, "reason": text(req(checks[k], "reason", f"{what} check {k!r}"))}
                       | ({"which": text(checks[k]["which"])} if checks[k].get("which") else {}) for k in failed],
            "checks": [{"name": k, "pass": check_pass(c, f"{what} check {k!r}"),
                        "reason": text(req(c, "reason", f"{what} check {k!r}"))}
                       | ({"frames": {f: frames_pair(c, f, f"{what} check {k!r}") for f in CHECK_FRAMES[k]}}
                          if FMT.get("unobserved") and k in CHECK_FRAMES else {}) for k, c in checks.items()],
            "window": [num(x, 4) for x in inputs["window"]] if inputs.get("window") else None,
            "piece": [num(max(x, 0.0), 4) if x > -1e8 else None for x in inputs["piece"]]
            if inputs.get("piece") and not FMT.get("fused") else None,
            "n": inputs.get("n"), "state": text(inputs.get("state")) or None,
            "law": law_view(inputs["law"], what) if FMT.get("gravity") and inputs.get("law") else None,
            "profile": [profile_view(x, what) for x in inputs["at_bound.profile"]] if inputs.get("at_bound.profile") else None} \
        | (direct_view(m, what) if FMT.get("camera") else {}) | (sigma_view(inputs, what) if FMT.get("fusion") else {}) \
        | (onset_check_view(m, what) if FMT.get("impulse") and "kinematic_onset" in m else {}) \
        | (span_view(m, what) if FMT.get("span") else {}) \
        | ({"bounds": bounds_view(m, what)} if FMT.get("undetermined") else {}) | frames


def bounds_view(m: dict, what: str):
    """A direct measurement's plausible range (rule D25b plausible.prior_scaled): {lo, hi, prior_m, pass}; present
    exactly on a direct measurement with a value, None otherwise."""
    pb = m.get("plausible_bounds")
    valued = m.get("name") == "direct" and m.get("value") is not None
    if valued != (pb is not None):
        raise ValueError(f"{AT['where']}: {what} with value {m.get('value')!r} {'without' if valued else 'with'} plausible_bounds")
    if pb is None:
        return None
    w = f"{what} plausible_bounds"
    if set(pb) != {"lo", "hi", "prior_m", "pass"} or not isinstance(pb["pass"], bool):
        raise ValueError(f"{AT['where']}: {w} {pb!r}")
    return {"lo": num(req(pb, "lo", w)), "hi": num(req(pb, "hi", w)), "prior_m": num(pb["prior_m"]), "pass": pb["pass"]}


def span_view(m: dict, what: str) -> dict:
    """A measurement's window.impulse record (its check, when testable, is the same record) and a direct size's
    size.full_extent; each None when the measurement does not carry it."""
    wi = m.get("window_impulse")
    if wi is not None and "window.impulse" in req(m, "checks", what):
        c = m["checks"]["window.impulse"]
        if {k: c.get(k) for k in ("pass", "reason", "piece", "window")} != {k: wi.get(k) for k in ("pass", "reason", "piece", "window")}:
            raise ValueError(f"{AT['where']}: {what} check window.impulse differs from its window_impulse record")
    return {"window_impulse": impulse_window_view(wi, what) if wi is not None else None,
            "size.full_extent": full_extent_view(m["size.full_extent"], what) if "size.full_extent" in m else None}


def check_pass(c: dict, what: str) -> bool:
    ok = req(c, "pass", what)
    if not isinstance(ok, bool):
        raise ValueError(f"{AT['where']}: {what} pass = {ok!r}")
    return ok


def onset_check_view(m: dict, what: str) -> dict:
    """A law-driven onset's speed bound (rule kinematic.onset): {"kin_onset": {pass (null: untestable), t_on,
    sigma_t_on, v, integral, sigma, bound, reason}}."""
    k = req(m, "kinematic_onset", what)
    w = f"{what} kinematic_onset"
    ok = req(k, "pass", w)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass {ok!r}")
    return {"kin_onset": {"pass": ok, "t_on": num(k.get("t_on"), 5), "sigma_t_on": num(k.get("sigma_t_on"), 3),
                          "v": num(k.get("v"), 4), "integral": num(k.get("integral"), 4), "sigma": num(k.get("sigma"), 3),
                          "bound": num(k.get("bound"), 4), "reason": text(req(k, "reason", w))}}


def plain(x):
    """A JSON value as plain JSON (numbers rounded), for records shown as written."""
    if isinstance(x, (list, tuple)):
        return [plain(a) for a in x]
    if isinstance(x, dict):
        return {text(k): plain(v) for k, v in x.items()}
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return num(x, 4)
    return text(x)


def sigma_view(inputs: dict, what: str) -> dict:
    """How the measurement's sigma was found (rule fit.sigma_chain): the text "sigma_how" with the chain piece holding
    t {piece, events [[t, kind]], window_short, events_near_t}; without a chain piece the text "sigma" when the
    measurement records one; the whole-fit events whose finite difference was one-sided or unstable."""
    cp = inputs.get("chain_piece")
    if cp is not None:
        w = f"{what} chain_piece"
        how = text(req(inputs, "sigma_how", what))
        chain = {"piece": [num(x, 4) for x in req(cp, "piece", w)],
                 "events": [[num(t, 4), text(k)] for t, k in req(cp, "events", w)],
                 "window_short": plain(req(cp, "window_short", w)), "events_near_t": plain(req(cp, "events_near_t", w))}
        if FMT.get("fused"):
            ch = req(cp, "chain", w)
            if not isinstance(ch, bool) or (not ch and chain["events"]):
                raise ValueError(f"{AT['where']}: {w} chain = {ch!r} with events {chain['events']}")
            chain["chain"] = ch
    else:
        if inputs.get("sigma_how") is not None:
            raise ValueError(f"{AT['where']}: {what} has sigma_how without chain_piece")
        sg = inputs.get("sigma")
        how = text(sg) if isinstance(sg, str) else None
        chain = None
    se = inputs.get("sigma_events") or {}
    flags = {k: plain(se.get(k) or []) for k in ("one_sided", "event_unstable")}
    return {"sigma_how": how, "chain": chain, "sigma_flags": {k: v for k, v in flags.items() if v} or None}


def direct_view(m: dict, what: str) -> dict:
    """A direct measurement's state span (rule window.state), support-plane depth and its silhouette cross-check
    (rule C12); each None when the measurement does not carry it."""
    w = m.get("window")
    win = None
    if w is not None:
        cb = req(w, "clipped_by", f"{what} window")
        win = {"t0": num(req(w, "t0", f"{what} window"), 4), "t1": num(req(w, "t1", f"{what} window"), 4),
               "state": text(req(w, "state", f"{what} window")) or None,
               "start": text(req(cb, "start", f"{what} window clipped_by")) or None,
               "end": text(req(cb, "end", f"{what} window clipped_by")) or None,
               "used": [num(x, 4) for x in w["window"]] if req(w, "window", f"{what} window") else None}
        if FMT.get("unobserved"):
            win |= {"frames": frames_pair(w, "frames", f"{what} window"),
                    "used_frames": frames_pair(w, "window_frames", f"{what} window") if w.get("window") else None}
        if FMT.get("roll"):
            inputs = req(m, "inputs", what) or {}
            win |= {"law": text(req(w, "law", f"{what} window")) or None, "check": text(req(w, "check", f"{what} window"))}
            if win["law"] is None and win["used"] is None and win["t0"] is None and m.get("value") is None:
                # a driven window over no tracked frame (t outside the track): nothing measured, no window.smooth
                if win["check"] != "window.smooth" or req(m, "checks", what) or inputs:
                    raise ValueError(f"{AT['where']}: {what} empty driven window with checks / inputs")
            elif win["law"] is None:   # rule window.driven: the state's own jitter is the noise of window.smooth
                if win["check"] != "window.smooth" or "window.smooth" not in req(m, "checks", what):
                    raise ValueError(f"{AT['where']}: {what} driven window without the check window.smooth")
                win["driven"] = {"state": text(req(inputs, "driven", f"{what} inputs")),
                                 "noise": [num(x, 3) for x in req(inputs, "noise_px", f"{what} inputs")],
                                 "how": text(req(inputs, "noise_how", f"{what} inputs"))}
    sd = m.get("support_depth")
    sdv = None
    if sd is not None:
        if req(sd, "applies", f"{what} support_depth"):
            z = float(req(sd, "z_plane", f"{what} support_depth"))
            sdv = {"applies": True, "z": num(z, 3), "n": int(req(sd, "n", f"{what} support_depth")),
                   "sigma_plane": num(req(sd, "sigma_plane", f"{what} support_depth"), 3),
                   "sigma_row": num(z * float(req(sd, "rel_noise", f"{what} support_depth")), 3)}
        else:
            sdv = {"applies": False, "reason": text(req(sd, "reason", f"{what} support_depth"))}
    da = (m.get("cross_checks") or {}).get("direct.depth_agree")
    if sdv and sdv["applies"] and da is None:
        raise ValueError(f"{AT['where']}: {what} has a support-plane depth without direct.depth_agree")
    if da is None:
        dav = None
    elif req(da, "pass", f"{what} direct.depth_agree") is None:
        # untestable: the direct path's own sample series tests nothing (no scene-level depth_rows verdict to copy)
        if "see" in da:
            raise ValueError(f"{AT['where']}: {what} direct.depth_agree copies {da['see']!r} with pass null")
        dav = {"pass": None, "reason": text(req(da, "reason", f"{what} direct.depth_agree"))}
    else:
        w = f"{what} direct.depth_agree"
        dav = {"pass": da["pass"], "z_mask": num(req(da, "z_mask", w), 3), "z_plane": num(req(da, "z_plane", w), 3),
               "diff": num(abs(float(da["z_mask"]) - float(da["z_plane"])), 3), "bound": num(req(da, "bound", w), 3),
               "reason": text(req(da, "reason", w))}
        if "see" in da:    # rule fit.object.record (c): the scene-level depth_rows verdict, copied
            if da["see"] != "depth_rows":
                raise ValueError(f"{AT['where']}: {w} see {da['see']!r}")
            dav |= {"see": "depth_rows", "n_rows": int(req(da, "n_rows", w)), "n_matched": int(req(da, "n_matched", w)),
                    "dropped": bool(req(da, "dropped", w))}
    ce = (m.get("inputs") or {}).get("centred") if FMT.get("static") else None
    cev = None if ce is None else {
        "from": [num(x, 4) for x in req(ce, "from", f"{what} centred")],
        "extremum": text(req(ce, "extremum", f"{what} centred")), "stop": text(req(ce, "stop", f"{what} centred"))} \
        | ({"from_frames": frames_pair(ce, "from_frames", f"{what} centred")} if FMT.get("unobserved") and ce is not None else {})
    return {"wstate": win, "support_depth": sdv, "depth_agree": dav, "centred": cev}


def profile_view(x: dict, what: str) -> dict:
    """One parameter of rule at_bound.profile: the answer at the physical bound and at the bound + 1 sigma inward."""
    w = f"{what} at_bound.profile"
    return {"param": text(req(x, "param", w)), "side": text(req(x, "side", w)), "bound": num(req(x, "bound", w), 3),
            "sigma": num(req(x, "sigma", w), 3), "at": num(req(x, "profiled_at", w), 3),
            "beyond": bool(req(x, "beyond_range", w)), "values": [num(v) for v in req(x, "values", w)],
            "delta": num(req(x, "delta", w), 3), "sigma_answer": num(req(x, "sigma_answer", w), 3),
            "pass": bool(req(x, "pass", w)), "why": text(req(x, "why", w))}


def law_view(law: dict, what: str) -> dict:
    """The ballistic law a measurement was read under (rule B2): {"g", "source"}."""
    return {"g": num(req(law, "g", f"{what} law")), "source": text(req(law, "source", f"{what} law"))}


def agreement_view(result: dict, answer: dict, status: str, state: dict | None = None) -> dict:
    """The measurements of the asked quantity (scene result), the chosen one, and the checks "agree" / "constraints"
    of the answer with each constraint residual (answer "measurement"."constraint_residuals")."""
    ms = [measurement_row(m) for m in req(result, "measurements", "scene result")]
    names = [m["name"] for m in ms]
    if len(set(names)) != len(names):
        raise ValueError(f"{AT['where']}: measurement names repeat: {names}")
    chosen = req(result, "chosen", "scene result")
    req(result, "selection", "scene result")
    if chosen is not None:
        m = next((x for x in ms if x["name"] == chosen), None)
        if m is None or not m["admissible"]:
            raise ValueError(f"{AT['where']}: chosen {chosen!r} is not an admissible measurement")
        kept = num(answer.get("pred") if status == "resolved" else answer.get("scene_value"))
        if kept != m["value"]:
            raise ValueError(f"{AT['where']}: the answer's value {kept} is not the chosen measurement's {m['value']}")
    checks = req(answer, "checks", "answer")
    ag, cs = req(checks, "agree", "answer checks"), req(checks, "constraints", "answer checks")
    agree = {"pass": req(ag, "pass", "agree"), "chi2": num(req(ag, "chi2", "agree"), 4), "dof": req(ag, "dof", "agree"),
             "bound": num(req(ag, "bound", "agree"), 4),
             "values": [[k, num(v[0]), num(v[1], 3)] for k, v in req(ag, "values", "agree").items()]}
    res = list(req(req(answer, "measurement", "answer"), "constraint_residuals", "answer measurement"))
    if FMT.get("unobserved"):
        # rule D30 contract.contained: the measurement's residuals carry the scene's contained rows too (same ids)
        sc = req(state, "constraints", "scene")
        rid = [text(c.get("id")) for c in res]
        if len(set(rid)) != len(rid) or {text(c.get("id")) for c in sc} != set(rid):
            raise ValueError(f"{AT['where']}: scene constraints {[c.get('id') for c in sc]} differ from the answer's residuals {sorted(rid)}")
    elif FMT.get("contained"):
        # rule A13 contained.same_depth: the container-depth rows are judged in "constraints" but are no stated
        # constraint, so they are only in the scene's own "constraints" list (kind "contained"), never in the residuals
        sc = req(state, "constraints", "scene")
        rid = {text(c.get("id")) for c in res}
        if any(text(c.get("kind")) == "contained" for c in res) or \
                {text(c.get("id")) for c in sc if c.get("kind") != "contained"} != rid:
            raise ValueError(f"{AT['where']}: scene constraints {[c.get('id') for c in sc]} differ from the answer's residuals {sorted(rid)}")
        res += [c for c in sc if c.get("kind") == "contained"]
    rows = []
    for c in res:
        w = f"constraint {c.get('id')!r}"
        obs, pred = req(c, "observed", w), req(c, "predicted", w)
        rows.append({"id": text(req(c, "id", w)), "kind": text(req(c, "kind", w)), "units": text(req(c, "units", w)),
                     "observed": num(obs), "predicted": num(pred),
                     "diff": num(abs(float(pred) - float(obs)), 3) if pred is not None else None,
                     "sigma_stated": num(req(c, "sigma_stated", w), 3), "sigma_pred": num(req(c, "sigma_pred", w), 3),
                     "bound": num(req(c, "bound", w), 3), "pass": req(c, "pass", w), "testable": req(c, "testable", w)}
                    | ({"stated": stated_view(req(c, "stated", w), w)} if FMT.get("kept") else {})
                    | ({"untestable": untestable_view(c, w)}
                       if FMT.get("fused") and c.get("pass") is None and c.get("inadmissible") is None else {})
                    | ({"inadmissible": inadmissible_view(c, w)} if FMT.get("contact") and c.get("inadmissible") is not None else {})
                    | ({"contained": contained_row_view(c, w)} if FMT.get("contained") and c.get("kind") == "contained" else {}))
    ids = {r["id"] for r in rows}
    failed_ids = list(req(cs, "failed_ids", "constraints"))
    if not set(failed_ids) <= ids:
        raise ValueError(f"{AT['where']}: constraints failed_ids {failed_ids} not among the residuals {sorted(ids)}")
    if FMT.get("fused"):
        un_ids = list(req(cs, "untestable_ids", "constraints"))
        if sorted(un_ids) != sorted(r["id"] for r in rows if r["pass"] is None and "inadmissible" not in r) or \
                len(req(cs, "untestable", "constraints")) != len(un_ids):
            raise ValueError(f"{AT['where']}: constraints untestable_ids {un_ids} differ from the residuals with pass null")
    if FMT.get("contact"):
        in_ids = list(req(cs, "inadmissible_ids", "constraints"))
        if sorted(in_ids) != sorted(r["id"] for r in rows if "inadmissible" in r) or \
                len(req(cs, "inadmissible", "constraints")) != len(in_ids):
            raise ValueError(f"{AT['where']}: constraints inadmissible_ids {in_ids} differ from the residuals marked inadmissible")
    wi = result.get("window_impulse") if FMT.get("span") else None
    return {"measurements": ms, "chosen": chosen, "agree": agree, "constraints": rows,
            "window_impulse": impulse_window_view(wi, "scene result", result=True) if wi is not None else None} \
        | ({"answer_span": answer_span_view(result.get("answer.span"))} if FMT.get("rendered") else {})


def answer_span_view(x: dict | None):
    """Rule B31 piece.untested_not_read: the span an averaging answer reads -- the fitted frames clipped to the first /
    last one outside every unobserved phase and untested piece (clipped_from: the span before; null: not clipped)."""
    if x is None:
        return None
    w = "scene result answer.span"
    if set(x) != {"rule", "pass", "t0", "t1", "frames", "clipped_from", "why"} or x["rule"] != "piece.untested_not_read" \
            or not isinstance(x["pass"], bool) or (x["pass"] and (x["t0"] is None or len(x["frames"] or []) != 2)):
        raise ValueError(f"{AT['where']}: {w} {x!r}")
    cf = x["clipped_from"]
    return {"pass": x["pass"], "t0": num(x["t0"], 4), "t1": num(x["t1"], 4),
            "frames": [int(f) for f in x["frames"]] if x["frames"] else None,
            "clipped_from": [num(t, 4) for t in cf] if cf is not None else None, "why": text(x["why"])}


def contained_row_view(c: dict, what: str) -> dict:
    """A constraint of rule A13 contained.same_depth: the object's predicted depth against its container's observed
    depth at one time of the edge's span; {object, container, t, sigma_container {sigma, sigma_zb, sigma_zb_how, h,
    extent_a, extent_b}}."""
    sc, w = req(c, "sigma_container", what), f"{what} sigma_container"
    ev = req(c, "evaluation", what)
    if text(req(ev, "kind", f"{what} evaluation")) != "contained" or text(req(ev, "container", f"{what} evaluation")) != text(req(c, "container", what)):
        raise ValueError(f"{AT['where']}: {what} evaluation {ev!r}")
    if num(req(sc, "sigma", w)) != num(req(c, "sigma_stated", what)):
        raise ValueError(f"{AT['where']}: {what} sigma_container differs from its sigma_stated")
    return {"object": text(req(c, "object", what)), "container": text(c["container"]), "t": num(req(c, "time", what), 4),
            "sigma": num(sc["sigma"], 3), "sigma_zb": num(req(sc, "sigma_zb", w), 3), "sigma_zb_how": text(req(sc, "sigma_zb_how", w)),
            "h": num(req(sc, "h", w), 3), "extent_a": num(req(sc, "extent_a", w), 3), "extent_b": num(req(sc, "extent_b", w), 3)}


def inadmissible_view(c: dict, what: str) -> dict:
    """A constraint of a failed object fit (rule fit.object): no verdict, not scale evidence; {rule, reason}."""
    if req(c, "pass", what) is not None or req(c, "testable", what) is not False:
        raise ValueError(f"{AT['where']}: {what} inadmissible with pass {c['pass']!r} / testable {c['testable']!r}")
    u, w = c["inadmissible"], f"{what} inadmissible"
    rule = text(req(u, "rule", w))
    if rule != "fit.object":
        raise ValueError(f"{AT['where']}: {w} rule {rule!r}")
    return {"rule": rule, "reason": text(req(u, "reason", w))}


def untestable_view(c: dict, what: str) -> dict:
    """A constraint that tests nothing (rule G13 constraint.untestable): its sigma_pred is at least its stated value;
    {rule, sigma_pred, source (sigma_pred's delta-method text), reason}."""
    if req(c, "testable", what) is not False:
        raise ValueError(f"{AT['where']}: {what} pass null but testable {c['testable']!r}")
    u = req(c, "untestable", what)
    w = f"{what} untestable"
    rule = text(req(u, "rule", w))
    if rule != "constraint.untestable":
        raise ValueError(f"{AT['where']}: {w} rule {rule!r}")
    return {"rule": rule, "sigma_pred": num(req(u, "sigma_pred", w), 4), "source": text(req(u, "source", w)),
            "reason": text(req(u, "reason", w))}


def stated_view(st: dict, what: str) -> dict:
    """A constraint's stated number as written (rule G10): its text (null: no text of the unit states it), digits,
    unit, sigma (SI, half a unit of the last written digit) and source."""
    return {"text": text(req(st, "text", f"{what} stated")) or None, "digits": text(req(st, "digits", f"{what} stated")),
            "unit": text(req(st, "unit", f"{what} stated")), "sigma": num(req(st, "sigma", f"{what} stated"), 3),
            "source": text(req(st, "source", f"{what} stated"))}


def dropped_view(c: dict, what: str) -> dict:
    """A repair candidate the loop dropped before the planner: its call and the rule that rejected it with the test's
    reason ("rule repair.applicable: rejected (rule <name>): <reason>")."""
    why = text(req(c, "why", f"{what} dropped candidate"))
    if FMT.get("contained") and "dropped" in c:
        # rule D27 repair.no_repeat: a repair that failed the same way twice is not offered again
        nr = req(c, NO_REPEAT, f"{what} dropped candidate")
        if c["dropped"] != NO_REPEAT:
            raise ValueError(f"{AT['where']}: {what} dropped candidate dropped {c['dropped']!r}")
        call = req(c, "call", f"{what} dropped candidate")
        return {"name": text(req(call, "name", what)), "params": {k: v for k, v in (call.get("params") or {}).items() if v is not None},
                "rule": NO_REPEAT, "why": why, "no_repeat": no_repeat_view(nr, what)}
    m = re.match(r"^rule repair\.applicable: rejected \(rule ([\w.]+)\):\s*(.*)$", why, re.S)
    call = req(c, "call", f"{what} dropped candidate")
    return {"name": text(req(call, "name", what)), "params": {k: v for k, v in (call.get("params") or {}).items() if v is not None},
            "rule": m.group(1) if m else None, "why": m.group(2) if m else why}


PLANNING_FAILED_KEYS = {"iter", "why", "accepted", "execution_status", "tool_calls"}
NO_REPEAT = "repair.no_repeat"   # rule D27 (pipeline.scene_agent.REPAIR_NO_REPEAT)
UNOBSERVED_KEPT = "graph_edit.unobserved_kept"   # rule A16 (pipeline.scenegraph.UNOBSERVED_KEPT)
REJECTED_CALL_REASONS = ("reference.not_repairable", NO_REPEAT, UNOBSERVED_KEPT)


def unobserved_kept_view(x: dict, what: str) -> dict:
    """Rule A16 graph_edit.unobserved_kept: an agent state edit over an unobserved phase, clipped to its observed parts
    (kept), or rejected when none of at least one frame step is left (kept null): {object, requested, unobserved, kept,
    dropped, why}."""
    if set(x) != {"rule", "object", "requested", "unobserved", "kept", "dropped", "why"} or x["rule"] != UNOBSERVED_KEPT:
        raise ValueError(f"{AT['where']}: {what} {UNOBSERVED_KEPT} record {x!r}")
    sp = lambda v: [num(t, 4) if t is not None else None for t in v]   # noqa: E731
    return {"object": text(x["object"]), "requested": sp(x["requested"]), "unobserved": [sp(v) for v in x["unobserved"]],
            "kept": [sp(v) for v in x["kept"]] if x["kept"] is not None else None, "dropped": [sp(v) for v in x["dropped"]],
            "why": text(x["why"])}


CONTRACT_ROW_KEYS = {"id", "kind", "object", "container", "time", "observed", "predicted", "residual", "units", "sigma",
                     "sigma_abs", "standardized", "testable", "untestable"}
CONTRACT_KINDS = ("depth", "prior", "contained")   # pipeline.evidence.CONSTRAINT_KINDS (rule D30 contract.contained)


def contract_view(m: dict | None, what: str):
    """Rule D30: the agent's frozen constraint contract replayed on a round's scene -- one row per replayed key
    {id, kind, object, container, time, observed {kind stated | fitted, object, value}, predicted, residual, units,
    sigma, standardized, untestable reason} and the untestable ids. None without metrics."""
    if m is None:
        return None
    cc, w = req(m, "constraint_contract", what), f"{what} constraint_contract"
    rows = []
    for r in req(cc, "residuals", w):
        ob = r.get("observed") or {}
        if set(r) != CONTRACT_ROW_KEYS or r["kind"] not in CONTRACT_KINDS or set(ob) != {"kind", "object", "value"} \
                or ob["kind"] not in ("stated", "fitted") or (r["kind"] == "contained") != (ob["kind"] == "fitted") \
                or r["testable"] != (r["untestable"] is None):
            raise ValueError(f"{AT['where']}: {w} residual row {r!r}")
        un = r["untestable"]
        if un is not None and set(un) != {"rule", "reason"}:
            raise ValueError(f"{AT['where']}: {w} residual untestable {un!r}")
        rows.append({"id": text(r["id"]), "kind": r["kind"], "object": text(r["object"]), "container": text(r["container"]) or None,
                     "time": num(r["time"], 4), "observed": {"kind": ob["kind"], "object": text(ob["object"]) or None, "value": num(ob["value"], 4)},
                     "predicted": num(r["predicted"], 4), "residual": num(r["residual"], 3), "units": text(r["units"]),
                     "sigma": num(r["sigma"], 3), "standardized": num(r["standardized"], 3),
                     "untestable": {"rule": text(un["rule"]), "reason": text(un["reason"])} if un else None})
    ut = [text(x) for x in req(cc, "untestable_ids", w)]
    if sorted(ut) != sorted(r["id"] for r in rows if r["untestable"]):
        raise ValueError(f"{AT['where']}: {w} untestable_ids {ut} differ from its untestable rows")
    return {"rows": rows, "untestable_ids": ut}


def support_plane_issue_view(x: dict, w: str) -> dict:
    """Rule D32 visibility.support_plane: a scene-graph support surface with no scene object, dropped from the
    visibility issues (it is the scene's support plane): {object, t, why}."""
    if set(x) != {"object", "kind", "t", "dropped", "why"} or x["dropped"] != "visibility.support_plane":
        raise ValueError(f"{AT['where']}: {w} visibility.support_plane {x!r}")
    return {"object": text(x["object"]), "kind": text(x["kind"]), "t": num(x["t"], 4), "why": text(x["why"])}


def no_repeat_view(x: dict, what: str) -> dict:
    """A repair key that failed the same way twice (rule D27): {tool, target, rounds, reason}."""
    if set(x) != {"tool", "target", "rounds", "reason"} or len(x["rounds"]) != 2:
        raise ValueError(f"{AT['where']}: {what} {NO_REPEAT} record {x!r}")
    return {"tool": text(x["tool"]), "target": text(x["target"]) or None, "rounds": [int(r) for r in x["rounds"]],
            "reason": text(x["reason"])}


def rejected_call_view(c: dict, w: str) -> dict:
    """A call the agent loop rejected before running it: {name, params, reason, why}."""
    reason = text(req(c, "reason", f"{w} rejected call"))
    if reason not in REJECTED_CALL_REASONS:
        raise ValueError(f"{AT['where']}: {w} rejected call reason {reason!r}")
    call = req(c, "call", f"{w} rejected call")
    if reason == NO_REPEAT:
        no_repeat_view(req(c, NO_REPEAT, f"{w} rejected call"), f"{w} rejected call")
    uk = None
    if reason == UNOBSERVED_KEPT:
        uk = unobserved_kept_view(req(c, UNOBSERVED_KEPT, f"{w} rejected call"), f"{w} rejected call")
        if uk["kept"] is not None:
            raise ValueError(f"{AT['where']}: {w} rejected call {UNOBSERVED_KEPT} with kept parts {uk['kept']}")
    return {"name": text(req(call, "name", f"{w} rejected call")),
            "params": {k: v for k, v in (call.get("params") or {}).items() if v is not None},
            "reason": reason, "why": text(req(c, "why", f"{w} rejected call"))} | ({"unobserved_kept": uk} if uk else {})


def partial_view(rec: dict):
    """Why an answer record is "partial" (FMT sizeonly): the agent loop stopped on a round whose plan never came back
    (stop reason planning_failed); {"round", "why"} from that round. None for a complete record; any other shape raises."""
    es = text(req(rec, "execution_status", "answer record"))
    st = req(rec, "stop", "answer record")
    if es == "complete":
        if st["reason"] == "planning_failed":
            raise ValueError(f"{AT['where']}: complete record with stop planning_failed")
        return None
    if es != "partial":
        raise ValueError(f"{AT['where']}: answer record execution_status {es!r}")
    its = req(rec, "iterations", "answer record")
    last = its[-1] if its else {}
    if FMT.get("sceneprog") and st["reason"] != "planning_failed":
        # v6: a unit is also partial when the loop stopped with a named reason (loop.exhausted) on an unresolved scene
        return {"round": int(req(st, "round", "answer record stop")), "why": f"{text(st['reason'])}: {text(req(st, 'why', 'answer record stop'))}"}
    if st["reason"] != "planning_failed" or set(last) != PLANNING_FAILED_KEYS or int(last["iter"]) != int(st["round"]):
        raise ValueError(f"{AT['where']}: partial record with stop {st!r}, last round keys {sorted(last)}")
    return {"round": int(st["round"]), "why": f"planning_failed: {text(last['why'])}"}


def rounds_view(rec: dict) -> dict | None:
    """The agent rounds of an answer record ("iterations"): per round its tool calls with their results (a rejected
    edit's result starts "rejected: "), the plan's rejected issues, whether it was accepted and why; "kept" = the
    round whose state the answer used (kept_state.iter, 0 = the initial fit). None when the agent made no round."""
    its = req(rec, "iterations", "answer record")
    kept = int(req(req(rec, "kept_state", "answer record"), "iter", "kept_state"))
    if kept and kept not in [it.get("iter") for it in its]:
        raise ValueError(f"{AT['where']}: kept_state.iter {kept} is not an agent round")
    rows = []
    for it in its:
        w = f"agent round {it.get('iter')!r}"
        if FMT.get("sizeonly") and set(it) == PLANNING_FAILED_KEYS:
            # a round whose plan never came back (the model call failed): no call, no accept test; the record is partial
            if it["execution_status"] != "failed" or it["accepted"] is not False or it["tool_calls"] or not text(it["why"]):
                raise ValueError(f"{AT['where']}: {w} planning round {it!r}")
            rows.append({"iter": int(it["iter"]), "layer": None, "calls": [], "rejected": [], "accepted": False,
                         "why": text(it["why"]), "planning_failed": True, "newly_failed": [], "swap": None,
                         "rebuilt": None, "candidates": [], "dropped": [], "rejected_calls": []})
            continue
        calls = [{"name": text(req(r, "name", w)), "params": {k: v for k, v in (r.get("params") or {}).items() if v is not None},
                  "result": text(req(r, "result", w))}
                 | ({"test": test_view(r["test"], w)} if FMT.get("pendulum") and r.get("test") else {})
                 | ({"unobserved_kept": unobserved_kept_view(r[UNOBSERVED_KEPT], w)}
                    if FMT.get("unobserved") and UNOBSERVED_KEPT in r else {})
                 for r in req(it, "results", w)]
        rej = [text(req(x, "why", f"{w} rejected")) for x in (it.get("rejected") or [])]
        acc = req(it, "accepted", w)
        row = {"iter": int(req(it, "iter", w)), "layer": text(it.get("layer")) or None, "calls": calls,
               "rejected": rej, "accepted": bool(acc), "why": text(req(it, "why", w))}
        if FMT.get("horizon"):
            row["newly_failed"] = [[text(k), text(v)] for k, v in (it.get("newly_failed") or {}).items()]
        if FMT.get("camera"):
            # a round stopped before its accept test (loop.no_repair) has no fixed / broken record
            fx, br = it.get("checks_fixed"), it.get("checks_broken")
            if (fx is None) != (br is None):
                raise ValueError(f"{AT['where']}: {w} has only one of checks_fixed / checks_broken")
            row["swap"] = None if fx is None else [[text(x) for x in fx], [text(x) for x in br]]
            rb = it.get("rebuilt")
            row["rebuilt"] = None if rb is None else {
                "steps": [text(x) for x in req(rb, "steps", f"{w} rebuilt")], "failed": text(req(rb, "failed", f"{w} rebuilt")) or None,
                "self_test": {k: v for k, v in (req(rb, "self_test", f"{w} rebuilt") or {}).items()
                              if isinstance(v, (int, float, str, bool))} or None}
        if FMT.get("roll"):
            row["candidates"] = [{"for": text(req(c, "for", f"{w} repair candidate")), "name": text(req(req(c, "call", w), "name", w)),
                                  "params": {k: v for k, v in (c["call"].get("params") or {}).items() if v is not None},
                                  "class": text(c.get("class")) or None}
                                 | ({"why": text(c.get("why")) or None} if FMT.get("rendered") else {})
                                 for c in (it.get("repair_candidates") or [])]
            row["dropped"] = [dropped_view(c, w) for c in (it.get("repair_candidates_dropped") or [])]
        if FMT.get("sizeonly"):
            # rule reference.size_only: a call that names a reference is rejected (reference.not_repairable)
            row["rejected_calls"] = [rejected_call_view(c, w) for c in req(it, "rejected_calls", w)]
        if FMT.get("contained"):
            # rule D27: the keys that joined the no-repeat set this round; a round stopped before planning
            row["no_repeat"] = [no_repeat_view(x, w) for x in it.get(NO_REPEAT) or []]
            ds = it.get("decision_status")
            row["not_planned"] = ds == "not_planned"
            if row["not_planned"] and (calls or it.get("execution_status") != "unchanged" or acc):
                raise ValueError(f"{AT['where']}: {w} not_planned with calls / execution_status {it.get('execution_status')!r}")
        if FMT.get("rendered"):
            # rule D32: a support surface the scene fit takes as the support plane is no omitted object
            row["support_plane"] = [support_plane_issue_view(x, w) for x in it.get("visibility.support_plane") or []]
        if FMT.get("unobserved"):
            row["contract"] = {"before": contract_view(it.get("metrics_before"), f"{w} metrics_before"),
                               "after": contract_view(it.get("metrics_after"), f"{w} metrics_after")}
        rows.append(row)
    out = {"kept": kept, "rounds": rows}
    if FMT.get("horizon"):
        st = req(rec, "stop", "answer record")
        tried = [{"iter": x.get("iter"), "name": text(req(x, "name", "stop.tried")),
                  "params": {k: v for k, v in (x.get("params") or {}).items() if v is not None},
                  "class": text(req(x, "class", "stop.tried")), "outcome": text(x.get("outcome"))}
                 for x in (st.get("tried") or [])]
        out["stop"] = {"reason": text(req(st, "reason", "stop")), "round": st.get("round"),
                       "why": text(st.get("why")) or None, "tried": tried}
        if FMT.get("contained"):
            out["stop"]["no_repeat"] = [no_repeat_view(x, "stop") for x in st.get(NO_REPEAT) or []]
            nps = [r["iter"] for r in rows if r.get("not_planned")]
            if nps and (st["reason"] != "loop.no_repair" or nps != [st.get("round")] or not out["stop"]["no_repeat"]):
                raise ValueError(f"{AT['where']}: not_planned rounds {nps} with stop {st['reason']!r} round {st.get('round')!r}")
    return out if rows else None


def dim_bounds(name: str, o: dict):
    """{dimension: [lo, hi | None]} of the object's "bound" dimensions (scene_fit rule D2a: an unobserved dimension is
    an interval, never a value); None for versions without the field. Its keys must be exactly the bound dimensions."""
    if not FMT.get("identity"):
        return None
    b = req(o, "dim_bounds", f"scene object {name!r}")
    bound = {k for k, v in req(o, "dim_source", f"scene object {name!r}").items() if v == "bound"}
    if set(b) != bound:
        raise ValueError(f"{AT['where']}: object {name!r} dim_bounds {sorted(b)} differ from its bound dims {sorted(bound)}")
    return {k: [num(lo), None if hi is None else num(hi)] for k, (lo, hi) in b.items()}


def scale_view(state: dict):
    """The scale of a prior-scaled scene: {"s", "source", "object", "prior", "how", "extent"} ("extent" for a
    prior-extent scale: the frames and depth observations the nominal extent was measured on); None without one."""
    if not FMT.get("identity"):
        return None
    sc = req(state, "scale", "scene")
    if sc is None:
        return None
    src = text(req(sc, "source", "scale"))
    ext = sc.get("extent")
    if src == "prior_object_extent" and not ext:
        raise ValueError(f"{AT['where']}: prior-extent scale without its 'extent'")
    out = {"s": num(req(sc, "s", "scale")), "source": src, "object": text(sc.get("object")) or None,
           "prior": text(req(sc, "prior", "scale")), "how": text(req(sc, "how", "scale")), "extent": text(ext) or None}
    if FMT.get("gravity"):
        out["checks"] = named_checks(req(sc, "checks", "scale"), list(req(sc, "failed", "scale")), "scale")
    if FMT.get("classref"):
        if req(sc, "class_reference", "scale") != req(state, "scale.class_reference", "scene"):
            raise ValueError(f"{AT['where']}: scale.class_reference differs from the scene's scale.class_reference")
        out["full_extent"] = scale_full_extent_view(req(sc, "full_extent", "scale"))
    if FMT.get("contained"):
        # rule B23: a scale read from a motion prior carries its record, every other scale has none
        if ("prior_motion" in sc) != (src == "prior_object_motion"):
            raise ValueError(f"{AT['where']}: scale source {src!r} {'with' if 'prior_motion' in sc else 'without'} scale.prior_motion")
        out["prior_motion"] = prior_motion_view(sc.get("prior_motion"), req(sc, "checks", "scale"))
    return out


PRIOR_MOTION_KEYS = {"name", "pass", "pieces", "unobserved", "piece", "t_lo", "t_hi", "law", "rms_px", "noise_px",
                     "bound_px", "read", "speed_px_s", "depth", "value", "displacement_value", "sigma_rel", "s",
                     "sigma_s", "why"}
PRIOR_MOTION_PIECE_KEYS = {"piece", "law", "t_lo", "t_hi", "n", "used", "read", "value", "sigma", "weight", "rms_px",
                           "bound_px", "rate", "depth", "displacement_value", "why"}
PRIOR_MOTION_READS = ("fit", "displacement", "mean")


def prior_motion_view(pm: dict | None, checks: dict):
    """Rule B23 scale.prior_motion: the motion prior read over the object's observed motion pieces; each piece's law,
    read (fit / displacement), value, weight (observed seconds), whether it entered the mean; the chosen read and the
    scale. None for a scale that is not read from a motion prior; the check scale.checks.prior_motion carries its pass
    and why."""
    if pm is None:
        if "prior_motion" in checks:
            raise ValueError(f"{AT['where']}: scale check prior_motion without its scale.prior_motion record")
        return None
    w = "scale.prior_motion"
    if set(pm) != PRIOR_MOTION_KEYS or pm["name"] != w:
        raise ValueError(f"{AT['where']}: {w} keys {sorted(pm)} name {pm.get('name')!r}")
    c = req(checks, "prior_motion", "scale checks")
    if c["pass"] is not pm["pass"] or text(c["reason"]) != text(pm["why"]):
        raise ValueError(f"{AT['where']}: scale check prior_motion differs from {w}")
    read = pm["read"]
    if read not in PRIOR_MOTION_READS + (None,) or (read is None) == bool(pm["pass"]):
        raise ValueError(f"{AT['where']}: {w} read {read!r} with pass {pm['pass']!r}")
    pieces = []
    for p in pm["pieces"]:
        wp = f"{w} piece {p.get('piece')!r}"
        if set(p) != PRIOR_MOTION_PIECE_KEYS:
            raise ValueError(f"{AT['where']}: {wp} keys {sorted(p)}")
        if p["read"] not in ("fit", "displacement", None) or (p["used"] and p["read"] is None):
            raise ValueError(f"{AT['where']}: {wp} read {p['read']!r} used {p['used']!r}")
        dz = p["depth"]
        pieces.append({"piece": int(p["piece"]), "law": text(p["law"]), "t": [num(p["t_lo"], 4), num(p["t_hi"], 4)],
                       "n": int(p["n"]), "used": bool(p["used"]), "read": text(p["read"]) or None,
                       "value": num(p["value"]), "sigma": num(p["sigma"], 3), "weight": num(p["weight"], 3),
                       "displacement": num(p["displacement_value"]),
                       "rms": [num(x, 3) for x in p["rms_px"]] if p["rms_px"] else None,
                       "bound": [num(x, 3) for x in p["bound_px"]] if p["bound_px"] else None,
                       "rate": num(p["rate"], 4),
                       "depth": None if dz is None else {"z": num(req(dz, "Z", f"{wp} depth"), 4), "how": text(req(dz, "how", f"{wp} depth"))},
                       "why": text(p["why"])})
    un = req(pm, "unobserved", w)
    return {"pass": pm["pass"], "read": read, "piece": pm["piece"], "t": [num(pm["t_lo"], 4), num(pm["t_hi"], 4)],
            "law": text(pm["law"]) or None, "value": num(pm["value"]), "displacement": num(pm["displacement_value"]),
            "sigma_rel": num(pm["sigma_rel"], 3), "s": num(pm["s"]), "sigma_s": num(pm["sigma_s"], 3),
            "pieces": pieces, "unobserved": {"border": text(req(un, "border", f"{w} unobserved")) or None,
                                             "n": int(req(un, "n", f"{w} unobserved")),
                                             "ends": {k: text(v) for k, v in (req(un, "ends", f"{w} unobserved") or {}).items()}},
            "why": text(pm["why"])}


CLASS_SKIP_REASONS = ("self_reference", "observe.line_spacing", "prior.stated", "part.unmeasured", "track.none",
                      "presence.absent", "support.plane", "dims.agent_lateral", "dims.agent_height", "size.full_extent",
                      "fit.none", "fit.object", "dim.unobserved", "extent.frames",
                      "read.frames", "support.none", "focal.undetermined",
                      "dim.truncated", "dim.occluded")   # pipeline.scene_fit (rules C15, reference.size_only, C22, C15e)
# reasons only a fitted body gives: never in the refit's first pass at an undetermined stage-A f (rule C22)
CLASS_SKIP_FIT = ("fit.none", "fit.object", "dim.unobserved", "read.frames", "support.none")
CLASS_ROLES = ("cross-check", "carries", "vote")   # scale.class_reference.role (rules C15, class_vote.regular)
CLASS_VOTE_KEYS = ("source", "object", "class", "dim", "key", "value", "sigma", "role", "px", "extent", "extent_rel",
                   "n_frames", "frames", "z_how", "pass", "reason", "z")
ANCHOR_READ_KEYS = {"t", "Z", "frame", "f", "rel", "px", "frames", "how"}
ANCHOR_SKIP_KEYS = {"t", "Z", "frame", "skip", "why"}


def anchored_readings(rd, where: str) -> list:
    """Rule C15e focal.anchored_class_vote: per stated depth (t, Z) at its frame either a reading f = extent_px x Z / D
    {t, Z, frame, f, rel, px, frames, how} or a skip {t, Z, frame, skip, why} (the frame not fully seen, no extent)."""
    out = []
    for r in rd if isinstance(rd, list) else [None]:
        if not isinstance(r, dict) or set(r) not in (ANCHOR_READ_KEYS, ANCHOR_SKIP_KEYS):
            raise ValueError(f"{AT['where']}: {where} reading {r!r}")
        base = {"t": num(r["t"], 4), "Z": num(r["Z"], 4), "frame": int(r["frame"])}
        if "skip" in r:
            if r["skip"] not in CLASS_SKIP_REASONS:
                raise ValueError(f"{AT['where']}: {where} reading skip {r['skip']!r}")
            out.append(base | {"skip": r["skip"], "why": text(r["why"])})
        else:
            out.append(base | {"f": num(r["f"], 4), "rel": num(r["rel"], 3), "px": num(r["px"], 4),
                               "n_frames": len(r["frames"]), "how": text(r["how"])})
    return out


def anchored_view(a, v: dict, where: str):
    """An anchored class vote's record: the stated depths it read, f per reading, every reading. None: fit-read."""
    if a is None:
        if v["extent"] is None:
            raise ValueError(f"{AT['where']}: {where} has neither an extent nor an anchored record")
        return None
    if not isinstance(a, dict) or set(a) != {"t", "Z", "source", "f", "readings"} or a["source"] != "depth_info" \
            or not (len(a["t"]) == len(a["Z"]) == len(a["f"]) >= 1) or v["extent"] is not None \
            or v.get("f") is None or not str(v["z_how"]).startswith(f"{len(a['t'])} stated depth(s) at "):
        raise ValueError(f"{AT['where']}: {where} anchored {a!r} extent {v['extent']!r} z_how {v['z_how']!r}")
    rd = anchored_readings(a["readings"], where)
    if [r["f"] for r in rd if "f" in r] != [num(x, 4) for x in a["f"]]:
        raise ValueError(f"{AT['where']}: {where} anchored f {a['f']} differs from its readings")
    return {"t": [num(x, 4) for x in a["t"]], "Z": [num(x, 4) for x in a["Z"]], "readings": rd}


CLASS_REFERENCE_KEYS = ("name", "mode", "votes", "used", "skipped", "carries_scale", "stated", "value", "sigma", "pass",
                        "reason")


def class_reference_view(state: dict, scale_source: str) -> dict:
    """The class votes for the scene's scale (rule C15 scale.class_reference): mode "s" (a prior-scaled scene: s = class
    value / nominal extent) or "f" (a depth-stated scene: f x extent / class value); per vote the object, class, dim,
    class value +- sigma, the observed extent, s or f +- sigma, z against the stated source and pass; the skipped
    objects with their reason; the stated source; whether the votes carry the scale. A depth-stated scene's votes are
    its focal.fusion cross-check rows; votes that carry the scale make camera scale.source "class"."""
    cr = req(state, "scale.class_reference", "scene")
    w = "scale.class_reference"
    so = FMT.get("sizeonly")
    if set(cr) != set(CLASS_REFERENCE_KEYS) | ({"role"} if so else set()) or cr["name"] != w or cr["mode"] not in ("s", "f"):
        raise ValueError(f"{AT['where']}: {w} keys {sorted(cr)} name {cr.get('name')!r} mode {cr.get('mode')!r}")
    mode = cr["mode"]
    if (mode == "f") != (scale_source == "depth_info") or bool(cr["carries_scale"]) != (scale_source == "class"):
        raise ValueError(f"{AT['where']}: {w} mode {mode!r} carries_scale {cr['carries_scale']!r} with scale.source {scale_source!r}")
    # rule class_vote.regular (sizeonly): a stated source -> the votes cross-check it; none in a depth-stated scene ->
    # regular focal.fusion votes; none in a prior-scaled scene -> the votes carry s (scale.source "class", or "none"
    # when they do not agree)
    role = text(req(cr, "role", w)) if so else None
    if so and (role not in CLASS_ROLES or (role == "cross-check") != (cr["stated"] is not None)
               or (role == "vote") != (mode == "f" and cr["stated"] is None)
               or (role == "carries" and scale_source not in ("class", "none"))):
        raise ValueError(f"{AT['where']}: {w} role {role!r} mode {mode!r} stated {cr['stated']!r} scale.source {scale_source!r}")
    refs = set(req(state, "references", "scene")) if so else set()
    votes = []
    for v in cr["votes"]:
        wv = f"{w} vote {v.get('object')!r}"
        # a carrying vote has no z: there is no stated source to test it against (its verdict is the carry's, "reason")
        keys = set(CLASS_VOTE_KEYS) - ({"z"} if role == "carries" else set()) | {mode, f"sigma_{mode}"} \
            | ({"size_only", "frame_set"} if so else set()) | ({"anchored"} if FMT.get("anchored") else set())
        if set(v) != keys or v["source"] != f"class:{v['object']}" or v["pass"] not in (True, False, None):
            raise ValueError(f"{AT['where']}: {wv} keys {sorted(v)} source {v.get('source')!r} pass {v.get('pass')!r}")
        # rule reference.size_only: a reference's vote is its size read, on its track's frame set
        if so and (v["size_only"] is not (v["object"] in refs) or (v["frame_set"] is None) == v["size_only"]
                   or (v["role"] == "reference") != v["size_only"]):
            raise ValueError(f"{AT['where']}: {wv} size_only {v['size_only']!r} frame_set {v['frame_set']!r} role "
                             f"{v['role']!r}, references {sorted(refs)}")
        votes.append({"object": text(v["object"]), "class": text(v["class"]), "dim": text(v["dim"]), "key": text(v["key"]),
                      "role": text(v["role"]), "value": num(v["value"], 4), "sigma": num(v["sigma"], 3),
                      "px": num(v["px"], 4), "extent": num(v["extent"], 4), "extent_rel": num(v["extent_rel"], 3),
                      "n_frames": int(v["n_frames"]), "frames": text(v["frames"]), "z_how": text(v["z_how"]),
                      "x": num(v[mode], 4), "sigma_x": num(v[f"sigma_{mode}"], 3), "z": num(v.get("z"), 3),
                      "pass": v["pass"], "reason": text(v["reason"])}
                     | ({"size_only": v["size_only"], "frame_set": text(v["frame_set"]) or None} if so else {})
                     | ({"anchored": anchored_view(v["anchored"], v, wv)} if FMT.get("anchored") else {}))
    skipped = []
    for x in cr["skipped"]:
        ws = f"{w} skipped {x.get('name')!r}"
        r = text(req(x, "reason", ws))
        if r not in CLASS_SKIP_REASONS:
            raise ValueError(f"{AT['where']}: {ws} reason {r!r}")
        if "anchored_readings" in x and not FMT.get("anchored"):
            raise ValueError(f"{AT['where']}: {ws} anchored_readings before batch 29")
        skipped.append({"object": text(req(x, "name", ws)), "class": text(req(x, "class", ws)), "reason": r,
                        "why": text(req(x, "why", ws))}
                       | ({"readings": anchored_readings(x["anchored_readings"], ws)} if "anchored_readings" in x else {}))
    names = [v["object"] for v in votes]
    # "used" names the votes by source ("class:<object>")
    by_source = {f"class:{n}": n for n in names}
    if not set(cr["used"]) <= set(by_source):
        raise ValueError(f"{AT['where']}: {w} used {cr['used']} not among the votes' sources {sorted(by_source)}")
    if mode == "f":
        fu = req(req(state, "focal", "scene"), "fusion", "focal")
        # the fusion's copy leaves out the votes: they are its "class reference" source rows
        if req(fu, "class_reference", "focal.fusion") != {k: v for k, v in cr.items() if k != "votes"}:
            raise ValueError(f"{AT['where']}: focal.fusion.class_reference differs from {w} (without its votes)")
        rows = {text(x["object"]): x for x in req(fu, "sources", "focal.fusion") if x.get("kind") == "class reference"}
        # cross-check rows carry the vote's pass; regular votes (role "vote") are used / outlier, pass = used
        same = (lambda x, v: x["verdict"] == "cross-check" and x["pass"] == v["pass"]) if role != "vote" else \
            (lambda x, v: x["verdict"] in ("used", "outlier") and "pass" not in x and (x["verdict"] == "used") == v["pass"])
        if set(rows) != set(names) or any(num(rows[v["object"]]["f"], 4) != v["x"] or not same(rows[v["object"]], v)
                                          for v in votes):
            raise ValueError(f"{AT['where']}: focal.fusion class-reference rows {sorted(rows)} differ from {w} votes {names}")
    st = cr["stated"]
    stated = None if st is None else {"value": num(req(st, "value", f"{w} stated"), 4),
                                      "sigma": num(req(st, "sigma", f"{w} stated"), 3),
                                      "source": text(req(st, "source", f"{w} stated"))}
    if cr["pass"] not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass {cr['pass']!r}")
    return {"mode": mode, "role": role, "votes": votes, "used": [by_source[x] for x in cr["used"]], "skipped": skipped,
            "carries": bool(cr["carries_scale"]), "stated": stated, "value": num(cr["value"], 4),
            "sigma": num(cr["sigma"], 3), "pass": cr["pass"], "reason": text(cr["reason"])}


REJECT_REASONS = ("class.unknown", "class.not_answered", "name.listed", "name.ground", "name.duplicate", "references.max",
                  "scene.rendered")   # pipeline.objects (rule C15 part 1)
OBJECT_ROLES = ("object", "structure", "reference", "class")   # "class": an instance of a class prior (objects.py)


RENDER_CUES = ("synthetic_surface", "engine_overlay", "impossible_camera", "engine_objects", "clean_image",
               "animation_style")   # pipeline.objects.RENDER_CUES (rule C15d scene.rendered)
RENDER_CUE_REJECTS = ("cue.kind_unknown", "cue.frame_unknown", "cue.why_missing")
RENDER_UNSUPPORTED = "rendered.unsupported"


def rendered_view(r: dict | None, w: str) -> dict | None:
    """Rule C15d scene.rendered (inventory format 9): the model's answer and the verdict -- true only with at least one
    valid cue {kind, where (a grid frame's stamp), why}; a true answer without one is rendered.unsupported. The record
    of an unresolved reference call has verdict / answer null."""
    keys = {"verdict", "answer", "cues", "cues_rejected", "why", "reason"}
    if r is None or set(r) != keys:
        raise ValueError(f"{AT['where']}: {w} {r!r}")
    cues = []
    for c in r["cues"]:
        if set(c) != {"kind", "where", "why"} or c["kind"] not in RENDER_CUES or not text(c["why"]):
            raise ValueError(f"{AT['where']}: {w} cue {c!r}")
        cues.append({"kind": c["kind"], "where": text(c["where"]), "why": text(c["why"])})
    bad = []
    for c in r["cues_rejected"]:
        if set(c) != {"kind", "where", "why", "reason"} or c["reason"] not in RENDER_CUE_REJECTS:
            raise ValueError(f"{AT['where']}: {w} rejected cue {c!r}")
        bad.append({"kind": text(c["kind"]), "where": text(c["where"]), "why": text(c["why"]), "reason": c["reason"]})
    v, a = r["verdict"], r["answer"]
    if v is None:
        if a is not None or cues or bad or r["reason"] is not None:
            raise ValueError(f"{AT['where']}: {w} null verdict with {r!r}")
    elif not isinstance(v, bool) or not isinstance(a, bool) or v != (a and bool(cues)) \
            or r["reason"] != (RENDER_UNSUPPORTED if a and not cues else None):
        raise ValueError(f"{AT['where']}: {w} verdict {v!r} answer {a!r} cues {len(cues)} reason {r['reason']!r}")
    return {"verdict": v, "answer": a, "cues": cues, "rejected": bad, "why": text(r["why"]) or None, "reason": r["reason"]}


def inventory_view(unit: str) -> dict:
    """The inventory's class references (format 7, rule C15 part 1): whether the video is a rendering, the class of
    every listed object (self: the question asks its own size), the reference objects the texts do not name with the
    model's reason, and the rejected ones with their reason. Every size_reference is pipeline.class_sizes'."""
    from pipeline.class_sizes import TABLE_VERSION, size_reference
    inv = unit_json("objects", unit, "object inventory")
    w = "object inventory"
    fmt = 9 if FMT.get("rendered") else 8 if FMT.get("sizeonly") else 7
    if req(inv, "version", w) != fmt or req(inv, "class_table", w) != TABLE_VERSION:
        raise ValueError(f"{AT['where']}: {w} version {inv.get('version')!r} class_table {inv.get('class_table')!r}")
    rendered, refs = req(inv, "rendered", w), req(inv, "references", w)
    if FMT.get("rendered"):
        rendered = rendered_view(rendered, f"{w} rendered")
    elif rendered not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} rendered {rendered!r}")
    if req(refs, "status", f"{w} references") not in ("ok", "unresolved"):
        raise ValueError(f"{AT['where']}: {w} rendered {rendered!r} references {refs!r}")
    objs, references = [], []
    for o in req(inv, "objects", w):
        wo = f"{w} object {o.get('name')!r}"
        key, sr, self_ref = req(o, "size_class", wo), req(o, "size_reference", wo), req(o, "self_reference", wo)
        if not isinstance(self_ref, bool) or (sr is not None and (key is None or sr != size_reference(key))):
            raise ValueError(f"{AT['where']}: {wo} size_class {key!r} self_reference {self_ref!r} size_reference "
                             f"differs from pipeline.class_sizes")
        ex = o.get("extra") or {}
        # rule reference.size_only (format 8): one role, "reference" at the top level and in extra, or neither
        if FMT.get("sizeonly") and (req(o, "role", wo) == "reference") != (ex.get("role") == "reference"):
            raise ValueError(f"{AT['where']}: {wo} role {o.get('role')!r} extra.role {ex.get('role')!r}")
        if ex.get("role") == "reference":
            if sr is None:
                raise ValueError(f"{AT['where']}: {wo} reference object without a size_reference")
            references.append({"name": text(o["name"]), "class": text(key), "reason": text(req(ex, "reason", f"{wo} extra"))})
        elif key is not None:
            objs.append({"name": text(o["name"]), "class": text(key), "self": self_ref, "accepted": sr is not None})
    rejected = []
    for x in req(inv, "reference_rejected", w):
        wr = f"{w} reference_rejected {x.get('name')!r}"
        r = text(req(x, "reason", wr))
        if r not in REJECT_REASONS:
            raise ValueError(f"{AT['where']}: {wr} reason {r!r}")
        rejected.append({"name": text(req(x, "name", wr)), "class": text(req(x, "class", wr)) or None, "reason": r,
                         "why": text(req(x, "why", wr))})
    if refs["status"] == "unresolved" and (req(refs, "reason", f"{w} references") != "objects.references_invalid"
                                           or not text(req(refs, "why", f"{w} references"))):
        raise ValueError(f"{AT['where']}: {w} unresolved references {refs!r}")
    return {"rendered": rendered, "status": text(refs["status"]), "why": text(refs.get("why")) or None,
            "objects": objs, "references": references, "rejected": rejected}


FULL_EXTENT_VARIANTS = ("nominal plane", "depth-model samples", "no depth")
FULL_EXTENT_EXCLUDED = ("overlap", "shrink", "no_depth", "border")


def scale_full_extent_view(fe: dict):
    """The articulated agents' size.full_extent re-run at the nominal scene's depth before the prior scale is measured
    (rule size.full_extent_nominal): [{agent, variant, depth, pass, used (frame count), excluded {kind: frames text}}]
    | None (no agent)."""
    out = []
    for name, x in fe.items():
        w = f"scale.full_extent {name!r}"
        v, ok, ex = text(req(x, "variant", w)), req(x, "pass", w), req(x, "frames_excluded", w)
        if v not in FULL_EXTENT_VARIANTS or ok not in (True, False, None) or set(ex) != set(FULL_EXTENT_EXCLUDED):
            raise ValueError(f"{AT['where']}: {w} variant {v!r} pass {ok!r} excluded {sorted(ex)}")
        out.append({"agent": text(name), "variant": v, "depth": text(req(x, "depth", w)), "pass": ok,
                    "used": int(req(x, "frames_used", w)),
                    "excluded": {k: text(ex[k]) for k in FULL_EXTENT_EXCLUDED if text(ex[k]) not in ("", "[]")}})
    return out or None


def named_checks(checks: dict, failed: list, what: str) -> list[dict]:
    """[{"name", "pass", "reason"}] of a record's named checks; `failed` must be exactly the ones that did not pass."""
    out = [{"name": k, "pass": bool(req(c, "pass", f"{what} check {k!r}")), "reason": text(req(c, "reason", f"{what} check {k!r}"))}
           for k, c in checks.items()]
    if failed != [c["name"] for c in out if not c["pass"]]:
        raise ValueError(f"{AT['where']}: {what} failed {failed} differs from its checks that did not pass")
    return out


GRAVITY_SOURCES = ("earth", "inferred", "prior", "unknown")


def up_source(state: dict, g: dict):
    """The gravity record's up direction source; a prior-scaled scene without a metric scale (camera scale.source
    "none", camera metric false: scene_fit does not measure g) has none -- its record is exactly
    {g, source, sigma null, rule, evidence [], checks {}, failed []}."""
    cam = req(state, "camera", "scene")
    if FMT.get("sizeonly") and cam.get("scale.source") == "none":
        keys = {"g", "source", "sigma", "rule", "evidence", "checks", "failed"} | ({"up", "up_source"} if FMT.get("up") else set())
        if cam.get("metric") is not False or set(g) != keys \
                or g["sigma"] is not None or g["evidence"] or g["checks"] or g["failed"]:
            raise ValueError(f"{AT['where']}: gravity {sorted(g)} in a scene without a metric scale")
        if not FMT.get("up"):
            return None
    return text(req(g, "up_source", "gravity"))


UP_PLANE = "support plane normal"
UP_CAMERA = ("camera -y (no support plane)", "camera -y (soft support plane)")
UP_PRIOR = "the camera prior (level)"


def gravity_up_view(state: dict) -> dict:
    """Rule C23b gravity.up_not_from_soft_plane: the scene's up -- the support plane's normal when the plane is not
    soft, else roll and pitch each from the first source that observes it (image verticals, ground-line horizon,
    visible horizon, another determined plane, the camera prior). {up, up_source, roll {deg, source}, pitch {deg,
    source}, skipped [{source, why}], why}; up_source agrees with the angles' sources and with gravity.up_source."""
    w = "gravity.up"
    x = req(state, w, "scene")
    if set(x) != {"name", "up", "up_source", "roll", "pitch", "skipped", "why"} or x["name"] != w:
        raise ValueError(f"{AT['where']}: scene {w} keys {sorted(x)}")
    ang = {}
    for k in ("roll", "pitch"):
        a = x[k]
        if not isinstance(a, dict) or set(a) != {"deg", "source"}:
            raise ValueError(f"{AT['where']}: scene {w} {k} {a!r}")
        ang[k] = {"deg": num(a["deg"], 4), "source": text(a["source"])}
    src, rs, ps = text(x["up_source"]), ang["roll"]["source"], ang["pitch"]["source"]
    if src == UP_PLANE:
        ok = rs == ps == UP_PLANE
    elif src in UP_CAMERA:
        ok = rs == ps == UP_PRIOR
    elif rs == ps:
        ok = src == rs and rs not in (UP_PLANE, UP_PRIOR)
    else:
        ok = src == f"roll: {rs}; pitch: {ps}"
    if not ok:
        raise ValueError(f"{AT['where']}: scene {w} up_source {src!r} with roll {rs!r}, pitch {ps!r}")
    up = x["up"]
    if not isinstance(up, list) or len(up) != 3 or abs(sum(c * c for c in up) - 1.0) > 1e-6:
        raise ValueError(f"{AT['where']}: scene {w} up {up!r}")
    g = req(state, "gravity", "scene")
    gu = g.get("up")
    if g.get("up_source") != src or gu is None or len(gu) != 3 or max(abs(a - b) for a, b in zip(gu, up)) > 1e-4:
        raise ValueError(f"{AT['where']}: gravity up {gu!r} / {g.get('up_source')!r} differs from scene {w}")
    skipped = []
    for r in x["skipped"]:
        if not isinstance(r, dict) or set(r) != {"source", "why"}:
            raise ValueError(f"{AT['where']}: scene {w} skipped {r!r}")
        skipped.append({"source": text(r["source"]), "why": text(r["why"])})
    if (src == UP_CAMERA[1] or src not in (UP_PLANE, *UP_CAMERA)) and not skipped and \
            req(state, "support.undetermined_soft", "scene")["pass"] is False:
        raise ValueError(f"{AT['where']}: scene {w} on a soft plane names no skipped source")
    return {"up": [num(c, 4) for c in up], "up_source": src, "roll": ang["roll"], "pitch": ang["pitch"],
            "skipped": skipped, "why": text(x["why"])}


def gravity_view(state: dict):
    """The scene's gravity (scene_fit.gravity_pass, rule B2): g, sigma, source, rule, up source, every evidence entry
    (a ballistic piece with its free-g fit and residual test, or a physics-chain probe) and the gravity checks."""
    if not FMT.get("gravity"):
        return None
    g = req(state, "gravity", "scene")
    src = text(req(g, "source", "gravity"))
    if src not in GRAVITY_SOURCES:
        raise ValueError(f"{AT['where']}: gravity source {src!r}")
    ev = []
    for e in req(g, "evidence", "gravity"):
        w = f"gravity evidence {e.get('object')!r}"
        kind = text(req(e, "kind", w))
        if kind == "ballistic_piece":
            law, adm = req(e, "law", w), req(e, "admissible", w)
            ev.append({"obj": text(req(e, "object", w)), "kind": kind, "t": [num(x, 4) for x in req(e, "t", w)],
                       "n": req(e, "n", w), "g": num(req(e, "g", w)), "sigma": num(req(e, "sigma", w), 3),
                       "law": {"pass": bool(req(law, "pass", f"{w} law")), "run": req(law, "run", f"{w} law"),
                               "n_test": req(law, "n_test", f"{w} law")},
                       "admissible": {"pass": bool(req(adm, "pass", f"{w} admissible")),
                                      "reason": text(req(adm, "reason", f"{w} admissible"))},
                       "use": bool(req(e, "use", w)), "why": text(e.get("why")) or None})
        elif kind == "physics_probe" and FMT.get("contact") and "excluded" in e:
            # rule fit.object (scene_fit.gravity_pass): the probe of a failed object fit is listed, excluded
            if set(e) != {"object", "kind", "g", "excluded"}:
                raise ValueError(f"{AT['where']}: {w} excluded probe keys {sorted(e)}")
            ev.append({"obj": text(e["object"]), "kind": kind, "g": num(e["g"]), "rho": None, "agree": None,
                       "excluded": text(e["excluded"])})
        elif kind == "physics_probe":
            ev.append({"obj": text(req(e, "object", w)), "kind": kind, "g": num(req(e, "g", w)),
                       "rho": num(req(e, "rho", w), 3), "agree": bool(req(e, "agree", w))})
        else:
            raise ValueError(f"{AT['where']}: {w} kind {kind!r}")
    et = g.get("earth_test")
    if (et is not None) != (src == "inferred"):
        raise ValueError(f"{AT['where']}: gravity source {src!r} with earth_test {et!r}")
    return {"g": num(req(g, "g", "gravity")), "sigma": num(req(g, "sigma", "gravity"), 3), "source": src,
            "earth_test": {"pass": bool(req(et, "pass", "earth_test")), "reason": text(req(et, "reason", "earth_test"))}
            if et is not None else None,
            "rule": text(req(g, "rule", "gravity")), "up_source": up_source(state, g), "evidence": ev,
            "checks": named_checks(req(g, "checks", "gravity"), list(req(g, "failed", "gravity")), "gravity")}


def refmeasure_view(tag: str, unit: str):
    """The unit's reference measurements (pipeline.refmeasure record, keyed "<object>|<quantity>"): object, quantity,
    graduation unit, length in px, its frame and named checks; None when the unit has no record."""
    if not FMT.get("refmeasure"):
        return None
    from pipeline.refmeasure import VERSION as REFM_VERSION
    p = ROOT / "outputs" / f"refmeasure_v{REFM_VERSION}_{tag}" / f"{unit}.json"
    if not p.exists():
        return None
    j = read_json(p, "refmeasure record")
    out = []
    for key, e in j.items():
        w = f"refmeasure {key!r}"
        if req(e, "unit_id", w) != unit:
            raise ValueError(f"{AT['where']}: {w} is for unit {e['unit_id']!r}")
        out.append({"obj": text(req(e, "object", w)), "quantity": text(req(e, "quantity", w)),
                    "method": text(e.get("method")) or None, "unit": text(e.get("unit")) or None,
                    "px": num(e.get("px")), "frame": e.get("frame"), "pass": bool(req(e, "pass", w)),
                    "checks": named_checks(req(e, "checks", w), list(req(e, "failed", w)), w)})
    return out or None


def dims_check(state: dict) -> list[dict]:
    """plausible.dims per object (pipeline.critic, rule D2a): the largest dimension as an interval [lo, hi | None]
    (a bound dimension contributes its dim_bounds interval) against the class range widened by critic.SIZE_MARGIN;
    fails only when the whole interval is outside. Recomputed from the saved scene and checked against the scene's
    own checks.plausible.parts.dims."""
    from pipeline.critic import SIZE_MARGIN, _class_range, _size_interval, fit_failed
    rows = []
    for name, o in req(state, "objects", "scene").items():
        lo, hi, _ = _size_interval(o)
        rng = _class_range(name)
        tested = rng is not None and not (hi is not None and hi <= 0)
        ok = not tested or not (lo > rng[1] * SIZE_MARGIN or (hi is not None and hi < rng[0] / SIZE_MARGIN))
        # rule critic.dims_untestable: a failed object fit is listed as untestable, not judged
        if FMT.get("contact") and fit_failed(o):
            ok = None
        rows.append({"obj": name, "lo": num(lo), "hi": None if hi is None else num(hi),
                     "range": [num(rng[0] / SIZE_MARGIN), num(rng[1] * SIZE_MARGIN)] if tested else None, "pass": ok})
    part = req(req(req(req(state, "checks", "scene"), "plausible", "scene checks"), "parts", "plausible"), "dims", "plausible parts")
    if req(part, "pass", "plausible.dims") != all(r["pass"] is not False for r in rows):
        raise ValueError(f"{AT['where']}: plausible.dims {part['pass']} differs from the recomputed intervals {rows}")
    return rows


def plausible_untestable(state: dict):
    """The plausible parts that could not judge an object (critic.agency_unknown): [[part, [reason]]] or None."""
    parts = req(req(req(state, "checks", "scene"), "plausible", "scene checks"), "parts", "plausible")
    out = [[text(k), [text(x) for x in v["untestable"]]] for k, v in parts.items() if v.get("untestable")]
    return out or None


def identity_view(key: str) -> dict:
    """The track identity judge of every object (track cache "meta"."identity"): its candidates in the order they were
    judged, with the outcome and why; the chosen one, or why the object is untracked."""
    j = unit_json("tracks", key, "track cache")
    out = {}
    for name, v in req(req(j, "meta", "track cache"), "identity", "track cache meta").items():
        w = f"identity of {name!r}"
        cands, chosen, untracked = req(v, "candidates", w), req(v, "chosen", w), req(v, "untracked", w)
        if (chosen is None) == (untracked is None):
            raise ValueError(f"{AT['where']}: {w} has chosen {chosen!r} and untracked {untracked!r}")
        if chosen is not None and req(cands[chosen], "outcome", w) != "accepted":
            raise ValueError(f"{AT['where']}: {w} chose candidate {chosen}, outcome {cands[chosen]['outcome']!r}")
        rows = []
        for c in cands:
            pres, mv = c.get("presence"), c.get("moves")
            if FMT.get("contained") and mv and (not {"moves", "px", "centre", "size", "evidence"} <= set(mv) or set(mv) - MOVES_KEYS):
                raise ValueError(f"{AT['where']}: {w} candidate moves keys {sorted(mv)}")
            if FMT.get("contained") and text(c.get("outcome")).startswith("dropped") and c["outcome"] != "dropped: candidate.full_frame":
                raise ValueError(f"{AT['where']}: {w} candidate outcome {c['outcome']!r}")
            rows.append({"source": text(req(c, "source", w)), "frame": req(c, "frame", w),
                         "box": [round(float(x)) for x in c["box"]] if req(c, "box", w) else None,
                         "n": c.get("n_frames") if text(c.get("outcome")).startswith("dropped") else req(c, "n_frames", w), "px": num(mv["px"], 3) if mv else None,
                         "moves": bool(mv["moves"]) if mv else None,
                         "votes": list(c["votes"]) if c.get("votes") else None,
                         "size": [{"ref": text(x["ref"]), "ratio": num(x["ratio"], 3),
                                   "bounds": [num(b, 3) if b is not None else None for b in x["bounds"]],
                                   "view": text(x["view"]).split(" (")[0], "pass": bool(x["pass"])}
                                  | ({"dims": {k: text(req(req(x, "dims", w), k, w)) for k in ("candidate", "ref")}}
                                     if FMT.get("kept") else {})
                                  for x in c.get("class_size") or []],
                         "span": [pres["first"], pres["last"]] if pres and pres.get("pass") else None,
                         "border": num(c["border_frac"], 2) if c.get("border_frac") is not None else None,
                         "outcome": text(req(c, "outcome", w)), "why": text(c.get("why")) or None}
                        | ({"terms": {k: num(mv[k], 3) for k in ("centre", "size", "evidence")}
                                     | ({"window": mv["window"]} if "window" in mv else {})} if FMT.get("contained") and mv else {}))
        out[name] = {"candidates": rows, "chosen": chosen, "untracked": text(untracked) or None}
        if FMT.get("contained"):
            out[name]["whole_set"] = whole_set_view(v["whole_set"], len(cands), w) if "whole_set" in v else None
    if FMT.get("contained"):
        choose = req(j["meta"], "choose", "track cache meta")
        for name, c in choose.items():
            if name not in out:
                raise ValueError(f"{AT['where']}: track meta choose names {name!r} without an identity record")
            out[name]["choice"] = choice_view(c, f"choose of {name!r}")
            ib = j["meta"]["identity"][name].get("binding")
            cb = c.get("binding")
            if (ib is None) != (cb is None) or (ib is not None and binding_view(ib, f"identity of {name!r}") != out[name]["choice"]["binding"]):
                raise ValueError(f"{AT['where']}: identity binding {ib!r} differs from the choose record's {cb!r}")
            if cb is not None and cb["pass"] != (out[name]["chosen"] is not None):
                raise ValueError(f"{AT['where']}: binding pass {cb['pass']} vs chosen {out[name]['chosen']!r}")
    return out


MOVES_KEYS = {"moves", "px", "centre", "size", "evidence", "window"}   # rule E15 identity.moves


def whole_set_view(x: dict, n: int, w: str) -> dict:
    """Rule E15 identity.whole_set: every candidate failed and some failed only "static" while the object's motion is
    asserted -- judged again most-moving first; {chosen, terms [{candidate, centre, size, evidence}], why}."""
    w = f"{w} whole_set"
    if set(x) != {"rule", "chosen", "motion_terms", "why"} or x["rule"] != "ranked" or \
            (x["chosen"] is not None and not 0 <= int(x["chosen"]) < n):
        raise ValueError(f"{AT['where']}: {w} {x!r}")
    return {"chosen": x["chosen"], "why": text(x["why"]),
            "terms": [{"candidate": req(t, "candidate", w), "centre": num(req(t, "centre", w), 3), "size": num(req(t, "size", w), 3),
                       "evidence": num(req(t, "evidence", w), 3)} for t in x["motion_terms"]]}


CHOICE_RULES = ("shared.full", "own.best_full", "own.nearest_full", "own.best_partial", "own.nearest_partial",
                "shared.no_scan", "shared.absent_everywhere", "own.present_interval")   # rule E20 choice.frame_stable
CHOOSE_RULES = ("choice", "choice.runs", "choice.whole_set", "recheck", "motion_check", "alternative",
                "alternative.runs", "alternative.whole_set", "untracked")   # rule E17 choose.track_binding
BINDING_KEYS = {"name", "pass", "choice_frame", "accepted_frame", "box_iou_at_start", "bound", "start_kept", "why"}
REJECTED_CHOICE_KEYS = {"frame", "box", "source", "why", "choice", "choices", "reason"}


def binding_view(b: dict, w: str) -> dict:
    """Rule E17 choose.track_binding: the choose record's box against the propagation's own box on its frame."""
    if set(b) != BINDING_KEYS or b["name"] != "choose.track_binding" or not isinstance(b["pass"], bool):
        raise ValueError(f"{AT['where']}: {w} binding {b!r}")
    return {"pass": b["pass"], "choice_frame": b["choice_frame"], "accepted_frame": b["accepted_frame"],
            "iou": num(b["box_iou_at_start"], 3), "bound": num(b["bound"], 3), "start_kept": bool(b["start_kept"]),
            "why": text(b["why"])}


def choice_view(c: dict, w: str) -> dict:
    """Rules E15 / E20 / E17: the frame each object's box was chosen on (choice.frame_stable: the shared probe frame
    unless the scan has the object only partly there) with the rule and why, the scan's labels per probe frame, the
    detector boxes dropped as the image itself (candidate.full_frame), and the record's binding to the accepted track
    (choose.track_binding): where the track starts, the chooser's picks it did not take, the box IoU at the start."""
    # v6: a record whose box came from the alternative rule has no probe-frame scan
    cf = req(c, "choice_frame", w)
    sc = c.get("scan") if FMT.get("sceneprog") and c.get("rule") == "alternative" else req(c, "scan", w)
    if set(cf) != {"frame", "why", "shared", "shared_status", "rule"} or cf["rule"] not in CHOICE_RULES or \
            cf["shared_status"] not in ("full", "partial", "absent", None) or cf["shared"] != req(c, "shared_frame", w) or \
            (sc is not None and (set(sc) != {"labels", "best"} or
                                 any(v not in ("full", "partial", "absent") for v in sc["labels"].values()))):
        raise ValueError(f"{AT['where']}: {w} choice_frame {cf!r} scan {sc!r}")
    dropped = []
    for d in c.get("dropped") or []:
        if set(d) != {"box", "why", "outcome"}:
            raise ValueError(f"{AT['where']}: {w} dropped keys {sorted(d)}")
        dropped.append({"box": [round(float(x)) for x in d["box"]], "why": text(d["why"]), "outcome": text(d["outcome"])})
    rule, untracked = req(c, "rule", w), c.get("untracked")
    if rule not in CHOOSE_RULES or (rule == "untracked") != (untracked is not None):
        raise ValueError(f"{AT['where']}: {w} rule {rule!r} untracked {untracked!r}")
    rejected = []
    for r in req(c, "rejected", w):
        # v6: a chooser that boxed nothing records no choice / choices / reason
        if set(r) != REJECTED_CHOICE_KEYS and not (FMT.get("sceneprog") and set(r) == {"frame", "box", "source", "why"}):
            raise ValueError(f"{AT['where']}: {w} rejected keys {sorted(r)}")
        rejected.append({"frame": r["frame"], "box": [round(float(x)) for x in r["box"]] if r["box"] else None,
                         "source": text(r["source"]), "why": text(r["why"])})
    af = req(c, "accepted_frame", w)
    if untracked is not None:
        if af is not None or c["box"] is not None or c["source"] != "none":
            raise ValueError(f"{AT['where']}: {w} untracked record names a box / start")
        binding = binding_view(c["binding"], w) if "binding" in c else None    # a failed binding untracks it
        if binding is not None and binding["pass"]:
            raise ValueError(f"{AT['where']}: {w} untracked with a passing binding")
    else:
        if set(af) != {"frame", "why"} or af["frame"] != c["frame"]:
            raise ValueError(f"{AT['where']}: {w} accepted_frame {af!r} vs record frame {c['frame']!r}")
        binding = binding_view(req(c, "binding", w), w)
        if binding["choice_frame"] != cf["frame"] or binding["accepted_frame"] != af["frame"]:
            raise ValueError(f"{AT['where']}: {w} binding frames {binding} vs choice {cf['frame']} / start {af['frame']}")
    return {"frame": cf["frame"], "why": text(cf["why"]), "rule": cf["rule"], "shared": cf["shared"],
            "shared_status": cf["shared_status"],
            "scan": [[int(k), v] for k, v in sorted(sc["labels"].items(), key=lambda kv: int(kv[0]))] if sc else None,
            "best": sc["best"] if sc else None,
            "dropped": dropped, "record_rule": rule, "source": text(c["source"]),
            "start": {"frame": af["frame"], "why": text(af["why"])} if af else None,
            "rejected": rejected, "binding": binding}


def focal_view(state: dict):
    """The scene's focal record (rule focal.gravity_hop): one row per hop of every passive object's airborne edges."""
    fo = req(state, "focal", "scene")
    rows = []
    for x in req(fo, "votes", "focal"):
        w = "focal vote"
        law = req(x, "law", w)
        rows.append({"object": text(req(x, "object", w)), "span": [num(a, 4) for a in req(x, "span", w)],
                     "hop": [num(a, 4) for a in req(x, "hop", w)], "f": num(req(x, "f", w), 4),
                     "sigma": num(req(x, "sigma", w), 3), "law": text(req(law, "result", w)),
                     "n": x.get("n"), "rms": num(x.get("rms"), 3), "used": bool(req(x, "used", w)),
                     "why": text(req(x, "why", w))})
    return {"votes": rows, "f_votes": num(fo.get("f_votes"), 4), "sigma": num(fo.get("sigma"), 3),
            "n_votes": fo.get("n_votes"), "f": num(req(fo, "f", "focal"), 4), "source": text(req(fo, "source", "focal"))}


def horizon_view(state: dict):
    """The support plane's pitch check (rule support.horizon) and the support plane's pitch source; None when the
    scene has no support plane record and no horizon record."""
    sh = req(state, "support_horizon", "scene")
    sp = (req(state, "camera", "scene") or {}).get("support_plane")
    plane = None
    if sp:
        dp = sp.get("depth_plane")
        plane = {"tilt": num(sp.get("tilt_from_up_deg"), 3), "pitch_source": text(sp.get("pitch_source")) or None,
                 "depth_tilt": num(dp.get("tilt_from_up_deg"), 3) if dp else None}
    if sh is None:
        return {"plane": plane, "check": None} if plane else None
    hz = req(sh, "horizon", "support_horizon")
    far = hz.get("far")
    check = {"pass": sh.get("pass"), "reason": text(req(sh, "reason", "support_horizon")), "repaired": bool(sh.get("repaired")),
             "phi_plane": num(sh.get("phi_plane_deg"), 3), "phi_horizon": num(sh.get("phi_horizon_deg"), 3),
             "sigma": num(sh.get("sigma_deg"), 2), "z": num(sh.get("z"), 2),
             "row_plane": num(sh.get("row_plane"), 4), "row_horizon": num(sh.get("row_horizon"), 4),
             "status": text(req(hz, "status", "horizon")), "row": num(hz.get("row"), 4), "sigma_row": num(hz.get("sigma_row"), 2),
             "spread": num(hz.get("spread_px"), 3), "n_found": hz.get("n_found"), "n_frames": hz.get("n_frames"),
             "far": {"band": num(far.get("band"), 3), "support_p95": num(far.get("support_p95"), 3)} if far else None}
    return {"plane": plane, "check": check}


def camera_roll_view(state: dict) -> dict:
    """The scene's roll check (rule camera.roll): the roll the scene used against the roll of the image's
    near-vertical segments, and the pose refit when they disagree."""
    cr = req(state, "camera_roll", "scene")
    ok = req(cr, "pass", "camera_roll")
    out = {"pass": ok, "reason": text(req(cr, "reason", "camera_roll")), "source": text(req(cr, "roll_source", "camera_roll")) or None,
           "fit": num(cr.get("roll_fit_deg"), 6), "how": text(cr.get("roll_fit_how")) or None, "image": None, "refit": None}
    im = cr.get("image")
    if im:
        out["image"] = {"status": text(req(im, "status", "camera_roll image")), "why": text(req(im, "why", "camera_roll image")),
                        "n_lines": im.get("n_lines"), "n_frames": im.get("n_frames")}
    if ok is not None:
        out |= {"image_roll": num(req(cr, "roll_image_deg", "camera_roll"), 6), "sigma": num(req(cr, "sigma_deg", "camera_roll"), 4),
                "z": num(req(cr, "z", "camera_roll"), 3)}
    rf = cr.get("refit")
    if rf is not None:
        if ok is not False:
            raise ValueError(f"{AT['where']}: camera_roll refit without a failed check")
        # the main support's evidence plane (rule support.main) is re-rolled without a stage-A refit: no rms
        keys = ("roll_deg", "pitch_deg", "dist") + \
            (("rms",) if not (FMT.get("contact") and state.get("main_plane")) or "rms" in req(rf, "was", "camera_roll refit") else ())
        out["refit"] = {k: [num(req(rf[s], k, f"camera_roll refit {s}"), 6) for s in ("was", "now")] for k in keys}
    return out


def extents_view(tag: str, unit: str, state: dict | None):
    """The unit's mask extents: per tracked object the frames and overlay-ring frames of pipeline.extents2d format 3
    (rule extent.overlay), and per object, dimension and anchor the blur-free extent the scene used (scene json
    "extents", rule extent.static): value +- sigma, frames used, frames rejected with why. None without either."""
    p = ROOT / "outputs" / f"extents2d_{tag}" / f"{unit}.json"
    rows = []
    if p.exists():
        j = read_json(p, "extents2d record")
        if req(j, "format", "extents2d record") != 3 or req(j, "unit", "extents2d record") != unit:
            raise ValueError(f"{AT['where']}: extents2d record format {j.get('format')} / unit {j.get('unit')!r}")
        kinds = req(j, "tracks", "extents2d record")
        for kind, objs in kinds.items():
            for name, frames in objs.items():
                w = f"extents2d {kind} {name!r}"
                fs = sorted((int(k), v) for k, v in frames.items())
                if not fs:
                    continue
                rings = [float(req(v, "ring_px", w)) for k, v in fs if req(v, "overlay", w)]
                rows.append({"obj": name, "kind": kind if len(kinds) > 1 else None, "n": len(fs), "frames": [fs[0][0], fs[-1][0]],
                             "overlay": len(rings), "ring": num(float(np.median(rings)), 3) if rings else None, "at": []})
    if FMT.get("static") and state is not None:
        ex = req(state, "extents", "scene")
        for name, o in req(state, "objects", "scene").items():
            oe = req(o, "extent", f"scene object {name!r}")
            if (oe or {}) != (ex.get(name) or {}):
                raise ValueError(f"{AT['where']}: object {name!r} extent differs from the scene's extents record")
        for name, dims in ex.items():
            row = next((r for r in rows if r["obj"] == name), None)
            if row is None:
                row = {"obj": name, "kind": None, "n": None, "frames": None, "overlay": None, "ring": None, "at": []}
                rows.append(row)
            for dim, d in dims.items():
                w = f"extents {name!r} {dim}"
                sm = num(req(d, "sigma_mask", w), 3)
                for label, x in req(d, "at", w).items():
                    row["at"].append({"dim": text(dim), "label": text(label), "value": num(req(x, "value", w), 4),
                                      "sigma": num(req(x, "sigma", w), 3), "sigma_mask": sm,
                                      "sigma_mask_how": text(req(d, "sigma_mask_how", w)),
                                      "used": [int(f) for f in req(x, "frames_used", w)],
                                      "rejected": [{"frame": int(req(r, "frame", w)), "speed": num(req(r, "speed_px", w), 3),
                                                    "extent": num(req(r, "extent", w), 3), "why": text(req(r, "why", w))}
                                                   for r in req(x, "frames_rejected", w)],
                                      "how": text(req(x, "how", w))})
    return {"rows": rows} if rows else None


def depth_bindings_view(state: dict):
    """The scene's depth statements (rule C14): how many were given, each bound one with its node, role and what the
    scene used it for (by null: not used, with why), each unbound one with its reason; None when none was given."""
    db = req(state, "depth_bindings", "scene")
    given = req(db, "given", "depth_bindings")
    bound = [{"name": text(req(b, "name", "depth binding")), "node": text(req(b, "node", "depth binding")),
              "as": text(req(b, "as", "depth binding")), "end": text(b.get("end")) or None,
              "by": text(req(req(b, "used", "depth binding"), "by", "depth binding used")) or None,
              "why": text(req(b["used"], "why", "depth binding used"))} for b in req(db, "bound", "depth_bindings")]
    unbound = [{"name": text(req(b, "name", "unbound depth statement")), "reason": text(req(b, "reason", "unbound depth statement"))}
               for b in req(db, "unbound", "depth_bindings")]
    if not given:
        if bound or unbound:
            raise ValueError(f"{AT['where']}: depth_bindings bound / unbound without a given statement")
        return None
    return {"given": len(given), "names": len({text(g.get("name")) for g in given}), "bound": bound, "unbound": unbound}


def camera_one_view(state: dict):
    """The scene camera fused from stage A and the pendulums' joint f (rule camera.one); None when no pendulum votes."""
    co = req(state, "camera_one", "scene")
    if co is None:
        return None
    votes = [{"source": text(req(v, "source", "camera_one vote")), "object": text(req(v, "object", "camera_one vote")),
              "f": num(req(v, "f", "camera_one vote"), 4), "sigma": num(req(v, "sigma", "camera_one vote"), 3),
              "used": bool(req(v, "used", "camera_one vote")), "why": text(req(v, "why", "camera_one vote"))}
             for v in req(co, "votes", "camera_one")]
    if text(req(state, "f_source", "scene")) != "camera.one":
        raise ValueError(f"{AT['where']}: camera_one record with f_source {state.get('f_source')!r}")
    return {"f": num(req(co, "f", "camera_one"), 4), "sigma": num(req(co, "sigma", "camera_one"), 3),
            "how": text(req(co, "how", "camera_one")), "votes": votes}


def forward_near(name: str, o: dict):
    """The exported projection's near depth (forward.near): the fit's own nearest depth, not a fixed 0.3 m."""
    fw = req(o, "forward", f"scene object {name!r}")
    if fw is None:
        return None
    n = req(fw, "near", f"scene object {name!r} forward")
    if not isinstance(n, (int, float)) or not n > 0:
        raise ValueError(f"{AT['where']}: object {name!r} forward.near {n!r}")
    return num(n, 4)


SEEN_FROM_BELOW = "plan.overhead_seen_from_below"
SEEN_VALUES = {"horizon_row", "bottom_row", "margin_px", "camera_height", "n_frames", "n_above"}


def seen_from_below_view(x, name: str | None, where: str) -> dict:
    """Rule D34 plan.overhead_seen_from_below: an overhead structure's underside faces the camera only above the
    camera's eye level -- its box's bottom edge above the support plane's horizon line (median over its tracked
    frames). {object, pass (None: untestable), test, why, values}."""
    if not isinstance(x, dict) or set(x) != {"object", "pass", "test", "why", "values"} or x["pass"] not in (True, False, None) \
            or (name is not None and x["object"] != name):
        raise ValueError(f"{AT['where']}: {where} {SEEN_FROM_BELOW} {x!r}")
    v = x["values"]
    if x["pass"] is not None and (not isinstance(v, dict) or set(v) != SEEN_VALUES
                                  or (num(v["margin_px"], 3) > 0) != x["pass"] or not 0 <= v["n_above"] <= v["n_frames"]):
        raise ValueError(f"{AT['where']}: {where} {SEEN_FROM_BELOW} values {v!r} pass {x['pass']!r}")
    return {"object": text(x["object"]), "pass": x["pass"], "test": text(x["test"]), "why": text(x["why"]),
            "values": {k: num(a, 4) for k, a in v.items()} if isinstance(v, dict) else None}


def support_view(name: str, o: dict) -> dict:
    """An object's support plan (scene_fit.plan): mode, target, airborne_part, source; a rewritten plan names its rule
    and what it was; an overhead structure carries its D34 test, and a failing one is a static box "static_above"."""
    w = f"scene object {name!r} support"
    sp = req(o, "support", f"scene object {name!r}")
    need = {"mode", "target", "airborne_part"} | (set() if FMT.get("sceneprog") else {"source"})
    if not isinstance(sp, dict) or not need <= set(sp) \
            or not set(sp) <= {"mode", "target", "airborne_part", "source", "rel", "rule", "was", SEEN_FROM_BELOW} \
            or ("rule" in sp) != ("was" in sp):
        raise ValueError(f"{AT['where']}: {w} {sp!r}")
    out = {k: sp[k] for k in ("mode", "target", "airborne_part", "source", "rel") if k in sp}
    if "was" in sp:
        out |= {"rule": text(sp["rule"]), "was": {k: text(sp["was"].get(k)) for k in ("mode", "target")}}
    sb = sp.get(SEEN_FROM_BELOW)
    if sb is not None:
        out["seen_from_below"] = seen_from_below_view(sb, name, w)
    if (sp["mode"] == "static_above") != (sb is not None and sb["pass"] is False) or \
            (sp["mode"] == "static_above" and (sp.get("rule") != SEEN_FROM_BELOW or sp["was"].get("mode") != "overhead")) or \
            (sb is not None and sp["mode"] not in ("overhead", "static_above")):
        raise ValueError(f"{AT['where']}: {w} mode {sp['mode']!r} rule {sp.get('rule')!r} with {SEEN_FROM_BELOW} {sb!r}")
    return out


def seen_from_below_list(j: dict) -> list:
    """The scene's top-level list of D34 tests: the same records as the objects' plans carry."""
    lst = req(j, SEEN_FROM_BELOW, "scene")
    own = [o["support"][SEEN_FROM_BELOW] for o in req(j, "objects", "scene").values()
           if isinstance(o.get("support"), dict) and SEEN_FROM_BELOW in o["support"]]
    if not isinstance(lst, list) or sorted(map(repr, lst)) != sorted(map(repr, own)):
        raise ValueError(f"{AT['where']}: scene {SEEN_FROM_BELOW} {lst!r} differs from the objects' plans")
    return [seen_from_below_view(x, None, "scene") for x in lst]


def scene_view(state: dict, source_path: Path) -> dict:
    """A saved scene as the page draws it, without the per-question results (a shared scene must not show other
    questions' answers)."""
    j = state
    if req(j, "f", "scene") is None:
        raise ValueError(f"{source_path.relative_to(ROOT)}: scene has no focal length")
    pl = req(j, "plane", "scene")
    # u1 / u2 (in-plane axes) are written only for some planes
    plane = {k: vec(req(pl, k, "plane") if k == "normal" else pl.get(k)) for k in ("normal", "u1", "u2")} | \
        {"dist": num(req(pl, "dist", "plane"), 5)}
    objs = []
    for name, o in req(j, "objects", "scene").items():
        what = f"scene object {name!r}"
        vis = req(o, "visible", what)
        objs.append({"name": name, "shape": req(o, "shape", what),
                     "dims": {k: num(v) for k, v in req(o, "dims", what).items()},
                     "rms": num(req(o, "rms", what), 3), "p0": vec(req(o, "p0", what)), "v0": vec(req(o, "v0", what)),
                     "a": vec(req(o, "a", what)), "yaw": num(req(o, "yaw", what), 5), "t_on": num(req(o, "t_on", what), 5),
                     # where each dimension comes from: observed / prior / bound (not seen, anti-collapse floor)
                     "dim_source": req(o, "dim_source", what), "dim_bounds": dim_bounds(name, o),
                     "support": support_view(name, o) if FMT.get("anchored") else req(o, "support", what),
                     "hang": req(o, "hang", what), "mode": req(o, "mode", what),
                     "out_of_frame": req(o, "out_of_frame", what), "visibility": req(o, "visibility", what),
                     "overhead": bool(req(o, "overhead", what)), "events": motion_events(name, o)[:12],
                     "law": motion_law(name, o) if FMT["decide"] == "agree" else None,
                     # motion that is not a single parabola: sample the centre for the viewer
                     "path": sampled_path(o) if any(req(o, k, what) for k in FMT["paths"]) else None,
                     "visible": [[num(a, 5), num(b, 5)] for a, b in vis] if vis else None,
                     "pendulum": pendulum_view(req(o, "pendulum", what), what) if FMT.get("pendulum") else None}
                    | ({"segfit": segfit_view(name, o), "events_graph": events_view(name, o)} if FMT.get("selftest") else {})
                    | (impulse_view(name, o) if FMT.get("impulse") else {})
                    | ({"fit_object": fit_object_view(name, o)} if FMT.get("contact") else {})
                    | ({"size.full_extent": full_extent_view(req(o, "size.full_extent", what), what)} if FMT.get("span") else {})
                    | (agent_dims_view(name, o) | chain_view(name, o) if FMT.get("launch") else {})
                    | ({"role": object_role(name, o)} if FMT.get("classref") else {})
                    | ({"near": forward_near(name, o)} if FMT.get("rendered") else {}))
    return {"file": str(source_path.relative_to(ROOT)), "f": num(j["f"], 5), "plane": plane, "objects": objs,
            "scale": scale_view(j), "gravity": gravity_view(j),
            "main_support": req(j, "main_support", "scene"),
            "f_source": text(req(j, "f_source", "scene")) or None, "f_first": req(j, "f_first", "scene"),
            "f_first_clamp": clamp_view(req(j, "f_first_clamp", "scene")) if FMT.get("pendulum") else None,
            # an empty vote list is legitimate (no focal vote taken): written as null
            "focal_votes": [{"source": text(req(x, "source", "focal vote")), "value": num(req(x, "value", "focal vote"), 4),
                             "weight": num(req(x, "weight", "focal vote"), 3)}
                            for x in req(j, "focal_votes", "scene")] or None} | \
        ({"focal": focal_view(j), "horizon": horizon_view(j)} if FMT.get("horizon") else {}) | \
        ({"camera": camera_view(j), "support_plane": support_plane_view(j)} if FMT.get("camera") else {}) | \
        ({"camera_roll": camera_roll_view(j)} if FMT.get("roll") else {}) | \
        ({"camera_one": camera_one_view(j)} if FMT.get("static") else {}) | \
        ({"fusion": fusion_view(j)} if FMT.get("fusion") else {}) | \
        ({"incline": incline_view(j)} if FMT.get("impulse") else {}) | \
        ({"scale_contact": scale_contact_view(j), "main_plane": main_plane_view(j)} if FMT.get("contact") else {}) | \
        ({"depth_rows": depth_rows_view(j)} if FMT.get("interval") else {}) | \
        ({"plane_fusion": plane_fusion_view(j), "plane_sigma": plane_sigma_view(j)} if FMT.get("launch") else {}) | \
        ({"class_reference": class_reference_view(j, camera_scale_source(req(j, "camera", "scene")))}
         if FMT.get("classref") else {}) | \
        ({"references": references_view(j)} if FMT.get("sizeonly") else {}) | \
        ({"class_object": class_object_view(j), "contained": contained_view(j)} if FMT.get("contained") else {}) | \
        ({"soft": soft_view(j), "no_new_law": [no_new_law_view(x) for x in req(j, "unobserved.no_new_law", "scene")] or None}
         if FMT.get("rendered") else {}) | \
        ({"up": gravity_up_view(j)} if FMT.get("up") else {}) | \
        ({"seen_from_below": seen_from_below_list(j)} if FMT.get("anchored") else {})


SOFT_SOURCES = {"support_mask": set(), "main_support": {"n_mask_points", "n_contacts"},
                "stage_a_fit": {"objects", "own_only", "horizon", "stage_a"}}   # scene_fit.support_undetermined_soft
SOFT_KEYS = ("plane", "pass", "sigma_roll", "sigma_pitch", "anchors_from", "why")
SOFT_SIGMA_DEG = 90.0


def soft_view(state: dict) -> dict:
    """Rule C23 support.undetermined_soft: the support plane judged once, after stage A -- its roll / pitch sigma and
    where it came from; soft (pass false) when a sigma >= 90 deg: no hard surface, a chain's bounded phases against it
    become flight (the objects' softened rows). {pass, plane, sigma_roll, sigma_pitch, from {..}, why, softened
    [{object, t0, t1, rel, to, was, now}]}."""
    w = "support.undetermined_soft"
    x = req(state, w, "scene")
    if set(x) != {"name", *SOFT_KEYS} or x["name"] != w or x["pass"] not in (True, False, None) \
            or req(state, "plane", "scene").get(w) != x:
        raise ValueError(f"{AT['where']}: scene {w} {x!r} (or plane's copy differs)")
    af, pl = x["anchors_from"], x["plane"]
    if pl is None:
        if af is not None or x["pass"] is not None or x["sigma_roll"] is not None or x["sigma_pitch"] is not None:
            raise ValueError(f"{AT['where']}: scene {w} without a plane {x!r}")
    elif set(pl) != {"roll_deg", "pitch_deg", "dist"} or af is None or af.get("source") not in SOFT_SOURCES \
            or set(af) != {"source"} | SOFT_SOURCES[af["source"]]:
        raise ValueError(f"{AT['where']}: scene {w} plane {pl!r} anchors_from {af!r}")
    sig = [x["sigma_roll"], x["sigma_pitch"]]
    if x["pass"] is True and (None in sig or max(sig) >= SOFT_SIGMA_DEG) \
            or x["pass"] is None and pl is not None and None not in sig:
        raise ValueError(f"{AT['where']}: scene {w} pass {x['pass']!r} with sigma {sig}")
    softened = []
    for name, o in req(state, "objects", "scene").items():
        ol = (o.get("motion_law") or {}).get(w)
        if ol is None:
            continue
        wo = f"object {name!r} motion law {w}"
        if x["pass"] is not False or set(ol) != {*SOFT_KEYS, "softened", "events"} \
                or any(ol[k] != x[k] for k in SOFT_KEYS):
            raise ValueError(f"{AT['where']}: {wo} differs from the scene's (pass {x['pass']!r})")
        for r in ol["softened"]:
            if set(r) != {"t0", "t1", "rel", "to", "was", "now"}:
                raise ValueError(f"{AT['where']}: {wo} softened {r!r}")
            softened.append({"object": text(name), "t0": num(r["t0"], 4), "t1": num(r["t1"], 4), "rel": text(r["rel"]),
                             "to": text(r["to"]) or None, "was": text(r["was"]), "now": text(r["now"]),
                             "events": text(ol["events"])})
    fr = None
    if af is not None:
        fr = {"source": af["source"]} | {k: af[k] for k in SOFT_SOURCES[af["source"]] if k != "objects"} \
            | ({"objects": [[text(k), int(n)] for k, n in af["objects"].items()]} if "objects" in af else {})
    return {"pass": x["pass"], "plane": None if pl is None else {k: num(v, 4) for k, v in pl.items()},
            "sigma_roll": num(sig[0], 3), "sigma_pitch": num(sig[1], 3), "from": fr, "why": text(x["why"]),
            "softened": softened}


NO_NEW_LAW_DIRECTIONS = ("forward", "backward", None)


def no_new_law_view(x: dict) -> dict:
    """Rule B31 unobserved.no_new_law: an unobserved phase is no state and no event for the fit -- the state edge
    before it continues through it (forward; backward at the clip's start; none: a gap), and its frames are absent rows:
    {object, phase [t0, t1], why, direction, continued [{rel, to, was, now}], dropped {frames, n}, note}."""
    w = "scene unobserved.no_new_law"
    keys = {"rule", "object", "phase", "why", "direction", "continued", "rows_dropped", "note"}
    if set(x) != keys or x["rule"] != "unobserved.no_new_law" or x["direction"] not in NO_NEW_LAW_DIRECTIONS \
            or len(x["phase"]) != 2 or set(x["rows_dropped"]) != {"frames", "n"} \
            or (x["direction"] is None) != (not x["continued"]):
        raise ValueError(f"{AT['where']}: {w} {x!r}")
    sp = lambda v: [num(t, 4) if t is not None else None for t in v]   # noqa: E731
    cont = []
    for c in x["continued"]:
        if set(c) != {"rel", "to", "was", "now"}:
            raise ValueError(f"{AT['where']}: {w} continued {c!r}")
        cont.append({"rel": text(c["rel"]), "to": text(c["to"]) or None, "was": sp(c["was"]), "now": sp(c["now"])})
    rd = x["rows_dropped"]
    fr = [[int(a), int(b)] for a, b in rd["frames"]]
    if int(rd["n"]) != sum(b - a + 1 for a, b in fr):
        raise ValueError(f"{AT['where']}: {w} rows_dropped n {rd['n']} vs frames {fr}")
    return {"object": text(x["object"]), "phase": sp(x["phase"]), "why": text(x["why"]), "direction": x["direction"],
            "continued": cont, "dropped": {"frames": fr, "n": int(rd["n"])}, "note": text(x["note"]) or None}


CLASS_OBJECT_KEYS = {"scope", "scope_why", "class", "instances", "given_size", "chosen"}
CLASS_INSTANCE_KEYS = {"name", "tracked", "untracked", "role", "static", "static_why", "read"}
CLASS_READ_KEYS = {"mode", "value", "sigma", "extent", "n_frames", "how"}
PRIOR_SCOPES = ("instance", "class", "unbound", None)


def class_object_view(state: dict):
    """Rule C16 prior.class_object: a size stated for a class is the given size of every instance; each instance's
    track, role, static flag and its read (s: the scale from its extent, f: its size-prior focal vote), the chosen one.
    None unless the prior's scope is "class" (then class / instances / given size / chosen are empty)."""
    w = "prior.class_object"
    c = req(state, w, "scene")
    if set(c) != CLASS_OBJECT_KEYS or c["scope"] not in PRIOR_SCOPES:
        raise ValueError(f"{AT['where']}: {w} keys {sorted(c)} scope {c.get('scope')!r}")
    cp = req(state, "class_prior", "scene")
    if c["scope"] != "class":
        if c["class"] is not None or c["instances"] or c["given_size"] is not None or c["chosen"] is not None or \
                (cp is not None and cp.get("scope") == "class"):
            raise ValueError(f"{AT['where']}: {w} scope {c['scope']!r} with a class / instances / given size / chosen")
        return None
    names = [text(req(i, "name", f"{w} instance")) for i in c["instances"]]
    if not names or text(c["chosen"]) not in names or cp is None or cp.get("scope") != "class" or cp.get("class") != c["class"]:
        raise ValueError(f"{AT['where']}: {w} chosen {c['chosen']!r} of {names}, class_prior {cp!r}")
    gs = req(c, "given_size", w)
    rows = []
    for i in c["instances"]:
        wi = f"{w} instance {i.get('name')!r}"
        r = req(i, "read", wi)
        if set(i) != CLASS_INSTANCE_KEYS or (r is not None and set(r) != CLASS_READ_KEYS) or (i["tracked"] == (i["untracked"] is not None)):
            raise ValueError(f"{AT['where']}: {wi} keys {sorted(i)} read {sorted(r or {})} tracked {i.get('tracked')!r}")
        if r is not None and r["mode"] not in ("s", "f"):
            raise ValueError(f"{AT['where']}: {wi} read mode {r['mode']!r}")
        rows.append({"name": text(i["name"]), "tracked": bool(i["tracked"]), "untracked": text(i["untracked"]) or None,
                     "role": text(i["role"]), "static": i["static"], "static_why": text(i["static_why"]),
                     "read": None if r is None else {
                         "mode": r["mode"], "value": num(r["value"]), "sigma": num(r["sigma"], 3), "extent": num(r["extent"]),
                         "n": r["n_frames"], "how": text(r["how"])}})
    return {"scope_why": text(c["scope_why"]), "class": text(c["class"]), "instances": rows, "chosen": text(c["chosen"]),
            "given": {"quantity": text(req(gs, "quantity", f"{w} given_size")), "value": num(req(gs, "value_si", f"{w} given_size")),
                      "sigma": num(req(gs, "sigma", f"{w} given_size"), 3)}}


CONTAINED_KEYS = {"rule", "pass", "why", "edges"}
CONTAINED_EDGE_KEYS = {"a", "b", "span", "source", "containment", "plan", "rows_dense", "z_a", "z_b", "sigma", "pass",
                       "testable", "ids", "rows", "why"}
CONTAINED_ROW_KEYS = {"t", "z_a", "z_b", "sigma", "sigma_container", "sigma_pred", "pass"}


def contained_view(state: dict):
    """Rule A13 contained.same_depth: every "A in B" edge's plan (contained), dense rows taken from the container, the
    depth of A against B at the span's first / middle / last shared frame; None without an "in" edge."""
    w = "contained.same_depth"
    c = req(state, w, "scene")
    if set(c) != CONTAINED_KEYS or c["rule"] != w:
        raise ValueError(f"{AT['where']}: {w} keys {sorted(c)}")
    if not c["edges"]:
        if c["pass"] is not None:
            raise ValueError(f"{AT['where']}: {w} pass {c['pass']!r} without an edge")
        return None
    cons = {text(x.get("id")): x for x in req(state, "constraints", "scene") if x.get("kind") == "contained"}
    edges = []
    for e in c["edges"]:
        we = f"{w} edge {e.get('a')!r} in {e.get('b')!r}"
        if set(e) != CONTAINED_EDGE_KEYS:
            raise ValueError(f"{AT['where']}: {we} keys {sorted(e)}")
        plan, ct = e["plan"], e["containment"]
        # the rule rewrites only a plan "support" on B (scene_fit.contained_plans); any other plan is kept as it was
        if req(plan, "mode", f"{we} plan") == "contained" and text(req(plan, "target", f"{we} plan")) != text(e["b"]):
            raise ValueError(f"{AT['where']}: {we} plan {plan!r}")
        if any(i not in cons for i in e["ids"]) or len(e["ids"]) != len(e["rows"]):
            raise ValueError(f"{AT['where']}: {we} ids {e['ids']} not the scene's contained constraints {sorted(cons)}")
        rows = []
        for r in e["rows"]:
            if set(r) != CONTAINED_ROW_KEYS:
                raise ValueError(f"{AT['where']}: {we} row keys {sorted(r)}")
            rows.append({k: (num(r[k], 4) if k != "pass" else r[k]) for k in ("t", "z_a", "z_b", "sigma", "sigma_container", "sigma_pred", "pass")})
        rd = e["rows_dense"]
        edges.append({"a": text(e["a"]), "b": text(e["b"]), "span": [num(x, 4) for x in e["span"]], "source": text(e["source"]),
                      "containment": None if ct is None else {"verdict": text(req(ct, "verdict", we)), "level": text(req(ct, "level", we)),
                                                              "inside_frac": num(req(ct, "inside_frac", we), 3),
                                                              "bound": num(req(ct, "bound", we), 3), "frames": req(ct, "frames", we)},
                      "rows_dense": None if rd is None else {"n_a": req(rd, "n_a", we), "n_b": req(rd, "n_b", we)},
                      "z_a": num(e["z_a"], 4), "z_b": num(e["z_b"], 4), "sigma": num(e["sigma"], 3),
                      "pass": e["pass"], "testable": bool(e["testable"]), "rows": rows, "why": text(e["why"])})
    return {"pass": c["pass"], "why": text(c["why"]), "edges": edges}


def object_role(name: str, o: dict) -> str:
    """A scene object's role (rule C15): object / structure / reference; a reference object (a class reference the
    texts do not name) has its class's body dimensions, dim_source "class" for every dim, and a size_reference.
    With FMT "sizeonly" (rule reference.size_only) a reference is never a scene object: it is in scene["references"]."""
    w = f"scene object {name!r}"
    role, sr = text(req(o, "role", w)), req(o, "size_reference", w)
    if role not in OBJECT_ROLES or (FMT.get("sizeonly") and role == "reference"):
        raise ValueError(f"{AT['where']}: {w} role {role!r}")
    if role == "reference" and (sr is None or any(v != "class" for v in req(o, "dim_source", w).values())):
        raise ValueError(f"{AT['where']}: reference {w} dim_source {o['dim_source']} size_reference {sr is not None}")
    return role


REFERENCE_KEYS = ("role", "size_only", "size_reference", "read", "state", "why", "reason", "size.full_extent",
                  "dims.agent_height")
REFERENCE_READ_KEYS = ("frames", "n_frames", "frame_set", "excluded", "n_depth", "headings", "rms_box_px", "dims",
                       "dim_source", "shape", "scale")
REFERENCE_EXCLUDED = ("size.full_extent", "border", "extent.static")
REFERENCE_SKIPS = ("read.frames", "support.none", "track.none", "focal.undetermined")


def references_view(state: dict) -> list:
    """The scene's reference objects (rule reference.size_only): each one's class, its size read -- the frames read of
    its track's frame set and the ones left out by size.full_extent / border / extent.static, depth samples, headings
    (own path / shared yaw), box rms, metric dims with their source -- or, without a read, the skip reason; its posed
    state as one static pose per read frame: "poses" [[t, x, y, z, hx, hy, hz, nx, ny, nz]] (centre, heading, up;
    pipeline.scene_state3d.position / orientation); its agent extent and height checks when it is an agent."""
    from pipeline.scene_state3d import orientation, position
    out = []
    for name, r in req(state, "references", "scene").items():
        w = f"reference {name!r}"
        if set(r) != set(REFERENCE_KEYS) or r["role"] != "reference" or r["size_only"] is not True \
                or not isinstance(r["size_reference"], dict):
            raise ValueError(f"{AT['where']}: {w} keys {sorted(r)} role {r.get('role')!r} size_only {r.get('size_only')!r}")
        sr = r["size_reference"]
        row = {"name": text(name), "class": text(req(sr, "class", f"{w} size_reference")),
               "dim": text(req(sr, "dim", f"{w} size_reference")), "value": num(req(sr, "value", f"{w} size_reference"), 4),
               "sigma": num(req(sr, "sigma", f"{w} size_reference"), 4)}
        rd, st = r["read"], r["state"]
        if rd is None:
            if text(r["reason"]) not in REFERENCE_SKIPS or not text(r["why"]) or st is not None:
                raise ValueError(f"{AT['where']}: {w} without a read: reason {r['reason']!r} why {r['why']!r} state {st is not None}")
            out.append(row | {"read": None, "reason": text(r["reason"]), "why": text(r["why"]), "poses": None,
                              "full_extent": None, "agent_height": None})
            continue
        if r["reason"] is not None or r["why"] is not None or set(rd) != set(REFERENCE_READ_KEYS) or not isinstance(st, dict):
            raise ValueError(f"{AT['where']}: {w} read keys {sorted(rd)} reason {r['reason']!r} state {st is not None}")
        ex = rd["excluded"]
        if not set(ex) <= set(REFERENCE_EXCLUDED):
            raise ValueError(f"{AT['where']}: {w} read excluded {sorted(ex)}")
        hd = rd["headings"]
        if set(hd) != {"path", "shared_yaw"} or int(hd["path"]) + int(hd["shared_yaw"]) != int(rd["n_frames"]):
            raise ValueError(f"{AT['where']}: {w} headings {hd} for {rd['n_frames']} frames")
        ds = rd["dim_source"]
        if set(ds) != set(rd["dims"]) or any(v not in DIM_SOURCES for v in ds.values()):
            raise ValueError(f"{AT['where']}: {w} dim_source {ds} dims {sorted(rd['dims'])}")
        times = req(req(st, "fit_model", f"{w} state"), "times", f"{w} state fit_model")
        if len(times) != int(rd["n_frames"]) or text(req(st, "mode", f"{w} state")) != "size_read":
            raise ValueError(f"{AT['where']}: {w} state {len(times)} times, mode {st.get('mode')!r}, read {rd['n_frames']} frames")
        poses = []
        for t in times:
            R = orientation(st, float(t))
            poses.append([num(t, 4)] + [num(float(x), 5) for x in position(st, float(t))]
                         + [num(float(x), 5) for x in R[0]] + [num(float(x), 5) for x in R[2]])
        ah = r["dims.agent_height"]
        if ah is not None and req(ah, "name", f"{w} dims.agent_height") != "dims.agent_height":
            raise ValueError(f"{AT['where']}: {w} dims.agent_height name {ah.get('name')!r}")
        out.append(row | {
            "read": {"frames": text(rd["frames"]), "n": int(rd["n_frames"]), "frame_set": text(rd["frame_set"]),
                     "excluded": {k: text(v) for k, v in ex.items()}, "n_depth": int(rd["n_depth"]),
                     "path": int(hd["path"]), "shared_yaw": int(hd["shared_yaw"]), "rms": num(rd["rms_box_px"], 3),
                     "dims": {k: num(v, 4) for k, v in rd["dims"].items()}, "dim_source": dict(ds),
                     "shape": text(rd["shape"]), "scale": num(rd["scale"], 5)},
            "reason": None, "why": None, "shape": text(req(st, "shape", f"{w} state")),
            "dims": {k: num(v, 5) for k, v in req(st, "dims", f"{w} state").items()}, "poses": poses,
            "full_extent": full_extent_view(r["size.full_extent"], w),
            "agent_height": None if ah is None else {
                "pass": req(ah, "pass", w), "ray": num(req(ah, "ray_angle_deg", w), 3),
                "threshold": num(req(ah, "threshold_deg", w), 3), "pitch": num(req(ah, "pitch_deg", w), 3),
                "H_source": text(req(ah, "H_source", w)), "reason": text(req(ah, "reason", w))}})
    return out


IMPULSE_VERDICTS = ("kick", "agent.unobserved", "chain ends", "no_law")   # or "rejected: <why>"
DIM_SOURCES = ("observed", "prior", "bound", "class")   # pipeline.critic.DIM_SOURCES (rule dims.source_display)
CHAIN_KINDS = ("rest", "bounded", "flight", "incline", "uniform", "driven")   # pipeline.fit3d.physics_states


def agent_dims_view(name: str, o: dict) -> dict:
    """An articulated agent's dimension checks: its lateral dims never observed from the box (rule dims.agent_lateral:
    {"pass", "sources" {dim: source}, "box_px", "box_m", "fits_box", "reason"}) and its height observed only when the
    ray meets the support plane at most at the threshold angle (rule dims.agent_height: {"pass", "pitch", "ray",
    "threshold", "H_source", "reason"}); both None for any other object. Every dim_source is a known source."""
    what = f"scene object {name!r}"
    ds = req(o, "dim_source", what)
    if any(v not in DIM_SOURCES for v in ds.values()):
        raise ValueError(f"{AT['where']}: {what} dim_source {ds}")
    la, ah = req(o, "dims.agent_lateral", what), req(o, "dims.agent_height", what)
    # an agent fitted on no support plane (scene_fit.set_agent_height: plane None) has no height decision
    no_plane = la is not None and ah is None and req(o, "plane", what) is None
    if (la is None) != (ah is None) and not no_plane:
        raise ValueError(f"{AT['where']}: {what} has only one of dims.agent_lateral / dims.agent_height")
    if la is None:
        return {"agent_lateral": None, "agent_height": None}
    w = f"{what} dims.agent_lateral"
    if req(la, "name", w) != "dims.agent_lateral" or req(la, "pass", w) not in (True, None):
        raise ValueError(f"{AT['where']}: {w} name {la.get('name')!r} pass {la.get('pass')!r}")
    src = {k[:-len("_source")]: text(v) for k, v in la.items() if k.endswith("_source")}
    if not src or any(ds.get(k) != v or v not in ("class", "bound", "prior") for k, v in src.items()):
        raise ValueError(f"{AT['where']}: {w} sources {src} differ from dim_source {ds}")
    lat = {"pass": la["pass"], "sources": src, "box_px": num(req(la, "box_w_px_median", w), 4),
           "box_m": num(req(la, "box_w_m", w), 4), "fits_box": req(la, "fits_box", w), "reason": text(req(la, "reason", w))}
    wh = f"{what} dims.agent_height"
    if no_plane:
        return {"agent_lateral": lat, "agent_height": None}
    if req(ah, "name", wh) != "dims.agent_height" or req(ah, "pass", wh) not in (True, False, None):
        raise ValueError(f"{AT['where']}: {wh} name {ah.get('name')!r} pass {ah.get('pass')!r}")
    hs = text(req(ah, "H_source", wh))
    if hs not in DIM_SOURCES or (ah["pass"] is False and hs != ds.get("H")):
        raise ValueError(f"{AT['where']}: {wh} H_source {hs!r} with pass {ah['pass']!r}, dim_source H {ds.get('H')!r}")
    return {"agent_lateral": lat,
            "agent_height": {"pass": ah["pass"], "pitch": num(req(ah, "pitch_deg", wh), 3),
                             "ray": num(req(ah, "ray_angle_deg", wh), 3), "threshold": num(req(ah, "threshold_deg", wh), 3),
                             "H_source": hs, "reason": text(req(ah, "reason", wh))}}


IMPULSE_SPAN_KEYS = ("name", "onset", "verdict", "ramp_end", "first_frame", "t_stop", "n_driven_frames", "speeds", "how",
                     "untestable")


def chain_view(name: str, o: dict) -> dict:
    """The physics chain's phases [[t start, kind, kick]] (kind "driven": no law, rule law.impulse_span) and its driven
    span record "impulse_span" {onset, verdict, ramp_end, first_frame, t_stop, n_driven, speeds, how, untestable,
    applied, free_piece} (None without one; the motion law's copy and the onset event's must agree with it)."""
    ph = req(o, "physics", f"scene object {name!r}")
    if ph is not None and "rejected" in ph and fitted_physics(name, o) is None:
        # a rejected chain: its reason (the plan's phases and kick span were not fitted)
        return {"chain": None, "impulse_span": None, "gate": None, "chain_rejected": text(ph["rejected"])}
    if ph is None:
        return {"chain": None, "impulse_span": None} | ({"gate": None} if FMT.get("classref") else {})
    w = f"object {name!r} physics"
    gate = {"gate": gate_view(req(ph, "gate", w), f"{w} gate")} if FMT.get("classref") else {}
    rej = {"chain_rejected": text(ph["rejected"])} if FMT.get("rendered") and "rejected" in ph else {}
    phases = []
    for t, k, kick in req(ph, "phases", w):
        if k not in CHAIN_KINDS or not isinstance(kick, bool):
            raise ValueError(f"{AT['where']}: {w} phase {k!r} kick {kick!r}")
        phases.append([num(t, 4), k, kick])
    sp = ph.get("impulse_span")
    driven = any(k == "driven" for _, k, _ in phases)
    if sp is None:
        if driven or "impulse_span" in ph:
            raise ValueError(f"{AT['where']}: {w} has a driven phase without impulse_span")
        return {"chain": phases, "impulse_span": None} | gate | rej
    ws = f"{w} impulse_span"
    # rule record.one_shape (classref): the record always has "why" (null when applied) and is the motion law's own
    # an untestable span with no frame-pair speeds has no kick_ramp record (scene_fit.impulse_span returns early)
    no_ramp = FMT.get("rendered") and sp.get("untestable") is True and sp.get("speeds") == [] and "kick_ramp" not in sp
    extra = {"applied", "free_piece"} | ({"why"} if FMT.get("classref") else set()) \
        | ({"kick_ramp"} if FMT.get("unobserved") and not no_ramp else set())
    if req(sp, "name", ws) != "law.impulse_span" or set(sp) != set(IMPULSE_SPAN_KEYS) | extra:
        raise ValueError(f"{AT['where']}: {ws} name {sp.get('name')!r} keys {sorted(sp)}")
    ml = (req(o, "motion_law", f"scene object {name!r}") or {}).get("impulse_span")
    if ml is None or any(ml.get(k) != sp[k] for k in IMPULSE_SPAN_KEYS) or set(ml) - set(sp) \
            or (FMT.get("classref") and ml != sp):
        raise ValueError(f"{AT['where']}: {ws} differs from its motion law's")
    if FMT.get("classref") and (sp["why"] is None) != bool(sp["applied"]):
        raise ValueError(f"{AT['where']}: {ws} applied {sp['applied']!r} with why {sp['why']!r}")
    if bool(sp["applied"]) != driven:
        raise ValueError(f"{AT['where']}: {ws} applied {sp['applied']!r} with chain phases {[k for _, k, _ in phases]}")
    ev = [e for e in req(ph, "events_graph", w) if "impulse_span" in e]
    for e in ev:
        x = e["impulse_span"]
        if any(x.get(k) != sp[k] for k in ("onset", "ramp_end", "n_driven_frames", "how")):
            raise ValueError(f"{AT['where']}: {w} events_graph {e.get('event')!r} impulse_span differs from impulse_span")
    if sp["applied"] and len(ev) != 1:
        raise ValueError(f"{AT['where']}: {w} an applied driven span on {len(ev)} onset events")
    return {"chain": phases,
            "impulse_span": {"onset": num(sp["onset"], 5), "verdict": text(sp["verdict"]), "ramp_end": num(sp["ramp_end"], 5),
                             "first_frame": num(sp["first_frame"], 5), "t_stop": num(sp["t_stop"], 5),
                             "n_driven": None if no_ramp and sp["n_driven_frames"] is None else int(sp["n_driven_frames"]), "speeds": [[num(t, 5), num(v, 4)] for t, v in sp["speeds"]],
                             "how": text(sp["how"]), "untestable": sp["untestable"], "applied": bool(sp["applied"]),
                             "free_piece": bool(sp["free_piece"]), "why": text(sp.get("why")) or None}
                       | ({"kick_ramp": kick_ramp_view(sp["kick_ramp"], ws) if sp.get("kick_ramp") is not None else None}
                          if FMT.get("unobserved") else {})} | gate | rej


GATE_KEYS = ("name", "pass", "rms_chain", "rms_kinematic", "frames", "bound", "rms_chain_law", "frames_law",
             "rms_kinematic_all", "frames_kinematic_all", "reason")


def gate_view(g: dict, w: str) -> dict:
    """The chain-vs-kinematics gate on the same frames (rule physics.gate_frames): chain rms <= 1.5 x kinematic + 2 px
    on the chain's law frames where the kinematic fit projects a box; each fit's own rms over its own frames beside."""
    if set(g) != set(GATE_KEYS) or g["name"] != "physics.gate" or g["pass"] not in (True, False):
        raise ValueError(f"{AT['where']}: {w} keys {sorted(g)} name {g.get('name')!r} pass {g.get('pass')!r}")
    if (float(g["rms_chain"]) <= float(g["bound"])) != g["pass"]:
        raise ValueError(f"{AT['where']}: {w} pass {g['pass']} with rms {g['rms_chain']} vs bound {g['bound']}")
    return {"pass": g["pass"], "rms_chain": num(g["rms_chain"], 3), "rms_kinematic": num(g["rms_kinematic"], 3),
            "frames": int(g["frames"]), "bound": num(g["bound"], 3), "rms_chain_law": num(g["rms_chain_law"], 3),
            "frames_law": int(g["frames_law"]), "rms_kinematic_all": num(g["rms_kinematic_all"], 3),
            "frames_kinematic_all": int(g["frames_kinematic_all"]), "reason": text(g["reason"])}


def impulse_view(name: str, o: dict) -> dict:
    """The physics record's impulse decisions (rule law.impulse), its refined onsets (kinematic.onset) and incline laws
    (support.incline): {"impulse": [{t, event, agent, source, verdict, why, jump, sigma, jump_how}] | None,
    "onset": [{phase, t, sigma, how}] | None, "incline_law": [{..}] | None}; all None without a physics record."""
    ph = fitted_physics(name, o)
    if ph is None:
        return {"impulse": None, "onset": None, "incline_law": None}
    w = f"object {name!r} physics"
    imp = req(ph, "impulse", w)
    if (o.get("motion_law") or {}).get("impulse", imp) != imp:
        raise ValueError(f"{AT['where']}: {w} impulse differs from its motion law's")
    rows = []
    for x in imp:
        v = text(req(x, "verdict", f"{w} impulse"))
        if v not in IMPULSE_VERDICTS and not v.startswith("rejected: "):
            raise ValueError(f"{AT['where']}: {w} impulse verdict {v!r}")
        rows.append({"t": num(req(x, "t", f"{w} impulse"), 4), "event": text(req(x, "event", f"{w} impulse")),
                     "agent": text(req(x, "agent", f"{w} impulse")) or None,
                     "source": text(req(x, "source", f"{w} impulse")) or None, "verdict": v,
                     "why": text(req(x, "why", f"{w} impulse")), "jump": num(x.get("jump_px_s"), 3),
                     "sigma": num(x.get("sigma_px_s"), 3), "jump_how": text(x.get("jump_how")) or None})
    onset = [{"phase": int(k), "t": num(req(v, "t", f"{w} onset"), 5), "sigma": num(req(v, "sigma", f"{w} onset"), 3),
              "how": text(req(v, "how", f"{w} onset"))} for k, v in req(ph, "onset", w).items()]
    il = []
    for x in req(ph, "incline_law", w):
        wl = f"{w} incline_law"
        il.append({"phase": int(req(x, "phase", wl)), "support": text(req(x, "support", wl)),
                   "t_on": num(req(x, "t_on", wl), 5), "sigma_t_on": num(req(x, "sigma_t_on", wl), 3),
                   "accel": num(req(x, "accel", wl), 4), "sigma_accel": num(req(x, "sigma_accel", wl), 3),
                   "accel_source": text(req(x, "accel_source", wl)), "accel_geometry": num(req(x, "accel_geometry", wl), 4),
                   "kappa": num(req(x, "kappa", wl), 4), "shape": text(req(x, "shape", wl)),
                   "theta": num(req(x, "theta_deg", wl), 4), "law": text(req(x, "law", wl))})
    return {"impulse": rows or None, "onset": onset or None, "incline_law": il or None}


def incline_view(state: dict):
    """The measured slope of every inclined support (rule support.incline): [{support, method, theta, sigma, sigma_f,
    length, inclined, up {level, angle, sigma, source, why}, why}] | None."""
    out = []
    for name, x in req(state, "support_incline", "scene").items():
        w = f"support_incline {name!r}"
        up = req(x, "up", w)
        out.append({"support": text(name), "method": text(req(x, "method", w)), "theta": num(req(x, "theta_deg", w), 4),
                    "sigma": num(req(x, "sigma_deg", w), 3), "sigma_f": num(x.get("sigma_f_deg"), 3),
                    "sigma_f_how": text(x.get("sigma_f_how")) or None, "length": num(x.get("length_m"), 4),
                    "inclined": bool(req(x, "inclined", w)),
                    "up": {"level": bool(req(up, "level", f"{w} up")), "angle": num(req(up, "angle_deg", f"{w} up"), 3),
                           "sigma": num(req(up, "sigma_deg", f"{w} up"), 3), "source": text(req(up, "source", f"{w} up")),
                           "why": text(req(up, "why", f"{w} up"))},
                    "why": text(req(x, "why", w))})
    return out or None


ZOOM_VOTES = {True: "visible", False: "not visible", None: "unknown"}   # a zoom vote: True / False / None (unknown)


def presence_view(tag: str, unit: str):
    """The unit's presence verdicts (outputs/objects_<tag>_presence_v4/<unit>.json, rules decisions.no_conf,
    presence.split_vote): [{name, present (visible / partial / absent / null), rule, reason (null: the record has
    none), votes, zoom (visible / not visible / unknown per zoom vote) | null, errors [{where, type, msg}] | null}]."""
    j = read_json(ROOT / "outputs" / f"objects_{tag}_presence_v4" / f"{unit}.json", "presence record")
    out = []
    for name, v in j.items():
        w = f"presence of {name!r}"
        if any("conf" in k for k in v):
            raise ValueError(f"{AT['where']}: {w} has a confidence field; format 4 decides by votes")
        rule = text(req(v, "rule", w))
        if not rule.startswith("presence."):
            raise ValueError(f"{AT['where']}: {w} rule {rule!r}")
        zv = v.get("zoom_votes")
        if zv is not None and any(x not in ZOOM_VOTES for x in zv):
            raise ValueError(f"{AT['where']}: {w} zoom votes {zv}")
        errs = [{k: text(req(e, k, f"{w} error")) for k in ("where", "type", "msg")}
                for e in (v.get("zoom_errors") or []) + (v.get("errors") or [])]
        out.append({"name": text(name), "present": req(v, "present", w), "rule": rule,
                    "reason": text(req(v, "reason", w)) if FMT.get("interval") else text(v["reason"]) if "reason" in v else None,
                    "votes": [text(x) for x in req(v, "votes", w)],
                    "zoom": [ZOOM_VOTES[x] for x in zv] if zv else None, "errors": errs or None})
    return out or None


FUSION_VERDICTS = ("used", "outlier", "disagree", "no sigma", "excluded")   # excluded: rule fit.object
KEPT_OUTLIERS = "outlier(s) of the scene's own sources kept:"   # pipeline.scene_fit focal_fusion_class


def fusion_view(state: dict) -> dict:
    """The scene's focal votes fused by independent source (rule focal.fusion): one row per source (object x kind)
    with its fused f +- sigma, votes, whether it is one of camera.f's own sources, verdict and why; the fused f +- sigma,
    the number of sources, pass (None: untestable) and reason, and camera.f the fits keep."""
    fu = req(req(state, "focal", "scene"), "fusion", "focal")
    w = "focal.fusion"
    rows = []
    for x in req(fu, "sources", w):
        v = text(req(x, "verdict", f"{w} source"))
        # rule scale.class_reference: a class vote cross-checks the stated camera, never part of the fused f
        cross = FMT.get("classref") and v == "cross-check"
        if v not in FUSION_VERDICTS and not cross:
            raise ValueError(f"{AT['where']}: {w} verdict {v!r}")
        if cross and (text(req(x, "kind", f"{w} source")) != "class reference" or req(x, "pass", f"{w} source") not in (True, False)):
            raise ValueError(f"{AT['where']}: {w} cross-check kind {x.get('kind')!r} pass {x.get('pass')!r}")
        b = x.get("bounded")
        rows.append({"object": text(req(x, "object", f"{w} source")), "kind": text(req(x, "kind", f"{w} source")),
                     "f": num(req(x, "f", f"{w} source"), 4), "sigma": num(req(x, "sigma", f"{w} source"), 3),
                     "n": int(req(x, "n_votes", f"{w} source")), "camera": bool(req(x, "camera", f"{w} source")),
                     "how": text(req(x, "how", f"{w} source")), "verdict": v, "why": text(req(x, "why", f"{w} source")) or None,
                     "bounded": [num(a, 4) for a in b] if b else None}
                    | ({"pass": bool(x["pass"])} if cross else {}))
    n = int(req(fu, "n_sources", w))
    # rule class_vote.regular: an outlier of the scene's own fusion stays an outlier and is not one of the sources the
    # class votes join; the record names those only in its reason ("outlier(s) of the scene's own sources kept: ...")
    reason = text(req(fu, "reason", w))
    kept_txt = reason.split(KEPT_OUTLIERS, 1)[1] if FMT.get("sizeonly") and KEPT_OUTLIERS in reason else ""
    for r in rows:
        r["kept"] = r["verdict"] == "outlier" and r["camera"] and f"{r['kind']} of '{r['object']}'" in kept_txt
    if kept_txt and not any(r["kept"] for r in rows):
        raise ValueError(f"{AT['where']}: {w} reason names kept outliers, no outlier row matches: {kept_txt!r}")
    if n != len([r for r in rows if r["verdict"] not in ("no sigma", "excluded", "cross-check") and not r["kept"]]):
        raise ValueError(f"{AT['where']}: {w} n_sources {n} with {len(rows)} sources")
    ok = req(fu, "pass", w)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} pass = {ok!r}")
    return {"sources": rows, "n": n, "f": num(fu.get("f"), 4), "sigma": num(fu.get("sigma"), 3),
            "camera_f": num(req(fu, "camera_f", w), 4), "camera_sigma": num(req(fu, "camera_sigma", w), 3),
            "pass": ok, "reason": text(req(fu, "reason", w))}


def camera_view(state: dict) -> dict:
    """The scene's one camera (rule C3a): f +- sigma, how sigma was found, its source, and each focal.consistent
    cross-check of a focal length one prior or one pendulum implies."""
    cam = req(state, "camera", "scene")
    rows = []
    for x in req(req(state, "focal", "scene"), "consistent", "focal"):
        w = "focal.consistent"
        fo, f = req(x, "f_obj", w), req(x, "f", w)
        cs = x.get("class_size")
        ind = req(x, "independent", w) if FMT.get("static") else None
        if FMT.get("static") and not isinstance(ind, bool):
            raise ValueError(f"{AT['where']}: focal.consistent independent = {ind!r}")
        if ind is False and x.get("pass") is not None:
            raise ValueError(f"{AT['where']}: a vote that is not independent has a verdict")
        xe = x.get("extent") if FMT.get("static") else None
        ext = None if xe is None else {"value": num(req(xe, "value", w), 4), "sigma": num(req(xe, "sigma", w), 3),
                                       "used": [int(f) for f in req(xe, "frames_used", w)],
                                       "rejected": len(req(xe, "frames_rejected", w))}
        rows.append({"independent": ind, "extent": ext} | {"source": text(req(x, "source", w)), "f_obj": num(fo, 4), "sigma_obj": num(req(x, "sigma_obj", w), 3),
                     "f": num(f, 4), "sigma": num(req(x, "sigma", w), 3), "bound": num(x.get("bound"), 3),
                     "diff": num(abs(float(fo) - float(f)), 3) if fo is not None and f is not None else None,
                     "pass": req(x, "pass", w), "reason": text(req(x, "reason", w)), "how": text(req(x, "how", w)),
                     "class_size": {"pass": bool(req(cs, "pass", w)), "reason": text(req(cs, "reason", w))} if cs else None})
    out = {"f": num(req(cam, "f", "camera"), 4), "sigma": num(req(cam, "f_sigma", "camera"), 3),
           "how": text(req(cam, "f_sigma_how", "camera")), "source": text(req(state, "f_source", "scene")) or None,
           "consistent": rows}
    if FMT.get("classref"):
        out["scale_source"] = camera_scale_source(cam)
    if FMT.get("fused"):
        out |= {"fused": fused_view(state, out), "agency_unknown": agency_unknown_view(state)}
    if FMT.get("undetermined"):
        out |= {"undetermined": undetermined_view(state, out), "prior_untracked": prior_untracked_view(state)}
    if FMT.get("rendered"):
        out["refit"] = refit_fixed_point_view(cam, out)
    return out


REFIT_KEYS = {"name", "pass", "iterations", "why", "admitted", "f", "f_scene", "steps"}
REFIT_STEP_KEYS = {"iteration", "f_scene", "admitted", "f", "sigma", "fusion_pass", "class_skipped"}


def refit_fixed_point_view(cam: dict, out: dict):
    """Rule C22 focal.refit_fixed_point: the passes of the focal refit -- each at its own f (0: stage A's), the votes
    the fusion admitted there and the f they give -- until the admitted set repeats (pass) or cycles / runs out
    (fail): {pass, why, steps [{iteration, f_scene, admitted, f, sigma, fusion_pass, class_skipped}]}; None without a
    refit. The record is the same object camera["focal.undetermined"]["refit_fixed_point"]."""
    w = "focal.refit_fixed_point"
    rp, fu = req(cam, w, "camera"), cam.get("focal.undetermined")
    if fu is None and rp is not None or fu is not None and fu.get("refit_fixed_point") != rp:
        raise ValueError(f"{AT['where']}: camera {w} differs from focal.undetermined.refit_fixed_point")
    if rp is None:
        return None
    st = rp["steps"] if isinstance(rp.get("steps"), list) else []
    if set(rp) != REFIT_KEYS or rp["name"] != w or not isinstance(rp["pass"], bool) or rp["iterations"] != len(st) - 1 \
            or not (len(rp["admitted"]) == len(rp["f"]) == len(rp["f_scene"]) == len(st)):
        raise ValueError(f"{AT['where']}: camera {w} keys {sorted(rp)} pass {rp.get('pass')!r} iterations {rp.get('iterations')!r}")
    steps = []
    for i, x in enumerate(st):
        ws = f"camera {w} step {i}"
        if set(x) != REFIT_STEP_KEYS or x["iteration"] != i or sorted(x["admitted"]) != sorted(rp["admitted"][i]) \
                or num(x["f_scene"], 3) != num(rp["f_scene"][i], 3) or num(x["f"], 3) != num(rp["f"][i], 3) \
                or (i > 0 and num(x["f_scene"], 3) != num(rp["f"][i - 1], 3)):
            raise ValueError(f"{AT['where']}: {ws} {x!r}")
        sk = []
        for k in x["class_skipped"]:
            if set(k) != {"name", "reason"} or k["reason"] not in CLASS_SKIP_REASONS or (i == 0 and (
                    k["reason"] in CLASS_SKIP_FIT if FMT.get("anchored") else k["reason"] != "focal.undetermined")):
                raise ValueError(f"{AT['where']}: {ws} class_skipped {k!r}")
            sk.append({"name": text(k["name"]), "reason": k["reason"]})
        steps.append({"iteration": i, "f_scene": num(x["f_scene"], 4), "admitted": [text(a) for a in x["admitted"]],
                      "f": num(x["f"], 4), "sigma": num(x["sigma"], 3), "fusion_pass": x["fusion_pass"], "class_skipped": sk})
    if num(steps[0]["f_scene"], 3) != num(req(fu, "stage_a_f", "focal.undetermined"), 3):
        raise ValueError(f"{AT['where']}: camera {w} iteration 0 f_scene {steps[0]['f_scene']} is not stage A's f")
    if rp["pass"] and (len(steps) < 2 or sorted(steps[-1]["admitted"]) != sorted(steps[-2]["admitted"])
                       or num(steps[-1]["f_scene"], 3) != num(out["f"], 3)):
        # the camera keeps the f the repeating pass ran at (f_scene); its fusion's f re-reads the fit-read votes there
        raise ValueError(f"{AT['where']}: camera {w} passes without a repeated admitted set at camera.f {out['f']}")
    return {"pass": rp["pass"], "why": text(rp["why"]), "steps": steps}


UNDETERMINED_KEYS = ("name", "pass", "reason", "stage_a_f", "stage_a_sigma", "stage_a_how", "kept", "f", "sigma",
                     "sources_used", "fusion_pass", "sources", "depth_model", "repair")


def undetermined_view(state: dict, cam: dict):
    """Stage A's focal length undetermined (sigma >= f), never kept; the camera's f from the focal fusion and the scene
    refitted at it (rule C17 focal.undetermined): {pass, reason, stage_a {f, sigma, how}, f, sigma, sources [..],
    used [text], depth_model {f, sigma, how, n} | None, class_skipped [..], depth_model_vote, refit {pass, f, sigma,
    reason} | None}. None when stage A's f is determined. A passing record is the scene's f (f_source focal.fusion)."""
    w = "focal.undetermined"
    fu = req(req(state, "camera", "scene"), w, "camera")
    src = text(req(state, "f_source", "scene"))
    if fu is None:
        if src == "focal.fusion":
            raise ValueError(f"{AT['where']}: f_source focal.fusion without a {w} record")
        return None
    for k in UNDETERMINED_KEYS:
        req(fu, k, w)
    ok = fu["pass"]
    if req(fu, "name", w) != w or not isinstance(ok, bool) or fu["kept"] is not False:
        raise ValueError(f"{AT['where']}: {w} name {fu['name']!r} pass {ok!r} kept {fu['kept']!r}")
    fa, sa = float(fu["stage_a_f"]), float(fu["stage_a_sigma"])
    if not sa >= fa:
        raise ValueError(f"{AT['where']}: {w} stage A sigma {sa} < f {fa}")
    if ok != (src == "focal.fusion"):
        raise ValueError(f"{AT['where']}: {w} pass {ok} with f_source {src!r}")
    if ok and num(fu["f"], 4) != cam["f"]:
        raise ValueError(f"{AT['where']}: {w} f {fu['f']} is not camera.f {cam['f']}")
    rows = []
    for x in fu["sources"]:
        ws = f"{w} source"
        v = text(req(x, "verdict", ws))
        if v not in FUSION_VERDICTS:
            raise ValueError(f"{AT['where']}: {ws} verdict {v!r}")
        rows.append({"object": text(req(x, "object", ws)), "kind": text(req(x, "kind", ws)), "f": num(req(x, "f", ws), 4),
                     "sigma": num(req(x, "sigma", ws), 3), "verdict": v, "why": text(req(x, "why", ws)) or None})
    used = [text(x) for x in fu["sources_used"]]
    if ok and len(used) != len([r for r in rows if r["verdict"] == "used"]) and not (FMT.get("sceneprog") and fu.get("repair")):
        raise ValueError(f"{AT['where']}: {w} sources_used {used} differ from the used sources")
    dm = fu["depth_model"]
    dmv = None if dm is None else {"f": num(req(dm, "f", f"{w} depth_model"), 4), "sigma": num(req(dm, "sigma", f"{w} depth_model"), 3),
                                   "how": text(req(dm, "how", f"{w} depth_model")),
                                   "n": len(req(dm, "per_frame", f"{w} depth_model"))}
    rp = fu["repair"]
    skipped = [{"name": text(req(x, "name", f"{w} class_skipped")), "class": text(req(x, "class", f"{w} class_skipped")),
                "reason": text(req(x, "reason", f"{w} class_skipped")), "why": text(req(x, "why", f"{w} class_skipped"))}
               for x in req(rp, "class_skipped", f"{w} repair")]
    rf = req(fu, "refit_fusion", w) if ok else fu.get("refit_fusion")
    rfv = None if rf is None else {"pass": req(rf, "pass", f"{w} refit_fusion"), "f": num(req(rf, "f", f"{w} refit_fusion"), 4),
                                   "sigma": num(req(rf, "sigma", f"{w} refit_fusion"), 3),
                                   "reason": text(req(rf, "reason", f"{w} refit_fusion"))}
    return {"pass": ok, "reason": text(req(fu, "reason", w)),
            "stage_a": {"f": num(fa, 4), "sigma": num(sa, 3), "how": text(fu["stage_a_how"])},
            "f": num(fu["f"], 4), "sigma": num(fu["sigma"], 3), "fusion_pass": fu["fusion_pass"], "sources": rows,
            "used": used, "depth_model": dmv, "class_skipped": skipped,
            "depth_model_vote": text(req(rp, "depth_model", f"{w} repair")), "refit": rfv}


PRIOR_UNTRACKED = re.compile(r"^prior object '(.+?)' has no track: ")   # pipeline.scene_fit.prior_untracked reason
PRIOR_UNTRACKED_CLASS = re.compile(r"^no instance of the class prior \((\w+)\) has a track: ")   # the same, rule C16


def prior_untracked_view(state: dict):
    """The prior's object without a track (rule D24 prior.untracked): {object, reason} or None. Every result of the
    unit then fails the check prior.untracked with that reason, and no result has it otherwise."""
    pu = text(req(req(state, "camera", "scene"), "prior.untracked", "camera")) or None
    for r in req(state, "results", "scene"):
        c = req(r, "checks", "scene result").get("prior.untracked")
        if (pu is None) != (c is None) or (c is not None and (c.get("pass") is not False or text(c.get("reason")) != pu)):
            raise ValueError(f"{AT['where']}: result check prior.untracked {c!r} with camera prior.untracked {pu!r}")
    if pu is None:
        return None
    if FMT.get("contained") and PRIOR_UNTRACKED_CLASS.match(pu):
        # rule C16: a class prior fails only when no instance is tracked; the instances are in prior.class_object
        co = req(state, "prior.class_object", "scene")
        if co.get("scope") != "class" or any(i.get("tracked") for i in co["instances"]):
            raise ValueError(f"{AT['where']}: class prior.untracked {pu!r} with prior.class_object {co.get('scope')!r}")
        return {"object": None, "instances": [[text(i["name"]), text(i["untracked"])] for i in co["instances"]], "reason": pu}
    m = PRIOR_UNTRACKED.match(pu)
    if not m:
        raise ValueError(f"{AT['where']}: prior.untracked reason {pu!r}")
    return {"object": m.group(1), "reason": pu}


def prior_untracked_repairs(rounds: dict | None, obj: str):
    """Rule D24 repair.prior_untracked: every agent round offers track_reselect on the untracked prior object."""
    for r in (rounds or {}).get("rounds", []):
        if r.get("planning_failed"):
            continue
        # v6: a reselect already tried on the unchanged state is dropped before the planner ("already tried on this
        # unchanged state"), which counts as offered
        dropped = [c for c in r.get("dropped", []) if FMT.get("sceneprog") and c["name"] == "track_reselect"
                   and c["params"].get("object") == obj and c["why"] == "already tried on this unchanged state"]
        if not dropped and not any(c["for"] == "prior.untracked" and c["name"] == "track_reselect" and c["params"].get("object") == obj
                   for c in r["candidates"]):
            raise ValueError(f"{AT['where']}: agent round {r['iter']} offers no track_reselect on {obj!r} for prior.untracked")


GROUND_VERSION = 4   # pipeline.ground VERSION (rule C19 support.full_frame)
FLOOR_DROP_REASONS = ("whole_frame", "under_1pct_after_cut", "empty", "under_1pct")


def floor_view(tag: str, unit: str):
    """The main support's coverage per sampled frame (rule C19 support.full_frame): raw, minus the tracked objects'
    boxes, the objects' pixels it covers, kept or the drop reason; None when the unit's ground record samples no
    frame. Each kept frame is one "floor" entry with the same coverages."""
    g = read_json(ROOT / "outputs" / f"ground_{tag}" / f"{unit}.json", "ground record")
    if req(g, "version", "ground record") != GROUND_VERSION or req(g, "unit", "ground record") != unit:
        raise ValueError(f"{AT['where']}: ground record version {g.get('version')!r} unit {g.get('unit')!r}")
    ff, fl = req(g, "floor_frames", "ground record"), req(g, "floor", "ground record")
    rows = []
    for x in ff:
        w = "ground floor_frames"
        kept, why = req(x, "kept", w), req(x, "reason", w)
        if not isinstance(kept, bool) or kept != (why is None) or (why is not None and why not in FLOOR_DROP_REASONS):
            raise ValueError(f"{AT['where']}: {w} frame {x.get('frame')} kept {kept!r} reason {why!r}")
        rows.append({"frame": int(req(x, "frame", w)), "raw": num(req(x, "coverage_raw", w), 4),
                     "minus": num(req(x, "coverage_minus_objects", w), 4), "objects": num(req(x, "objects_covered", w), 4),
                     "kept": kept, "reason": why})
    floor = {int(req(e, "frame", "ground floor")): (num(req(e, "coverage_raw", "ground floor"), 4),
                                                   num(req(e, "coverage_minus_objects", "ground floor"), 4)) for e in fl}
    if floor != {r["frame"]: (r["raw"], r["minus"]) for r in rows if r["kept"]}:
        raise ValueError(f"{AT['where']}: ground floor entries differ from the kept floor_frames")
    if not rows:
        return None
    return {"support": text(req(g, "support", "ground record")) or None, "frames": rows}


SCALE_SOURCES = ("prior", "depth_info", "class", "none")   # camera["scale.source"], results[i]["scale.source"]
PRIOR_CAMERA_KEYS = ("f_source", "focal_spread", "support_plane", "support_rejected", "s", "metric", "plane_fusion")


def camera_scale_source(cam: dict) -> str:
    """camera["scale.source"]; the prior-scale camera keys are always present (rule record.one_shape) and all null in a
    scene whose camera comes from its depth statements -- null there is absent by design, not a missing value."""
    for k in PRIOR_CAMERA_KEYS:
        req(cam, k, "camera")
    src = text(req(cam, "scale.source", "camera"))
    if src not in SCALE_SOURCES:
        raise ValueError(f"{AT['where']}: camera scale.source {src!r}")
    if src == "depth_info" and any(cam[k] is not None for k in PRIOR_CAMERA_KEYS):
        raise ValueError(f"{AT['where']}: depth-stated camera with prior-scale keys "
                         f"{[k for k in PRIOR_CAMERA_KEYS if cam[k] is not None]}")
    return src


def fused_view(state: dict, cam: dict):
    """camera.f as the fused focal estimate (rule C1a camera.fused): {f, sigma, n, how, was {f, source}}; None when
    camera.f keeps its own source."""
    fo = req(state, "focal", "scene")
    if cam["source"] != "camera.fused":
        if fo.get("fused") is not None:
            raise ValueError(f"{AT['where']}: focal.fused with f_source {cam['source']!r}")
        return None
    fz, w = req(fo, "fused", "focal"), "focal.fused"
    was = req(fz, "was", w)
    out = {"f": num(req(fz, "f", w), 4), "sigma": num(req(fz, "sigma", w), 3), "n": int(req(fz, "n_sources", w)),
           "how": text(req(fz, "how", w)),
           "was": {"f": num(req(was, "f", f"{w} was"), 4), "source": text(req(was, "source", f"{w} was"))}}
    if out["f"] != cam["f"]:
        raise ValueError(f"{AT['where']}: focal.fused f {out['f']} is not camera.f {cam['f']}")
    return out


def agency_unknown_view(state: dict):
    """Objects whose agency is unknown cast no gravity focal vote (rule agency.unknown): [{object, why}] or None."""
    rows = req(state, "focal", "scene").get("agency_unknown") or []
    return [{"object": text(req(x, "object", "focal.agency_unknown")), "why": text(req(x, "why", "focal.agency_unknown"))}
            for x in rows] or None


def support_plane_view(state: dict):
    """The named check support.plane (rule D9) with its sub-tests in the scene's order; pass None with the reason and
    no sub-test when the scene has no support plane."""
    sp = req(state, "support_plane", "scene")
    if req(sp, "pass", "support_plane") is None:
        if req(sp, "tests", "support_plane"):
            raise ValueError(f"{AT['where']}: support_plane without a result has sub-tests")
        return {"pass": None, "reason": text(req(sp, "reason", "support_plane")), "tests": []}
    tests = [{"name": k, "pass": req(t, "pass", f"support.plane {k}"), "reason": text(req(t, "reason", f"support.plane {k}"))}
             | ({"level": level_view(t)} if FMT.get("contact") and k == "level" else {})
             for k, t in req(sp, "tests", "support_plane").items()]
    failed = list(req(sp, "failed", "support_plane"))
    if failed != [t["name"] for t in tests if t["pass"] is False] or bool(req(sp, "pass", "support_plane")) == bool(failed):
        raise ValueError(f"{AT['where']}: support_plane pass / failed {failed} differ from its sub-tests")
    return {"pass": bool(sp["pass"]), "reason": None, "tests": tests}


def num_inf(x, sig=3):
    """num(), but an infinite sigma (a value that depends on no least-squares parameter) stays "inf"."""
    return "inf" if isinstance(x, float) and math.isinf(x) and x > 0 else num(x, sig)


LEVEL_PARTS = ("depth_plane", "depth_plane_frames", "depth_alignment", "scene_plane")


def level_view(t: dict, what: str = "support.plane level") -> dict:
    """The support plane against the depth model's plane through the main support's mask (rule support.main, sub-test
    level): angle vs 3 sigma, sigma and its parts (the depth plane: per-frame spread, alignment; the scene plane's own);
    every field null when there is no depth-model plane of the main support."""
    sp = req(t, "sigma_parts_deg", what)
    if sp is not None and set(sp) != set(LEVEL_PARTS):
        raise ValueError(f"{AT['where']}: {what} sigma parts {sorted(sp)}")
    return {"angle": num(req(t, "angle_deg", what), 3), "sigma": num_inf(req(t, "sigma_deg", what)),
            "parts": {k: num_inf(sp[k]) for k in LEVEL_PARTS} if sp is not None else None}


def main_plane_view(state: dict):
    """The plane through the main support's own evidence (rule support.main): its source, mask points, contacts,
    roll / pitch, residual, the objects fitted free because they stand on another support, and the refit when stage A's
    plane was not level (rule support.level_refit) as was -> now; None when the scene's plane is not from it."""
    mp = req(state, "main_plane", "scene")
    if not mp:
        return None
    w = "main_plane"
    tilt = req(mp, "tilt_deg", w)
    out = {"source": text(req(mp, "source", w)), "how": text(req(mp, "how", w)), "n_mask": int(req(mp, "n_mask_points", w)),
           "n_contacts": int(req(mp, "n_contacts", w)), "roll": num(tilt[0], 3), "pitch": num(tilt[1], 3),
           "height": num(-float(req(mp, "dist", w)), 4), "residual": num(req(mp, "residual_mad_m", w), 3),
           "spread": num(req(mp, "spread_deg", w), 3), "roll_source": text(mp.get("roll_source")) or None,
           "excluded": {text(k): [text(x) for x in v] for k, v in (mp.get("supports_excluded") or {}).items()} or None,
           "refit": None}
    if [text(x) for x in req(mp, "excluded", w)] != list(out["excluded"] or {}):
        raise ValueError(f"{AT['where']}: {w} excluded {mp['excluded']} differs from supports_excluded")
    rf = mp.get("refit")
    if rf is not None:
        wr = f"{w} refit"
        was, now = req(rf, "was", wr), req(rf, "now", wr)
        lv = req(was, "level", f"{wr} was")
        if req(lv, "level", f"{wr} was level") is not False:
            raise ValueError(f"{AT['where']}: {wr} of a level plane")
        out["refit"] = {k: [num(req(was, k, wr), 4), num(req(now, k, wr), 4)] for k in ("roll_deg", "pitch_deg", "rms")} | \
            {"height": [num(-float(req(was, "dist", wr)), 4), num(-float(req(now, "dist", wr)), 4)],
             "f": [num(req(was, "f", wr), 4), num(req(now, "f", wr), 4)], "level": level_view(lv, f"{wr} was level"),
             "why": text(req(lv, "why", f"{wr} was level")), "rule": text(req(mp, "why", w))}
    return out


def contact_event_view(e: dict, what: str) -> dict:
    """A contact of the scale object as a depth equation (rule scale.contact): the free-flight depth Z_scale against
    the other object's support-plane depth at the contact."""
    ok = req(e, "pass", what)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {what} pass = {ok!r}")
    return {"t": num(req(e, "t", what), 4), "other": text(req(e, "other", what)), "kind": text(req(e, "kind", what)),
            "source": text(req(e, "source", what)), "z_scale": num(req(e, "Z_scale", what), 4),
            "s_scale": num(req(e, "sigma_scale", what), 3), "z_other": num(req(e, "Z_other", what), 4),
            "s_other": num(req(e, "sigma_other", what), 3), "n": e.get("n_frames"), "gap": num(e.get("gap_s"), 3),
            "z": num(req(e, "z", what), 3), "pass": ok, "why": text(req(e, "why", what))} \
        | (reach_view(e, what) if FMT.get("interval") else {}) \
        | ({"s_line": num(req(e, "sigma_line", what), 3), "s_angles": num(req(e, "sigma_plane_angles", what), 3),
            "extrap": num(req(req(e, "extrapolation", what), "rel", f"{what} extrapolation"), 3)}
           if FMT.get("launch") else {})


REACH_POINTS = ("support", "hand")


def reach_view(e: dict, what: str) -> dict:
    """The reach of an articulated agent in a contact's sigma (rule scale.contact_reach): {"s_plane", "reach": {point,
    H, H_how, sigma} | None (the other object is no agent)}."""
    r = req(e, "reach", what)
    out = {"s_plane": num(req(e, "sigma_plane_depth", what), 3), "reach": None}
    if r is not None:
        w = f"{what} reach"
        if req(r, "point", w) not in REACH_POINTS:
            raise ValueError(f"{AT['where']}: {w} point {r['point']!r}")
        out["reach"] = {"point": r["point"], "H": num(req(r, "H", w), 3), "H_how": text(req(r, "H_how", w)),
                        "sigma": num(req(r, "sigma_reach", w), 3)}
    return out


def scale_contact_view(state: dict):
    """The scale object's contacts as depth equations (rule scale.contact): every event, pass (None: untestable) and
    reason; with FMT "launch" (rule plane.fusion) nothing is refitted on its verdict: "after" is the record re-tested
    at the fused plane, present exactly when the fusion rescaled the depths. None when the scene has no prior scale
    (an empty record)."""
    sc = req(state, "scale_contact", "scene")
    if not sc:
        return None
    w = "scale_contact"
    if req(sc, "name", w) != "scale.contact":
        raise ValueError(f"{AT['where']}: {w} name {sc['name']!r}")
    sp = (req(state, "camera", "scene") or {}).get("support_plane") or {}
    if "refit" in sc or "contact_refit" in sp:
        raise ValueError(f"{AT['where']}: {w} refit / camera.support_plane.contact_refit (rule plane.fusion removed them)")
    ok = req(sc, "pass", w)
    out = {"object": text(req(sc, "object", w)), "pass": ok, "reason": text(req(sc, "reason", w)),
           "events": [contact_event_view(e, f"{w} event") for e in req(sc, "events", w)], "after": None}
    pf = req(state, "camera", "scene").get("plane_fusion")    # absent in a depth-stated scene (plane_fusion_view)
    rescaled = pf is not None and pf.get("rescaled") is not None
    if ("after" in sc) != rescaled:
        raise ValueError(f"{AT['where']}: {w} after {'present' if 'after' in sc else 'absent'} with the fusion "
                         f"{'rescaled' if rescaled else 'not rescaled'}")
    if rescaled:
        af = sc["after"]
        out["after"] = {"pass": req(af, "pass", f"{w} after"), "reason": text(req(af, "reason", f"{w} after")),
                        "events": [contact_event_view(e, f"{w} after event") for e in req(af, "events", f"{w} after")]}
    return out


PLANE_SOURCES = ("depth_model.plane", "scale.contact")   # pipeline.scene_fit.plane_fusion


def plane_fusion_view(state: dict):
    """The camera height fused by inverse variance from the depth model's plane and every scale.contact depth
    equation (rule plane.fusion): {"pass" (None: one source), "z" (the largest pull, vs 3), "h", "sigma", "k",
    "h_dm", "sources" [{source, other, h, sigma, n_events, t, how}], "rescaled" {rule, k, n_dense, n_plane_anchors,
    n_depth_contacts, n_extents} | None, "reason"}; None in a scene without one."""
    cam = req(state, "camera", "scene")
    # a depth-stated scene's camera has neither a support_plane nor a plane_fusion record; any other has both (null:
    # no support plane)
    if ("plane_fusion" in cam) != ("support_plane" in cam):
        raise ValueError(f"{AT['where']}: scene camera has {'only plane_fusion' if 'plane_fusion' in cam else 'no plane_fusion'}")
    pf, sp = cam.get("plane_fusion"), cam.get("support_plane")
    if pf is None:
        if sp and sp.get("plane_fusion") is not None:
            raise ValueError(f"{AT['where']}: camera.support_plane.plane_fusion without camera.plane_fusion")
        return None
    w = "camera.plane_fusion"
    if req(pf, "name", w) != "plane.fusion" or not sp or req(sp, "plane_fusion", "camera.support_plane") != pf:
        raise ValueError(f"{AT['where']}: {w} name {pf.get('name')!r} / not mirrored in camera.support_plane")
    ok = req(pf, "pass", w)
    src = []
    for x in req(pf, "sources", w):
        ws = f"{w} source"
        k = req(x, "source", ws)
        if k not in PLANE_SOURCES:
            raise ValueError(f"{AT['where']}: {ws} {k!r}")
        row = {"source": k, "h": num(req(x, "h", ws), 4), "sigma": num(req(x, "sigma", ws), 3), "how": text(req(x, "how", ws))}
        if k == "scale.contact":
            row |= {"other": text(req(x, "other", ws)), "n_events": int(req(x, "n_events", ws)),
                    "t": [num(t, 4) for t in req(x, "t", ws)]}
        src.append(row)
    n = len(src)
    if not n or src[0]["source"] != "depth_model.plane" or ok not in (True, False, None) or (ok is None) != (n == 1) \
            or (req(pf, "z", w) is None) != (n == 1) or ("rescaled" in pf) != (n >= 2):
        raise ValueError(f"{AT['where']}: {w} pass {ok!r} z {pf.get('z')!r} rescaled {'rescaled' in pf} over {n} source(s)")
    rs = None
    if n >= 2:
        r, wr = pf["rescaled"], f"{w} rescaled"
        rs = {"rule": text(req(r, "rule", wr)), "k": num(req(r, "k", wr), 4)} | \
            {k: int(req(r, k, wr)) for k in ("n_dense", "n_plane_anchors", "n_depth_contacts", "n_extents")}
    return {"pass": ok, "z": num(pf["z"], 4), "h": num(req(pf, "h", w), 4), "sigma": num(req(pf, "sigma", w), 3),
            "k": num(req(pf, "k", w), 4), "h_dm": num(req(pf, "h_depth_model", w), 4), "sources": src, "rescaled": rs,
            "reason": text(req(pf, "reason", w))}


def plane_sigma_view(state: dict):
    """The plane's distance sigma at each object's foot row (rule plane.fusion): the row's depth beyond the support
    mask's depth range and the pitch lever there, and the odd-plane share (rule depth_rows.odd): {"extrapolation":
    [{obj, row, z_row, mask_depth [p05, p95], lever, ratio, phi, rel, beyond, odd_rel, plane_rel, how}], "odd":
    [[obj, rel]]} | None (none recorded)."""
    ps = req(state, "plane_sigma", "scene")
    if ps is None:
        return None
    if set(ps) != {"extrapolation", "odd"}:
        raise ValueError(f"{AT['where']}: plane_sigma keys {sorted(ps)}")
    ex = []
    for name, x in ps["extrapolation"].items():
        w = f"plane_sigma extrapolation {name!r}"
        md, beyond = req(x, "mask_depth", w), req(x, "beyond", w)
        if len(md) != 2 or not isinstance(beyond, bool):
            raise ValueError(f"{AT['where']}: {w} mask_depth {md!r} beyond {beyond!r}")
        ex.append({"obj": text(name), "row": num(req(x, "row", w), 4), "z_row": num(req(x, "Z_row", w), 4),
                   "mask_depth": [num(md[0], 4), num(md[1], 4)], "lever": num(req(x, "lever_m", w), 3),
                   "ratio": num(req(x, "ratio", w), 3), "phi": num(req(x, "phi_frame_deg", w), 3),
                   "rel": num(req(x, "rel", w), 3), "beyond": beyond, "odd_rel": num(req(x, "odd_rel", w), 3),
                   "plane_rel": num(req(x, "plane_rel", w), 3), "how": text(req(x, "how", w))})
    odd = [[text(k), num(v, 3)] for k, v in ps["odd"].items()]
    return {"extrapolation": ex, "odd": odd} if ex or odd else None


def depth_rows_view(state: dict):
    """Each object's silhouette depth against its support-plane depth before the fit (rule scale.contact_rescale,
    test direct.depth_agree): [{obj, pass (None: untestable), z_mask, z_plane, bound, n_rows, n_matched, dropped,
    reason}] | None."""
    out = []
    for name, d in req(state, "depth_rows", "scene").items():
        w = f"depth_rows {name!r}"
        if req(d, "name", w) != "direct.depth_agree":
            raise ValueError(f"{AT['where']}: {w} name {d['name']!r}")
        ok, dropped = req(d, "pass", w), req(d, "dropped", w)
        odd = req(d, "odd", w) if FMT.get("launch") else None
        # rule depth_rows.odd: a failed test names the odd source; "plane" keeps the rows, "silhouette" drops them
        if ok not in (True, False, None) or (odd is None) != (ok is not False) or odd not in (None, "plane", "silhouette") \
                or dropped is not (ok is False and odd != "plane"):
            raise ValueError(f"{AT['where']}: {w} pass {ok!r} odd {odd!r} with dropped {dropped!r}")
        gr = None
        if FMT.get("classref"):    # rule record.one_shape: every key present, null when D10b did not run
            for k in ("n_matched", "odd", "ground", "plane_sigma_added"):
                req(d, k, w)
        if FMT.get("launch") and d.get("ground") is not None:
            g, wg = d["ground"], f"{w} ground"
            gr = {"z": num(req(g, "z_ground", wg), 4), "n": int(req(g, "n", wg)), "pass": req(g, "pass", wg),
                  "bound": num(req(g, "bound", wg), 3), "reason": text(req(g, "reason", wg))}
        # an untestable record (pass null) has no depths or bound: no tracked frame on the main support (n_matched
        # null), or fewer matched dense rows than the test needs (n_matched the count)
        untestable = ok is None and FMT.get("sizeonly")
        if untestable and (any(k in d for k in ("z_mask", "z_plane", "bound"))
                           or not (d["n_matched"] is None or isinstance(d["n_matched"], int))
                           or not text(d["reason"]).startswith("untestable")):
            raise ValueError(f"{AT['where']}: {w} untestable with keys {sorted(d)}")
        g3 = (lambda k: None) if untestable else (lambda k: req(d, k, w))
        nm = req(d, "n_matched", w)
        out.append({"obj": text(name), "pass": ok, "z_mask": num(g3("z_mask"), 4), "z_plane": num(g3("z_plane"), 4),
                    "bound": num(g3("bound"), 3), "n_rows": int(req(d, "n_rows", w)),
                    "n_matched": None if nm is None and untestable else int(nm), "dropped": dropped, "reason": text(req(d, "reason", w))}
                   | ({"odd": odd, "ground": gr,
                       "sigma_added": num(d["plane_sigma_added"] if FMT.get("classref") else d.get("plane_sigma_added"), 3)}
                      if FMT.get("launch") else {}))
    return out or None


FIT_OBJECT_ROWS = ("box", "depth_anchor", "prior")


def fit_object_view(name: str, o: dict) -> dict:
    """An object's fit as a whole against its calibration rows (rule fit.object): s^2 vs its bound (None: untestable,
    no stated row), dof, and chi^2 / n per row kind; a fail is a failed object fit."""
    w = f"object {name!r} fit_object"
    fo = req(o, "fit_object", f"scene object {name!r}")
    if req(fo, "name", w) != "fit.object":
        raise ValueError(f"{AT['where']}: {w} name {fo['name']!r}")
    ok = req(fo, "pass", w)
    if ok not in (True, False, None) or (ok is None) != (req(fo, "bound", w) is None):
        raise ValueError(f"{AT['where']}: {w} pass {ok!r} with bound {fo.get('bound')!r}")
    rows = req(fo, "rows", w)
    if not set(rows) <= set(FIT_OBJECT_ROWS):
        raise ValueError(f"{AT['where']}: {w} row kinds {sorted(rows)}")
    return {"pass": ok, "s2": num(req(fo, "s2", w), 4), "bound": num(fo["bound"], 4), "dof": int(req(fo, "dof", w)),
            "chi2": num(req(fo, "chi2", w), 4), "n_fits": int(req(fo, "n_fits", w)),
            "rows": [[k, num(req(rows[k], "chi2", w), 4), int(req(rows[k], "n", w))] for k in FIT_OBJECT_ROWS if k in rows],
            "reason": text(req(fo, "reason", w))} | (fit_object_facts(fo, w) if FMT.get("interval") else {})


def fit_object_facts(fo: dict, w: str) -> dict:
    """The facts fit.object records beside its verdict (rule fit.object_noise), none of them a gate: each stated row's
    tension z, the box sides' measured noise and bias, the box-noise test's s^2 vs its bound, and where the chain
    flies while the graph puts the object on a support."""
    st = [{"kind": text(req(x, "kind", f"{w} stated")), "z": num(req(x, "z", f"{w} stated"), 3),
           "residual": num(req(x, "residual", f"{w} stated"), 3), "s_stated": num(req(x, "sigma_stated", f"{w} stated"), 3),
           "s_pred": num(req(x, "sigma_pred", f"{w} stated"), 3)} for x in req(fo, "stated", w)]
    bn, bb, bt = req(fo, "box_noise_px", w), req(fo, "box_bias_px", w), req(fo, "box_noise_test", w)
    if (bn is None) != (bb is None) or (bn is None) != (bt is None):
        raise ValueError(f"{AT['where']}: {w} box_noise_px / box_bias_px / box_noise_test not all present or all null")
    out = {"stated": st or None, "sides": None, "box_noise": None, "box_bias": None, "box_rms": None, "box_test": None,
           "mismatch": None}
    if bn is not None:
        sides = list(req(bn, "sides", f"{w} box_noise_px"))
        if sides not in BOX_SIDES or list(req(bb, "sides", f"{w} box_bias_px")) != sides:
            raise ValueError(f"{AT['where']}: {w} box sides {sides} / {bb.get('sides')}")
        cols = {"box_noise": req(bn, "sigma", f"{w} box_noise_px"), "box_bias": req(bb, "bias_px", f"{w} box_bias_px"),
                "box_rms": list(zip(req(bb, "rms_raw", f"{w} box_bias_px"), req(bb, "rms_left", f"{w} box_bias_px")))}
        if any(len(v) != len(sides) for v in cols.values()):
            raise ValueError(f"{AT['where']}: {w} per-side lists differ in length from sides {sides}")
        out |= {"sides": sides, "box_noise": [num(x, 3) for x in cols["box_noise"]],
                "box_bias": [num(x, 3) for x in cols["box_bias"]],
                "box_rms": [[num(a, 3), num(b, 3)] for a, b in cols["box_rms"]]}
        wt = f"{w} box_noise_test"
        ok = req(bt, "pass", wt)
        if ok is None:     # untestable: no dof left after the bias parameters
            out["box_test"] = {"pass": None, "dof": int(req(bt, "dof", wt)), "why": text(req(bt, "why", wt))}
        elif isinstance(ok, bool):
            out["box_test"] = {"pass": ok, "s2": num(req(bt, "s2", wt), 4), "bound": num(req(bt, "bound", wt), 4),
                               "dof": int(req(bt, "dof", wt)), "why": text(req(bt, "why", wt))}
        else:
            raise ValueError(f"{AT['where']}: {wt} pass {ok!r}")
    sm = req(fo, "state_mismatch", w)
    ws = f"{w} state_mismatch"
    if sm is None or req(sm, "name", ws) != "fit.object.state_mismatch":
        raise ValueError(f"{AT['where']}: {ws} is {sm!r}, expected a fit.object.state_mismatch record")
    ok = req(sm, "pass", ws)
    if ok not in (True, False, None):
        raise ValueError(f"{AT['where']}: {ws} pass {ok!r}")
    t = req(sm, "t", ws)
    out["mismatch"] = {"pass": ok, "frames": int(req(sm, "frames", ws)), "t": [num(x, 4) for x in t] if t else None,
                       "reason": text(req(sm, "reason", ws))}
    return out


BOX_SIDES = (["x1", "y1", "x2", "y2"], ["u", "v", "d"])   # box sides, or a pendulum's centre u, v and diameter d


SWING_SERIES_KEYS = {"name", "noise", "troughs", "rejected", "period", "sigma", "n_periods", "basis", "why"}


def swing_period_view(sw: dict, p: dict, w: str) -> dict:
    """Rule B30 swing_period.agree: the period read from both series (d: the ball's diameter, u: its column) -- each
    series' troughs above 3 sigma of its own noise, its period; both agree -> their inverse-variance mean (chosen),
    else none (pendulum.period untestable)."""
    w = f"{w} swing_period"
    if set(sw) != {"series", "chosen", "agree", "gates", "start", "why"} or set(sw["agree"]) != {"pass", "z"} \
            or sw["agree"]["pass"] not in (True, False, None):
        raise ValueError(f"{AT['where']}: {w} keys {sorted(sw)} agree {sw.get('agree')!r}")
    series = []
    for x in sw["series"]:
        if set(x) != SWING_SERIES_KEYS or x["name"] not in ("d", "u") or x["basis"] not in ("noise", "quantization", None):
            raise ValueError(f"{AT['where']}: {w} series {sorted(x)} name {x.get('name')!r} basis {x.get('basis')!r}")
        tr = []
        for t in x["troughs"]:
            if set(t) != {"frame", "t", "depth", "sigma_t", "bottom"} or len(t["bottom"]) != 2:
                raise ValueError(f"{AT['where']}: {w} trough {t!r}")
            tr.append({"frame": int(t["frame"]), "t": num(t["t"], 4), "depth": num(t["depth"], 3), "sigma_t": num(t["sigma_t"], 3),
                       "bottom": [int(b) for b in t["bottom"]]})
        rj = []
        for r in x["rejected"]:
            if set(r) != {"frame", "depth", "why"}:
                raise ValueError(f"{AT['where']}: {w} rejected trough {r!r}")
            rj.append({"frame": int(r["frame"]), "depth": num(r["depth"], 3), "why": text(r["why"])})
        series.append({"name": x["name"], "noise": num(x["noise"], 3), "troughs": tr, "rejected": rj, "period": num(x["period"], 4),
                       "sigma": num(x["sigma"], 3), "n_periods": x["n_periods"], "basis": x["basis"], "why": text(x["why"])})
    ch = sw["chosen"]
    if ch is not None:
        if set(ch) != {"period", "sigma", "n_periods", "from"} or num(ch["period"], 4) != num(p["period_s"], 4) \
                or "+".join(ch["from"]) != p["period_from"]:
            raise ValueError(f"{AT['where']}: {w} chosen {ch!r} vs period_s {p['period_s']!r} from {p['period_from']!r}")
    elif p["period_s"] is not None or p["period_from"] is not None:
        raise ValueError(f"{AT['where']}: {w} chosen null with period_s {p['period_s']!r} from {p['period_from']!r}")
    return {"series": series, "chosen": None if ch is None else {"period": num(ch["period"], 4), "sigma": num(ch["sigma"], 3),
                                                                 "n_periods": ch["n_periods"], "from": list(ch["from"])},
            "agree": {"pass": sw["agree"]["pass"], "z": num(sw["agree"]["z"], 3)}, "gates": sw["gates"],
            "start": num(sw["start"], 4), "why": text(sw["why"])}


def pendulum_view(p: dict | None, what: str):
    """A pendulum object's record (rules scale.converged / pendulum.period): fitted L, the swing period and its sigma,
    and the focal length solved jointly with the swing under the stated size (f_free). None for a non-pendulum."""
    if p is None:
        return None
    w = f"{what} pendulum"
    ff = req(p, "f_free", w)
    out = {"L": num(req(p, "L", w), 4), "period": num(req(p, "period_s", w), 4), "period_sigma": num(req(p, "period_sigma_s", w), 3),
           "periods": req(p, "periods", w), "period_from": text(req(p, "period_from", w)) or None, "f": num(req(p, "f", w), 4),
           "f_free": None}
    if FMT.get("unobserved"):
        out["swing"] = swing_period_view(req(p, "swing_period", w), p, w)
    if ff is not None:
        wf = f"{w} f_free"
        ab = req(ff, "at_bound", wf)
        out["f_free"] = {"f": num(req(ff, "f", wf), 4), "f_start": num(req(ff, "f_start", wf), 4),
                         "bounds": [num(x, 4) for x in req(ff, "bounds", wf)], "at_bound": text(ab) if ab is not None else None,
                         "D": num(req(ff, "D", wf), 4), "D_prior": num(req(ff, "D_prior", wf), 4),
                         "rel": num(req(ff, "rel", wf), 3), "converged": bool(req(ff, "converged", wf))}
    return out


def clamp_view(c: dict | None):
    """The first focal estimate clamped into the person-height band (scene f_first_clamp), reported; None if unclamped."""
    if c is None:
        return None
    return {"from": num(req(c, "from", "f_first_clamp"), 4), "to": num(req(c, "to", "f_first_clamp"), 4),
            "band": [num(x, 4) for x in req(c, "band", "f_first_clamp")]}


def video_meta(vid: str, make_thumb: bool):
    """(duration_s, width, height, thumb_rel_path) from the mp4; thumbnail from the first frame."""
    p = video_path("val", vid)
    if not p.exists():
        raise ValueError(f"{vid}: video missing: {p}")
    cap = cv2.VideoCapture(str(p))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not (ok and fps > 0 and n > 0):
        raise ValueError(f"{vid}: video unreadable: {p}")
    out = THUMBS / f"{vid}.jpg"
    if make_thumb or not out.exists():
        th = max(1, round(frame.shape[0] * THUMB_W / frame.shape[1]))
        small = cv2.resize(frame, (THUMB_W, th), interpolation=cv2.INTER_AREA)
        for q in (72, 60, 48, 36, 25):
            ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok2 and len(buf) <= THUMB_MAX_BYTES:
                break
        out.write_bytes(buf.tobytes())
    return num(n / fps, 4), w, h, str(out.relative_to(WEB))


def fourcc(p: Path) -> str:
    cap = cv2.VideoCapture(str(p))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((code >> (8 * k)) & 0xFF) for k in range(4)).strip("\x00").lower()


def video_info(vid: str) -> dict:
    """Page-relative URL of the original mp4 (media/<split> are git-ignored symlinks to the video dirs of video_path()).

    Browsers only decode H.264 here. All validation mp4s are MPEG-4 Part 2 ('mp4v'); the test set has the same
    video_id at higher resolution, usually in H.264, so the page plays that copy instead and says so.
    "video" is None when no browser-playable copy exists; "orig" is always the original file.
    """
    orig = f"media/val/{video_path('val', vid).name}"
    codec = fourcc(video_path("val", vid))
    info = {"orig": orig, "codec": codec, "video": orig if codec in ("avc1", "h264") else None, "video_from": "val"}
    if info["video"] is None and video_path("test", vid).exists() and fourcc(video_path("test", vid)) in ("avc1", "h264"):
        info.update(video=f"media/test/{video_path('test', vid).name}", video_from="test")
    return info


def harness_groups(saved: dict) -> list[dict]:
    """The independent scenes of one video: one per prior group ("groups"), or the video's single scene."""
    if "groups" in saved:
        return [{"group": str(g["group"]), "prior": text(g["prior"]), "depth_info": text(g["depth_info"]),
                 "state": g["scene"]} for g in saved["groups"]]
    context = saved["question_context"][0]
    return [{"group": "0", "prior": text(context["prior"]), "depth_info": text(context["depth_info"]), "state": saved}]


def bind_harness_question(row: dict, answers: list[dict], groups: list[dict]) -> tuple[dict | None, dict | None, str | None]:
    """Bind row identity and scene_id exactly, including same text under different priors."""
    candidates = [a for a in answers if req(a, "row", "answer record") == row["row"]
                  and req(a, "question", "answer record") == row["question"]
                  and text(req(a, "prior", "answer record")) == text(row["prior"])]
    if len(candidates) != 1:
        return None, None, "answer identity missing or duplicated"
    answer = candidates[0]
    scene_id = req(answer, "scene_id", "answer record")
    if not scene_id or text(row["scene_id"]) != scene_id:
        return answer, None, "scene_id missing or differs between CSV and answer record"
    matches = [g for g in groups if req(g["state"], "scene_id", "scene") == scene_id]
    if len(matches) != 1:
        return answer, None, "scene_id missing or duplicated among independent group scenes"
    group = matches[0]
    contexts = [c for c in req(group["state"], "question_context", "scene")
                if req(c, "row", "question context") == row["row"] and req(c, "question", "question context") == row["question"]
                and text(req(c, "prior", "question context")) == text(row["prior"])
                and text(req(c, "depth_info", "question context")) == text(row["depth_info"])]
    if len(contexts) != 1:
        return answer, None, "row/prior/depth identity not unique in the matched scene"
    return answer, group, None


OUTSIDE_VIDEO = "asked time outside the recorded video"


class Overlays:
    """The overlay record of one version, web/data/runtime/<tag>/overlays.json: {"<scene_id> <t>": entry}, entry =
    {"t", "scene_id", "renderer", "renderer_version", "renderer_hash", "path" | "unavailable"}."""

    def __init__(self, tag: str):
        self.tag, self.dir = tag, WEB / "data" / "runtime" / tag
        self.file = self.dir / "overlays.json"
        self.render = base_version(tag) == CURRENT
        if not self.file.exists() and not self.render:
            raise ValueError(f"{tag} is not the current version ({CURRENT}) and has no overlay record {self.file.relative_to(WEB)}")
        self.record = json.loads(self.file.read_text()) if self.file.exists() else {}
        self.added = 0
        self.used: set[str] = set()

    def get(self, vid: str, scene: dict, question: str, fps: float, duration: float, subdir: str) -> list[dict]:
        """The overlays of one question at its times; rendered images go to <tag>/<subdir>/."""
        times = list(dict.fromkeys(float(t) for t in parse_question(question).times))[:2] or [0.]
        out = []
        for t in times:
            key = f"{scene['scene_id']} {t:.9g}"
            entry = self.record.get(key)
            self.used.add(key)
            if entry is None:
                if not self.render:
                    raise ValueError(f"{vid}: overlay {key} not in {self.file.relative_to(WEB)}; only {CURRENT} renders")
                entry = self.record[key] = self._render(vid, scene, t, fps, duration, subdir)
                self.added += 1
            if entry.get("path") and not (WEB / entry["path"]).exists():
                raise ValueError(f"{vid}: overlay image missing: {entry['path']}")
            out.append(entry)
        return out

    def _render(self, vid: str, scene: dict, t: float, fps: float, duration: float, subdir: str) -> dict:
        """Render saved state only; no fitting, assembly, model requests or GT inputs."""
        from pipeline.data import Video
        from pipeline.scene_state3d import render_overlay
        renderer_hash = hashlib.sha256((ROOT / "pipeline" / "scene_state3d.py").read_bytes()).hexdigest()[:16]
        entry = {"t": t, "scene_id": scene["scene_id"], "renderer": "scene_state3d", "renderer_version": CURRENT,
                 "renderer_hash": renderer_hash}
        if not math.isfinite(t) or t < 0 or t > duration + 1 / fps:
            entry["unavailable"] = OUTSIDE_VIDEO
            return entry
        frame = round(t * fps)
        identity = hashlib.sha256(f"{scene['scene_id']}:{renderer_hash}:{t:.9g}".encode()).hexdigest()[:24]
        path = self.dir / subdir / f"{identity}_f{frame}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            video = Video(video_path("val", vid), fps=fps)
            try:
                image = render_overlay(scene, t, frame=video.frame_at(t))
            finally:
                video.close()
            if image is None or not cv2.imwrite(str(path), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86]):
                raise ValueError("runtime produced no image")
            entry["path"] = str(path.relative_to(WEB))
        except Exception as exc:   # noqa: BLE001 - the renderer's failure is this version's result, shown on the page
            entry["unavailable"] = f"{type(exc).__name__}: {exc}"[:240]
        return entry

    def save(self):
        """Write the record. The current version also drops the overlays no question of this build references and
        deletes their images (the build covers every question the version is shown for)."""
        removed = 0
        if self.render:
            for key in sorted(set(self.record) - self.used):
                path = self.record.pop(key).get("path")
                if path and (WEB / path).exists():
                    (WEB / path).unlink()
                removed += 1
            for d in sorted(self.dir.glob("*/"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        if removed:
            print(f"{self.tag}: {removed} unreferenced overlays removed")
        if self.added or removed:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(self.record, ensure_ascii=False, indent=1, sort_keys=True))


def manifest_units() -> pd.DataFrame:
    """The validation table with each row's unit id (<video_id>_q<k>, k = order of the row within its video)."""
    manifest = load_split("val").reset_index(drop=True)
    manifest["k"] = manifest.groupby("video_id", sort=False).cumcount() + 1
    manifest["unit"] = [unit_id(v, int(k)) for v, k in zip(manifest["video_id"], manifest["k"])]
    return manifest


def harness_rows(frame: pd.DataFrame, manifest: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    """The rows of `units` from a harness CSV, in that order, each bound to its manifest row by row identity and
    checked against it; labels always come from the manifest. A unit without its row in the CSV is an error."""
    if "row" not in frame:
        raise ValueError("harness CSV lacks row identities")
    by_unit = manifest.set_index("unit")
    missing = [u for u in units if u not in by_unit.index]
    if missing:
        raise ValueError(f"units not in the validation table: {missing}")
    rows = pd.to_numeric(frame["row"], errors="raise").astype(int)
    if rows.duplicated().any():
        raise ValueError("harness CSV has duplicated row identities")
    frame = frame.set_axis(rows.to_numpy())
    absent = [u for u in units if int(by_unit.index.get_loc(u)) not in frame.index]
    if absent:
        raise ValueError(f"units without an answer row in the harness CSV: {absent}")
    out = []
    for u in units:
        i = int(by_unit.index.get_loc(u))
        r, m = frame.loc[i].to_dict(), manifest.loc[i]
        for key in ("video_id", "question", "prior", "depth_info"):
            if text(r[key]) != text(m[key]):
                raise ValueError(f"{u}: harness CSV {key} differs from the validation table")
        if "k" in r and int(r["k"]) != int(m["k"]):
            raise ValueError(f"{u}: harness CSV k {r['k']} differs from the validation table's {m['k']}")
        for key in ("answer", "category", "dim", "fps", "video_type", "video_source", "inference_type", "k"):
            r[key] = m[key]
        r["row"], r["unit"] = i, u
        out.append(r)
    return pd.DataFrame(out)


def unit_id(vid: str, k: int) -> str:
    """The unit of one question: <video_id>_q<k>, k = its 1-based order among the video's rows of the split table."""
    return f"{vid}_q{k}"


def dev_units() -> list[str]:
    """The development question set: the unit ids in the first column of the table in docs/DEV20.md."""
    units = re.findall(r"^\|\s*(\S+_q\d+)\s*\|", (ROOT / "docs" / "DEV20.md").read_text(), re.M)
    if not units or len(set(units)) != len(units):
        raise ValueError(f"docs/DEV20.md: no or duplicated unit ids: {units}")
    return units


def own_result(state: dict, row: dict) -> dict | None:
    """This question's entry of a scene's "results" (exactly one, or none when the scene gave it no value)."""
    results = [r for r in req(state, "results", "scene") if req(r, "question", "scene result") == row["question"]
               and req(r, "row", "scene result") == row["row"]]
    if len(results) > 1:
        raise ValueError(f"row {row['row']}: {len(results)} scene results for one question")
    return results[0] if results else None


def run_rows(tag: str, csv_path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """The harness CSV of one run, its "row" identities checked to be unique rows of the validation table."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(f"invalid harness run tag: {tag!r}")
    version_format(tag)
    if not csv_path.exists():
        raise ValueError(f"{tag}: harness CSV missing: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "row" not in frame:
        raise ValueError(f"{csv_path.name}: harness CSV lacks row identities")
    rows = pd.to_numeric(frame["row"], errors="raise").astype(int)
    outside = sorted(set(rows) - set(manifest.index))
    if outside:
        raise ValueError(f"{csv_path.name}: rows not in the validation table: {outside}")
    return frame


def failed_unit(rec: dict, row: dict) -> dict:
    """A unit whose chain stopped (answer record "execution_status" failed): its "error" as the harness wrote it (an
    unresolved inventory or scene graph names its reason, objects.llm_invalid / scenegraph.llm_invalid), the steps it
    ran; no answer, no scene."""
    err = text(req(rec, "error", "answer record"))
    if not err or req(rec, "answer", "answer record") is not None:
        raise ValueError(f"{AT['where']}: failed unit with error {err!r} and an answer")
    if text(row["status"]) != "failed" or text(row["source"]) != "execution_failed" or num(row["pred"]) is not None:
        raise ValueError(f"{AT['where']}: failed unit with CSV status {row['status']!r} / source {row['source']!r}")
    return {"error": err, "execution_status": "failed", "time": timing_view(req(rec, "timing", "answer record"),
                                                                           row.get("seconds"), prefix=True),
            "checks": [], "failed": [], "scene": None, "sgraph": None, "tracks": None, "overlays": [], "fit": None,
            "rounds": None, "identity": None, "refmeasure": None, "extents": None, "depth_bindings": None,
            "measurement": None, "provenance": None, "quality": {}, "align": None, "group": None, "binding_error": None}


def stale_row_unit(rec: dict, row: dict) -> dict:
    """A v6 unit whose CSV row says execution_failed while its answer record is complete (a retry under another run
    signature, docs/VAL_V6.md): the CSV's verdict, no harness answer or scene, the record's own timing."""
    sig = json.loads(text(row["version"]))["run_signature"]
    if sig != req(req(rec, "version", "answer record"), "run_signature", "answer record version") or \
            text(row["status"]) != "failed" or num(row["pred"]) is not None:
        raise ValueError(f"{AT['where']}: execution_failed CSV row with a complete record of another signature or a pred")
    out = failed_unit({"error": "listed execution_failed in the run's CSV; the answer record of its retry is complete",
                       "answer": None, "timing": {**req(rec, "timing", "answer record")}}, row | {"seconds": req(req(rec, "timing", "answer record"), "total_seconds", "timing")})
    out["time"] = timing_view(req(rec, "timing", "answer record"), req(req(rec, "timing", "answer record"), "total_seconds", "timing"))
    return out


def timing_view(t: dict, csv_seconds, prefix: bool = False) -> dict:
    """A unit's timing ("timing" of its answer record) for the page: total seconds and one entry per step, in pipeline
    order: {"step", "s", "reused" (false | "partial" | true), "cached", "shared"}. The CSV "seconds" must be the total.
    prefix: a failed unit, whose steps are the first steps of the order."""
    total = req(t, "total_seconds", "timing")
    if not (isinstance(csv_seconds, (int, float)) and math.isfinite(csv_seconds) and abs(csv_seconds - total) < 0.05):
        raise ValueError(f"{AT['where']}: CSV seconds {csv_seconds} differs from the answer record's total {total}")
    steps, order = req(t, "steps", "timing"), FMT["steps"]
    if prefix:
        order = order[:len(steps)]
    if list(steps) != list(order):
        raise ValueError(f"{AT['where']}: timing steps {list(steps)}, expected {list(order)}")
    out = []
    for name in order:
        x = steps[name]
        reused = req(x, "reused", f"timing step {name}")
        if reused not in (False, True, "partial"):
            raise ValueError(f"{AT['where']}: timing step {name} reused = {reused!r}")
        out.append({"step": name, "s": num(req(x, "seconds", f"timing step {name}"), 4), "reused": reused,
                    "cached": bool(req(x, "cached", f"timing step {name}")), "shared": x.get("shared")})
    return {"total": num(total, 5), "steps": out}


def question_record(tag: str, unit: str, row: dict) -> dict:
    """The answer record outputs/harness/<tag>/answers/<unit>.json of a unit-question run (schema "unit-1"), its
    identity checked against the question's row."""
    rec = unit_json("answers", unit, "answer record")
    if req(rec, "schema", "answer record") != "unit-1":
        raise ValueError(f"{unit}: answer record schema {rec['schema']!r}, expected 'unit-1'")
    for key in ("k", "row", "question", "prior", "depth_info"):
        if text(req(rec, key, "answer record")) != text(row[key]):
            raise ValueError(f"{unit}: answer record {key} {rec[key]!r} differs from the table's {row[key]!r}")
    if not run_label(tag, req(rec, "version", "answer record")["pipeline"]):
        raise ValueError(f"{unit}: answer artifact version {rec['version']['pipeline']} differs from requested {tag}")
    return rec


def question_scene(unit: str, row: dict, record: dict) -> dict | None:
    """The unit's own saved scene (agent/<unit>.json): exactly its one row as question context, no groups.
    None only when the run failed before writing it."""
    if not (DIRS["scenes"] / f"{unit}.json").exists() and record["execution_status"] == "failed":
        return None
    state = unit_json("scenes", unit, "question scene")
    if "groups" in state or int(req(state, "k", "question scene")) != int(row["k"]):
        raise ValueError(f"{unit}: question scene has groups or k {state.get('k')}")
    ctx = req(state, "question_context", "question scene")
    if len(ctx) != 1 or req(ctx[0], "row", "question context") != row["row"] or ctx[0].get("unit") != unit \
            or req(ctx[0], "question", "question context") != row["question"]:
        raise ValueError(f"{unit}: question scene context is not exactly this question")
    return state


def build_run(tag: str, frame: pd.DataFrame, manifest: pd.DataFrame, units: list[str], videos: dict,
              make_thumb: bool) -> list[dict]:
    """The questions `units` from one run (all of them must be in its CSV), every value read from that run's own
    artifacts in its own format. Unit "video": the run's per-video artifacts go to videos[vid]["runs"][tag]. Unit
    "question": each question reads only its own unit's files and carries them; nothing per video."""
    FMT.clear()
    FMT.update(version_format(tag))
    unit_kind = FMT["unit"]
    DIRS.clear()
    DIRS.update(scenes=ROOT / "outputs/scene_fit_eval" / f"h_{tag}" / "agent", sgraph=ROOT / "outputs" / f"scenegraph_{tag}",
                tracks=ROOT / "outputs" / f"tracks_{tag}", objects=ROOT / "outputs" / f"objects_{tag}",
                answers=ROOT / "outputs/harness" / tag / "answers")
    df = harness_rows(frame, manifest, units)
    df["mra"] = [mra_row(p, g) for p, g in zip(df["pred"], df["answer"])]
    overlays = Overlays(tag)
    questions = []
    for vid, rows in df.groupby("video_id", sort=True):
        first = rows.iloc[0]
        AT["where"] = f"{tag} {vid}"
        if vid not in videos:
            duration, width, height, thumb = video_meta(vid, make_thumb)
            videos[vid] = {"id": vid, "source": first.video_source, "video_type": first.video_type,
                           "fps": num(first.fps), "duration": duration, "size": [width, height], "thumb": thumb,
                           **video_info(vid), "runs": {}}
        video = videos[vid]
        duration = video["duration"]
        groups = []
        if unit_kind == "video":
            record = read_json(DIRS["answers"] / f"{vid}.json", "answer record")
            version = record["version"]
            if version["pipeline"] != tag.removesuffix("_control"):
                raise ValueError(f"{vid}: answer artifact version {version['pipeline']} differs from requested {tag}")
            run = video["runs"][tag] = {"run_at": version["time"]}
            scene_path = DIRS["scenes"] / f"{vid}.json"
            if scene_path.exists():
                groups = harness_groups(json.loads(scene_path.read_text()))
            elif record["execution_status"] != "failed":
                raise ValueError(f"{vid}: scene missing for a run that did not fail: {scene_path.relative_to(ROOT)}")
            run |= {"groups": [{"group": g["group"], "scene_id": g["state"]["scene_id"],
                                "scene": scene_view(g["state"], scene_path)} for g in groups],
                    "sgraph": sgraph(vid), "tracks": track_intervals(vid)}
        for row in rows.to_dict("records"):
            unit = row["unit"]
            AT["where"] = f"{tag} {unit}"
            query = {"id": unit, "version": tag, "video": vid, "k": int(row["k"]), "row": int(row["row"]),
                     "cat": row["category"], "type": row["inference_type"],
                     "q": row["question"], "prior": text(row["prior"]), "depth": text(row["depth_info"]),
                     "answer": num(row["answer"]), "pred": num(row["pred"]), "mra": num(row["mra"], 6),
                     "src": text(row["source"]), "status": text(row["status"]),
                     "method": row["method"], "scene_id": text(row["scene_id"]), "time": None}
            if FMT["decide"] == "conf":
                req(row, "conf", "harness CSV row")    # read as the format's field, not shown (decisions.no_conf)
            donly = False
            if unit_kind == "video":
                answer, group, error = bind_harness_question(row, record["answers"], groups)
                state = group["state"] if group else {}
                result = own_result(state, row) if group else None
                exec_status = record["execution_status"]
            else:
                rec = question_record(tag, unit, row)
                if FMT.get("sceneprog") and text(row["source"]) == "execution_failed" and \
                        req(rec, "execution_status", "answer record") == "complete":
                    # docs/VAL_V6.md "Known gap": a unit retried on the bug-fix commit has a complete answer record,
                    # but the run's CSV, written under the first pass's run signature, lists it execution_failed; the
                    # CSV's row is shown as it is (no harness answer), with the record's timing
                    query |= stale_row_unit(rec, row)
                    query["tq"], query["tjump"] = question_times(query["q"], None)
                    questions.append(query)
                    continue
                if FMT.get("fused") and req(rec, "execution_status", "answer record") == "failed":
                    query |= failed_unit(rec, row)
                    query["tq"], query["tjump"] = question_times(query["q"], None)
                    questions.append(query)
                    continue
                inventory = inventory_view(unit) if FMT.get("classref") else unit_json("objects", unit, "object inventory")
                state = question_scene(unit, row, rec)
                answer, error = req(rec, "answer", "answer record"), None
                if req(answer, "row", "answer") != row["row"] or text(req(answer, "scene_id", "answer")) != text(row["scene_id"]):
                    raise ValueError(f"{unit}: answer row / scene_id differs from the CSV")
                if state is not None and req(state, "scene_id", "question scene") != row["scene_id"]:
                    raise ValueError(f"{unit}: scene_id of its scene differs from the CSV")
                group = {"group": None, "state": state} if state is not None else None
                result = own_result(state, row) if state is not None else None
                donly = state is not None and FMT.get("undetermined") and direct_only_scene(state, answer)
                exec_status = rec["execution_status"]
                query["time"] = timing_view(req(rec, "timing", "answer record"), row.get("seconds"))
            if FMT["decide"] in ("checks", "agree"):
                if unit_kind != "question":
                    raise ValueError(f"{tag}: named checks are read from per-question answer records only")
                query |= answer_checks(answer, row)
            if FMT["decide"] == "agree" and unit_kind == "question" and donly:
                # rule D24 scene.direct_only: no scene, no agree / constraints; the result's measurements as recorded
                query["direct_only"] = direct_only_view(state, result)
            elif FMT["decide"] == "agree":
                if result is None:
                    raise ValueError(f"{unit}: the scene has no result for this question")
                query |= agreement_view(result, answer, query["status"], state)
            quality = req(result, "quality", "scene result") if result else {}
            alignment = align(quality.get("align_tq"))   # absent or null when no alignment check ran
            mid = (result or {}).get("measurement_id")
            if mid is not None and mid not in req(state, "measurements", "scene"):
                raise ValueError(f"{unit}: measurement {mid} not in its scene")
            query |= {"group": group["group"] if group else None, "binding_error": error,
                      "measurement": answer["measurement"] if group else None, "measurement_id": mid,
                      "measurement_record": state["measurements"][mid] if mid is not None else None,
                      "provenance": req(answer, "provenance", "answer record") if answer else None, "quality": quality, "align": alignment,
                      "fit": fit_row(result) if result else None,
                      "overlays": overlays.get(vid, state, row["question"], float(first.fps), duration,
                                               vid if unit_kind == "video" else unit) if group and not donly else [],
                      "execution_status": exec_status}
            if unit_kind == "question":
                query["refmeasure"] = refmeasure_view(tag, unit)
                if FMT.get("classref"):
                    query["inventory"] = inventory
                    query["scale_source"] = None
                    if result is not None and not donly:
                        ss = text(req(result, "scale.source", "scene result"))
                        if ss != camera_scale_source(req(state, "camera", "scene")):
                            raise ValueError(f"{AT['where']}: result scale.source {ss!r} differs from the camera's")
                        query["scale_source"] = ss
                sv = state if not donly else None     # a direct-only scene has no scene to show
                if FMT.get("roll"):
                    query["extents"] = extents_view(tag, unit, sv)
                if FMT.get("static"):
                    query["depth_bindings"] = depth_bindings_view(sv) if sv is not None else None
                query |= {"scene": scene_view(sv, DIRS["scenes"] / f"{unit}.json") if sv is not None else None,
                          "sgraph": sgraph(unit, rec, sv), "tracks": track_intervals(unit)}
                if FMT.get("span"):
                    query["track_ends"] = track_ends(unit)
                    if sv is not None and "measurements" in query:
                        same_depth_rows(state, query["measurements"])
                if FMT.get("kept"):
                    query["rounds"] = rounds_view(rec)
                if FMT.get("contained") and sv is not None:
                    # rule D27: agent_status lists every no-repeat key (always present), as the rounds joined them
                    nr = [no_repeat_view(x, "agent_status") for x in req(req(state, "agent_status", "scene"), NO_REPEAT, "agent_status")]
                    joined = [x for r in (query.get("rounds") or {}).get("rounds", []) for x in r.get("no_repeat", [])]
                    if sorted(map(str, nr)) != sorted(map(str, joined)):
                        raise ValueError(f"{AT['where']}: agent_status {NO_REPEAT} {nr} differs from the rounds' {joined}")
                if FMT.get("identity"):
                    query |= {"identity": identity_view(unit),
                              "dims_check": dims_check(sv) if sv is not None else None}
                if FMT.get("fused"):
                    query["plausible_untestable"] = plausible_untestable(sv) if sv is not None else None
                if FMT.get("sizeonly") and donly:
                    query["plausible_excluded"] = None
                    query["partial"] = partial_view(rec)
                elif FMT.get("sizeonly"):
                    # rule reference.size_only: plausible excludes exactly the scene's references (always present)
                    pl = req(req(rec, "checks", "answer record"), "plausible", "answer record checks")
                    ex = req(pl, "excluded", "plausible")
                    if set(ex) != set(EXCLUDED_KEYS):
                        raise ValueError(f"{AT['where']}: plausible.excluded keys {sorted(ex)}")
                    refs = sorted(req(state, "references", "scene")) if state is not None else []
                    if sorted(ex["reference.size_only"]) != refs:
                        raise ValueError(f"{AT['where']}: plausible.excluded {ex['reference.size_only']} differs from the scene's references {refs}")
                    query["plausible_excluded"] = excluded_view(pl)
                    query["partial"] = partial_view(rec)
                if FMT.get("impulse"):
                    query["presence"] = presence_view(tag, unit)
                if FMT.get("undetermined"):
                    query["floor"] = floor_view(tag, unit)
                    pu = prior_untracked_view(state) if state is not None and not donly else None
                    if donly:
                        c = (result or {}).get("checks", {}).get("prior.untracked")
                        if c is None:
                            pu = None
                        elif FMT.get("contained") and PRIOR_UNTRACKED_CLASS.match(text(c["reason"])):
                            pu = {"object": None, "instances": None, "reason": text(c["reason"])}
                        else:
                            pu = {"object": PRIOR_UNTRACKED.match(text(c["reason"])).group(1), "reason": text(c["reason"])}
                    if pu is not None:
                        if "prior.untracked" not in req(answer, "failed", "answer"):
                            raise ValueError(f"{AT['where']}: prior.untracked is not among the answer's failed checks")
                        if pu["object"] is not None:
                            prior_untracked_repairs(query.get("rounds"), pu["object"])
                    query["prior_untracked"] = pu
            query["tq"], query["tjump"] = question_times(query["q"], alignment)
            questions.append(query)
    overlays.save()
    print(f"{tag}: {len(questions)} questions, {overlays.added} overlays rendered")
    return questions


def direct_only_scene(state: dict, answer: dict) -> bool:
    """A direct-only scene (pipeline.state.direct_only: stage A failed, objects {}, no camera; rule D24
    scene.direct_only): its results each fail "fit" and its answer is unresolved with "fit" among the failed checks."""
    from pipeline.state import direct_only
    if not direct_only(state):
        return False
    res = req(state, "results", "direct-only scene")
    if req(state, "objects", "direct-only scene") or not res or \
            any(req(req(r, "checks", "scene result"), "fit", "scene result checks")["pass"] is not False for r in res):
        raise ValueError(f"{AT['where']}: direct-only scene with objects or a result without a failed fit check")
    if req(answer, "status", "answer") != "unresolved" or "fit" not in req(answer, "failed", "answer") \
            or req(answer, "method", "answer") != "scene-fit-failed":
        raise ValueError(f"{AT['where']}: direct-only answer status {answer.get('status')!r} method {answer.get('method')!r} "
                         f"failed {answer.get('failed')!r}")
    return True


def direct_only_view(state: dict, result: dict | None) -> dict:
    """The stage-A failure of a direct-only scene and the measurements its result recorded: {error, fit, measurements}."""
    if result is None:
        raise ValueError(f"{AT['where']}: direct-only scene without this question's result")
    return {"error": text(req(state, "error", "direct-only scene")),
            "fit": text(req(req(result, "checks", "scene result")["fit"], "reason", "fit")),
            "measurements": [measurement_row(m) for m in req(result, "measurements", "scene result")]}


EXCLUDED_KEYS = ("reference.size_only", "structure.static", "buoyant", "contained")   # pipeline.critic plausible.excluded


def excluded_view(pl: dict) -> list[dict]:
    """The objects every plausible part leaves out, by rule (C15c reference.size_only, D25 critic.structure_static /
    critic.buoyant): [{rule, names, why {name: text}}] for the rules that exclude any. The three keys are always present;
    "excluded_why" may name a reason per object of a rule that excludes some, and only those."""
    ex = req(pl, "excluded", "plausible")
    why = pl.get("excluded_why") or {}
    if set(why) - set(EXCLUDED_KEYS):
        raise ValueError(f"{AT['where']}: plausible.excluded_why keys {sorted(why)}")
    out = []
    for k in EXCLUDED_KEYS:
        names = [text(x) for x in ex[k]]
        w = {text(n): text(t) for n, t in (why.get(k) or {}).items()}
        if set(w) - set(names):
            raise ValueError(f"{AT['where']}: plausible.excluded_why[{k!r}] names {sorted(set(w) - set(names))} not excluded")
        if names:
            out.append({"rule": k, "names": names, "why": w})
    return out


def same_depth_rows(state: dict, ms: list[dict]):
    """A direct depth_agree copied from the scene (see "depth_rows") must be one of the scene's depth_rows records."""
    key = lambda x: (x["pass"], x["n_rows"], x["n_matched"], x["dropped"], x["reason"])
    rows = {key(r) for r in depth_rows_view(state) or []}
    for m in ms:
        a = m.get("depth_agree")
        if a and a.get("see") and key(a) not in rows:
            raise ValueError(f"{AT['where']}: measurement {m['name']!r} direct.depth_agree names depth_rows, no scene row matches")


def build_harness_val(runs: list[tuple[str, Path]], make_thumb: bool = False) -> dict:
    """Display completed harness runs without re-running anything. `runs` is ordered: each question of the
    validation table is shown from the first run whose CSV has its row, and every value of that question comes from
    that run alone (labelled with its version on the page). Every DEV20 question must be in some run."""
    if len({t for t, _ in runs}) != len(runs):
        raise ValueError(f"a run is listed twice: {[t for t, _ in runs]}")
    manifest, dev = manifest_units(), dev_units()
    claimed: dict[int, str] = {}
    frames = {}
    for tag, csv_path in runs:
        frames[tag] = run_rows(tag, csv_path, manifest)
        for i in pd.to_numeric(frames[tag]["row"], errors="raise").astype(int):
            claimed.setdefault(int(i), tag)
    by_unit = {u: i for i, u in enumerate(manifest["unit"])}
    missing = [u for u in dev if by_unit[u] not in claimed]
    if missing:
        raise ValueError(f"docs/DEV20.md units in none of the runs: {missing}")
    videos, questions = {}, []
    for tag, _ in runs:
        units = [u for i, u in enumerate(manifest["unit"]) if claimed.get(i) == tag]
        if not units:
            raise ValueError(f"{tag}: every question of this run is taken by an earlier run")
        questions += build_run(tag, frames[tag], manifest, units, videos, make_thumb)
    questions.sort(key=lambda q: q["row"])
    print(f"val: {len(questions)} questions ({len(dev)} DEV20), {len(videos)} videos")
    return {"runs": [{"version": t, "unit": version_format(t)["unit"], "decide": version_format(t)["decide"],
                      "csv": str(p.relative_to(ROOT)), "unit_set": unit_set(t, frames[t], p)} for t, p in runs],
            "dev": dev, "videos": [videos[v] for v in sorted(videos)], "questions": questions}


def unit_set(tag: str, frame: pd.DataFrame, csv_path: Path):
    """The unit set a run's CSV names (the harness names its CSV by the set: "_dev20"; the full split has no suffix);
    None for formats without it. One value per CSV, and the file name must carry it."""
    if not version_format(tag).get("static"):
        return None
    vals = sorted({text(x) for x in req(frame, "unit_set", f"{csv_path.name}")})
    if len(vals) != 1 or not vals[0] or \
            not csv_path.stem.endswith("_answers" if vals[0] == "full" else f"_answers_{vals[0]}"):
        raise ValueError(f"{csv_path.name}: unit_set {vals} does not name this CSV")
    return vals[0]


# ---------- the scene-program chain of a v6 run (docs/SCENE_PROGRAM.md, docs/CHAIN_NO_LOOP.md) ----------
# masks -> build -> fit -> answer: the answer step reads the fit's directory outputs/sceneprog/<BUILD>/<unit>/; its
# answers are one JSON {unit: {final, unit, scene_value, sigma, determined, seconds, reason, ...}} beside a directory
# of the same stem holding each unit's answer record (<unit>.json, schema sceneprog-answer/1)
SP_ANSWER_KEYS = {"final", "unit", "scene_value", "sigma", "determined", "prompt_tokens", "completion_tokens", "calls",
                  "memo_hit", "seconds", "reason", "inputs_cited"}
SP_MEDIA = "media/sceneprog"          # git-ignored symlinks media/sceneprog/<tag> -> outputs/sceneprog/<tag>
SP_RAW: dict[str, tuple] = {}         # unit -> (answer_final, answer_scene or None) unrounded, for comparison_ours


def sp_media(tag: str, unit: str, name: str) -> str:
    """The page path of one scene-program product; the file must exist behind the media symlink."""
    p = WEB / SP_MEDIA / tag / unit / name
    if not p.is_file():
        raise ValueError(f"{unit}: {SP_MEDIA}/{tag}/{unit}/{name} missing (symlink {SP_MEDIA}/{tag} -> outputs/sceneprog/{tag})")
    return f"{SP_MEDIA}/{tag}/{unit}/{name}"


def sp_mra(pred, truth) -> float:
    """MRA of one answer: the baselines' rule (tools/build_baseline.py mra), so our row and theirs are scored alike."""
    from build_baseline import mra
    return mra(None if pred is None else float(pred), float(truth))


def sceneprog_answers(path: Path) -> dict:
    """The answer step's table {unit: entry} (every entry with exactly SP_ANSWER_KEYS) and its records directory."""
    j = read_json(path, "scene-program answers")
    for u, e in j.items():
        if set(e) != SP_ANSWER_KEYS:
            raise ValueError(f"{path.name}: {u} keys {sorted(e)}")
    rec_dir = path.with_suffix("")
    if not rec_dir.is_dir():
        raise ValueError(f"{path.name}: no records directory {rec_dir}")
    return j


def sceneprog_view(build: str, rec_dir: Path, q: dict, e: dict) -> dict:
    """One question's scene-program products: the two answers (answer_scene, the strict read-out of the fitted program,
    undetermined = no value; answer_final, the answer step's, the reported one) with their MRA against the question's
    ground truth, the fit's status, the time of each chain step (build: plan.json timing; fit: fit.json time_s;
    answer: the answer table's seconds), and the renders of the build tag: the fitted program's overlays, the plan
    grid and the shapes."""
    unit = q["id"]
    AT["where"] = f"{build} {unit}"
    bd = ROOT / "outputs" / "sceneprog" / build / unit
    a = read_json(rec_dir / f"{unit}.json", "scene-program answer")
    if req(a, "schema", "scene-program answer") != "sceneprog-answer/1" or a["unit"] != unit:
        raise ValueError(f"{unit}: answer record schema / unit {a.get('schema')!r} {a.get('unit')!r}")
    sc, fi = req(a, "answer_scene", "answer record"), req(a, "answer_final", "answer record")
    det = bool(e["determined"])
    if det != bool(req(sc, "determined", "answer_scene")) or num(fi.get("value")) != num(e["final"]) or \
            (det and num(sc.get("value")) != num(e["scene_value"])):
        raise ValueError(f"{unit}: answer table and record differ (final {e['final']!r} / {fi.get('value')!r})")
    fit = read_json(bd / "fit.json", "fit")
    plan = read_json(bd / "plan.json", "plan")
    overlays = sorted(p.name for p in bd.glob("overlay_f*.jpg"))
    if not overlays:
        raise ValueError(f"{unit}: no overlay_f*.jpg in {bd.relative_to(ROOT)}")
    scene = num(e["scene_value"]) if det else None
    SP_RAW[unit] = (e["final"], e["scene_value"] if det else None)
    return {"scene": scene, "sigma": num(e["sigma"], 3) if det else None, "determined": det,
            "reason": text(sc.get("reason")) or None, "scene_mra": sp_mra(e["scene_value"] if det else None, q["answer"]),
            "final": num(e["final"]), "final_mra": sp_mra(e["final"], q["answer"]),
            "unit": text(e["unit"]), "rationale": text(fi.get("rationale") or e["reason"]) or None,
            "fit_status": text(req(fit, "status", "fit")),
            "time": {"build": num(req(req(plan, "timing", "plan"), "total_s", "plan timing"), 4),
                     "fit": num(req(fit, "time_s", "fit"), 4), "answer": num(e["seconds"], 4)},
            "renders": {"overlays": [sp_media(build, unit, n) for n in overlays],
                        "plan_grid": sp_media(build, unit, "plan_grid.jpg"), "shapes": sp_media(build, unit, "shapes.png")}}


def add_sceneprog(questions: list[dict], harness: str, build: str, answers: Path) -> dict:
    """Attach the scene-program products of build tag `build` and the answer table `answers` to every question of
    harness run `harness` ("sp"); every question of the run must have an answer entry and no other unit may.
    Returns the run's {"harness", "build", "answers"} record."""
    table = sceneprog_answers(answers)
    mine = [q for q in questions if q["version"] == harness]
    if set(table) != {q["id"] for q in mine}:
        raise ValueError(f"{answers.name}: units differ from the {harness} run: "
                         f"{sorted(set(table) ^ {q['id'] for q in mine})}")
    for q in mine:
        q["sp"] = sceneprog_view(build, answers.with_suffix(""), q, table[q["id"]])
    return {"harness": harness, "build": build, "answers": answers.name}


OURS_JSON = WEB / "data" / "comparison_ours.json"


def write_comparison_ours(questions: list[dict], sp: dict) -> int:
    """data/comparison_ours.json, our row of the method comparison (index.html, baseline.html): every question of the
    baselines' set (data/baseline.json) with its category and both MRAs of the scene-program run against that set's
    ground truth (the baselines' rule, sp_mra); the pages compute the scores with the baselines' own rule. Each baseline question must be one of ours."""
    base = read_json(WEB / "data" / "baseline.json", "baseline data")
    byid = {q["id"]: q for q in questions}
    out = {}
    for b in req(base, "questions", "baseline data"):
        q = byid.get(b["id"])
        if q is None or "sp" not in q or q["cat"] != b["cat"]:
            raise ValueError(f"baseline question {b['id']} not among the scene-program questions (or another category)")
        final, scene = SP_RAW[b["id"]]
        out[b["id"]] = {"cat": q["cat"], "final": q["sp"]["final"], "final_mra": sp_mra(final, b["answer"]),
                        "scene_mra": sp_mra(scene, b["answer"])}
    OURS_JSON.write_text(json.dumps({"label": "Ours", "version": base_version(sp["harness"]),
                                     "run": [sp["harness"], sp["build"], sp["answers"]], "questions": out},
                                    ensure_ascii=False, separators=(",", ":")))
    return len(out)


PIPELINE_HTML = WEB / "pipeline.html"
CLASS_TABLE_MARKS = ("<!-- class_sizes: generated by tools/build_videos.py from pipeline/class_sizes.py -->",
                     "<!-- /class_sizes -->")


def write_class_table() -> int:
    """Regenerate the class-size table of pipeline.html (scene program section, rule C15) from pipeline.class_sizes itself, between
    CLASS_TABLE_MARKS: one row per class with what it is, every dim's range lo-hi (m) and sigma, the primary dim, role,
    observe and source, under the module's TABLE_VERSION. Nothing is copied by hand; the marks must be there."""
    from html import escape
    from pipeline.class_sizes import CLASSES, PART_DIMS, size_reference
    page = PIPELINE_HTML.read_text()
    a, b = CLASS_TABLE_MARKS
    if page.count(a) != 1 or page.count(b) != 1 or page.index(a) > page.index(b):
        raise ValueError(f"{PIPELINE_HTML.name}: class-size table marks missing or out of order")
    g = lambda x: f"{x:g}"
    rows = []
    for key, c in CLASSES.items():
        ref = size_reference(key)
        dims = "".join(f'<span class="dim">{escape(d["dim"].replace("_", " "))} {g(d["lo"])}–{g(d["hi"])}'
                       f' <span class="muted">σ {g(round(d["sigma"], 6))}</span></span>' for d in ref["dims"])
        rows.append(f'  <tr><td><code>{escape(key)}</code></td><td>{escape(c["what"])}</td><td class="dims">{dims}</td>'
                    f'<td>{escape(c["primary"].replace("_", " "))}{" (part)" if c["primary"] in PART_DIMS else ""}</td>'
                    f'<td>{escape(c["role"])}</td><td>{escape(c["observe"].replace("_", " "))}</td>'
                    f'<td class="small">{escape(c["source"])}</td></tr>')
    # data-i18n: the UI text is translated in i18n.js (the class descriptions and sources are data and stay English)
    table = (f'\n<details class="classtable"><summary><code>pipeline/class_sizes.py</code> · '
             f'<span data-i18n="pl.classes" data-i18n-args=\'{{"n": {len(CLASSES)}}}\'>{len(CLASSES)} classes</span>'
             f'</summary>\n<div class="scroll-x"><table class="small classsizes">\n'
             '  <tr><th data-i18n>Class</th><th data-i18n>What</th><th data-i18n>Dims (m)</th><th data-i18n>Primary</th>'
             '<th data-i18n>Role</th><th data-i18n>Observe</th><th data-i18n>Source</th></tr>\n'
             + "\n".join(rows) + '\n</table></div>\n</details>\n')
    new = page[:page.index(a) + len(a)] + table + page[page.index(b):]
    if new != page:
        PIPELINE_HTML.write_text(new)
    return len(CLASSES)


TIME_TABLE_MARKS = ("<!-- step_times: generated by tools/build_videos.py from the answer records of the current run -->",
                    "<!-- /step_times -->")
STEP_LABELS = {"objects": "1 Objects", "scenegraph_grid": "2 Scene graph (grid)", "observe": "3 Observe", "track": "4 Track",
               "measure": "5 Measure", "scenegraph_pf": "6 Scene graph (per frame)", "ground": "7 Ground",
               "scene": "8 Scene", "agent": "9 Agent loop", "answer": "10 Answer"}
# the steps of the run's records that the harness still runs (pipeline.harness CHAIN_TRACK), as pipeline.html names them;
# the run predates the task plan and focal steps, so they have no timing in its records
SHOWN_STEPS = {"objects": "1 Objects", "scenegraph_grid": "3 Scene graph", "track": "4 Track"}


def write_time_table(questions: list[dict], tag: str) -> int:
    """Regenerate the time-per-step table of pipeline.html between TIME_TABLE_MARKS from the answer records' timing of
    run `tag` (outputs/harness/<tag>/answers/<unit>.json, "timing"): median / min / max seconds of each step the harness
    still runs (SHOWN_STEPS), over the questions whose record ran every step (a failed chain's prefix is left out). Read from the records
    themselves, not the run's CSV (docs/VAL_V6.md: the CSV lists four retried units as failed). Returns the number."""
    page = PIPELINE_HTML.read_text()
    a, b = TIME_TABLE_MARKS
    if page.count(a) != 1 or page.count(b) != 1 or page.index(a) > page.index(b):
        raise ValueError(f"{PIPELINE_HTML.name}: step-time table marks missing or out of order")
    FMT.clear()
    FMT.update(version_format(tag))
    qs = []
    for q in questions:
        if q["version"] != tag:
            continue
        AT["where"] = f"{tag} {q['id']}"
        rec = read_json(ROOT / "outputs/harness" / tag / "answers" / f"{q['id']}.json", "answer record")
        if req(rec, "execution_status", "answer record") == "failed":
            continue
        t = req(rec, "timing", "answer record")
        qs.append({"time": timing_view(t, req(t, "total_seconds", "timing"))})
    if not qs:
        raise ValueError(f"{tag}: no question ran every step")
    steps = [x["step"] for x in qs[0]["time"]["steps"]]
    if list(STEP_LABELS) != steps or any([x["step"] for x in q["time"]["steps"]] != steps for q in qs):
        raise ValueError(f"{tag}: timing steps {steps}, expected {list(STEP_LABELS)}")
    cells = lambda v: "".join(f'<td class="n">{x:.1f}</td>' for x in (float(np.median(v)), min(v), max(v)))
    rows = [f'  <tr><td data-i18n>{SHOWN_STEPS[k]}</td><td><code>{k}</code></td>{cells([q["time"]["steps"][i]["s"] for q in qs])}</tr>'
            for i, k in enumerate(steps) if k in SHOWN_STEPS]
    n_all = len([q for q in questions if q["version"] == tag])
    table = ('\n<div class="scroll-x"><table class="tt">\n'
             '  <tr><th data-i18n>Step</th><th data-i18n>Timing key</th><th class="n" data-i18n>Median (s)</th>'
             '<th class="n" data-i18n>Min</th><th class="n" data-i18n>Max</th></tr>\n'
             + "\n".join(rows) + '\n</table></div>\n'
             f'<p class="muted small" data-i18n="pl.time.ran" data-i18n-args=\'{{"n": {len(qs)}, "all": {n_all}}}\'>'
             f'The {len(qs)} of the {n_all} questions that ran every step '
             f'(<code>timing.steps</code> of their answer records).</p>\n')
    new = page[:page.index(a) + len(a)] + table + page[page.index(b):]
    if new != page:
        PIPELINE_HTML.write_text(new)
    return len(qs)


def parse_run(spec: str) -> tuple[str, Path]:
    """<tag> (CSV outputs/val_<tag>_answers.csv) or <tag>=<CSV path relative to the repo root>."""
    tag, _, csv = spec.partition("=")
    return tag, (ROOT / csv if csv else ROOT / "outputs" / f"val_{tag}_answers.csv").resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-runs", nargs="+", default=[CURRENT], metavar="TAG[=CSV]",
                    help="completed harness validation runs, in order of preference: a question is shown from the "
                         "first run that has it (default: pipeline/VERSION, %(default)s)")
    ap.add_argument("--sceneprog", metavar="BUILD",
                    help="scene-program tag of the first run (outputs/sceneprog/<BUILD>: programs, fits, overlays); "
                         "required when its version has the scene program")
    ap.add_argument("--answers", type=Path, metavar="JSON",
                    help="the answer step's table on those programs ({unit: entry}, records in the directory of the "
                         "same stem); required with --sceneprog")
    ap.add_argument("--rethumb", action="store_true", help="regenerate thumbnails that already exist")
    args = ap.parse_args()
    THUMBS.mkdir(parents=True, exist_ok=True)
    runs = [parse_run(s) for s in args.val_runs]
    has_sp = [t for t, _ in runs if version_format(t).get("sceneprog")]
    if has_sp and (has_sp != [runs[0][0]] or not args.sceneprog or not args.answers):
        raise ValueError(f"runs with the scene program {has_sp}: only the first run, and --sceneprog BUILD --answers JSON are required")
    if args.sceneprog and not has_sp:
        raise ValueError("--sceneprog given, but the first run's version has no scene program")
    data = {"generated": date.today().isoformat(), "val": build_harness_val(runs, args.rethumb)}
    if has_sp:
        data["val"]["sceneprog"] = add_sceneprog(data["val"]["questions"], runs[0][0], args.sceneprog, args.answers.resolve())
        print(f"comparison_ours.json: {write_comparison_ours(data['val']['questions'], data['val']['sceneprog'])} baseline questions")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    used = {Path(v["thumb"]).name for v in data["val"]["videos"]}
    stale = [p for p in THUMBS.glob("*.jpg") if p.name not in used]
    for p in stale:
        p.unlink()
    if stale:
        print(f"{len(stale)} unreferenced thumbs removed")
    print(f"pipeline.html: class-size table, {write_class_table()} classes")
    first = data["val"]["runs"][0]["version"]
    print(f"pipeline.html: step-time table, {write_time_table(data['val']['questions'], first)} {first} questions")
    sizes = [p.stat().st_size for p in THUMBS.glob("*.jpg")]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); {len(sizes)} thumbs, {sum(sizes) / 1e6:.2f} MB, max {max(sizes, default=0)} B")


if __name__ == "__main__":
    main()
