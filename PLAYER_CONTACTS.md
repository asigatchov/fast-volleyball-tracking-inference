# Player-aware rally analysis

`src/track_calculator_with_court.py` can combine the ball trajectory with player
detections. This separates real play from a player simply passing the ball over
for the next serve, and labels every ball touch with the technique used.

```bash
uv run src/track_calculator_with_court.py \
  --csv_path ../uploads/mix/beach-mixt_predict_ball.csv \
  --court_json_path ../uploads/mix/beach-mixt_court.json \
  --players_json_path ../uploads/mix/beach-mixt_predictions.json \
  --beach --fps 29.97 --output_dir output
```

Without `--players_json_path` the pipeline behaves exactly as before.

`--csv_path` may be omitted: the ball is then taken from the `ball` detections
of `--players_json_path`, and no ball CSV is needed at all. At least one of the
two paths must be given.

```bash
uv run src/track_calculator_with_court.py \
  --court_json_path ../uploads/mix/beach-mixt_court.json \
  --players_json_path ../uploads/mix/beach-mixt_predictions.json \
  --beach --fps 29.97 --output_dir output
```

The output directory is named after the source file either way:
`beach-mixt_predictions.json` and `beach-mixt_predict_ball.csv` both write to
`output/beach-mixt/tracks`.

The players file is what `src/inference_player_ball_openvino.py` (or its ONNX
twin `src/inference_player_ball_onnx.py`) writes: the streamed
`predictions.jsonl` / `predictions.jsonl.gz` (`ravel-vb-predictions-v3`, the
pipeline's format) or the older single `ravel-vb-predictions-v1` JSON; both are
read through `src/predictions_io.py`. Each player carries `score`, `track_id`,
`frame_index` and `bbox_xyxy`. Boxes are rescaled if the prediction
`source_size` differs from the video size used for the court, and detections
whose feet fall far outside the court (spectators, referees) are dropped.
`--player_score_threshold` (default 0.30) is the minimum score of a player box
used here.

## The ball from the players model

That file also carries a `ball` class, and the two detectors miss different
frames. The CSV is the primary source; the players-model ball is used to

* **fill dropouts** - a frame with no CSV detection takes the ball from the
  players model, but only when the CSV has detections within 8 frames on both
  sides. A lone detection during a pause would otherwise stitch a technical
  return to the rally that follows it.
* **repair outliers** - when both detectors see the ball but the CSV point is
  more than 35 px away from what the neighboring frames predict, and the other
  detection is at least twice as close, the CSV point is replaced.

Every row records where it came from in `BallSource` (`csv`, `players_fill`,
`players_repair`). `--no_ball_fill` disables this, `--ball_score_threshold`
sets the minimum detection score (default 0.35).

This matters for segmentation: a dropout right at a touch splits one rally in
two, because the ball reverses direction while it is invisible and the tracker
cannot bridge that. On `beach-mixt` this fills 295 frames and repairs 74, and
the two halves of frames 9041-9350 become one track again.

Two more player-aware steps run on top:

* **contact merge** - consecutive tracks separated by up to 0.5 s, ending and
  starting close together, with a player within reach at the seam, are merged:
  the touch explains the gap. This is the fallback when neither detector saw
  the ball across the seam.
* **blind-gap split** - if *neither* detector sees the ball for 0.8 s inside a
  track, the ball is not in flight (it is being carried to the serve, or the
  rally is over) and the track is cut there. Without it, the 40-frame tolerance
  of the tracker stretches an episode across a pause.

## What counts as a contact

A touch is a break in the flight path, not just a ball near a player. For every
ball sample the velocity is fitted before and after, and gravity is subtracted:
free flight only changes the vertical velocity by `g * dt`. What remains is the
velocity a player imparted, so a ball flying over a player's head or through the
apex of its own parabola produces no contact.

A break becomes a contact when a player is within reach - an ellipse around the
box covering arms up, sideways and dive reach. When several players qualify, the
one closest to their own body wins, which keeps a large near-camera box from
swallowing touches that belong to a smaller, more distant player. How fast the
player closes on the ball (`approach_px_frame`, measured against a fixed ball
position so only the player's own movement counts) raises the confidence.

Breaks are picked by non-maximum suppression inside a 0.3 s window: the
strongest first, then everything close to it is dropped. Chaining candidate to
candidate instead lets one noisy stretch grow into a single group that swallows
the real touches inside it.

## Techniques

The ball height at the contact is measured against the player's **standing**
height, tracked as a moving quantile of their box height. A crouching player has
a short box, so the raw box top would make a chest-high dig look like a ball
played above the head.

| Ball height (0.0 = top of head, 1.0 = feet) | Type            | Meaning              |
| ------------------------------------------ | --------------- | -------------------- |
| above the head                              | `attack`, `block`, `serve` | played above the head |
| 0.0 .. 0.25                                 | `overhead_pass` | pass sverkhu         |
| 0.25 .. 0.95                                | `dig`           | priyom snizu         |
| below the feet                              | `low_touch`     | ball at ground level |

Posture is the second cue. A box wider than 0.65 of its height means a lunge, a
bend or a dive - an upright player sits around 0.35-0.5 - so the ball is played
from below even at head height, and the `overhead_pass` band shrinks to nothing.
Each contact reports its `posture_ratio` next to `depth_ratio`.

`block` is a contact above the head next to the net.

The serve is not decided by height at all - a jump serve is struck above the
head, a float off a toss at head height, an underhand one at the waist. It is
decided by context: the contact must belong to the very player the trajectory
departs from, in the serving zone, within a second of the ball leaving their
hands, and it must send the ball away faster than it arrived. Of the candidates
in that window the one with the biggest speed gain wins, so the strike is picked
and not the toss that precedes it. A ball that is already flying fast when it
reaches a player standing deep is a rally touch, not a serve.

## Who touches, and how many times

Touches are grouped into possessions - runs of touches by one side. A team may
keep the ball for three touches before it has to cross, and those passes are
high and slow while the ball stays on their half, so that rule also settles the
attribution: near and far players project onto the same image region, and when
two of them are almost equally plausible for a touch (within 1.6x of the
ranking cost), the side that keeps the rally legal wins - continue the
possession while it has touches left, switch once it is spent or right after a
serve, which always hands the ball over.

The summary carries `possessions` (side, touches, frames, players, techniques),
`touches_by_player`, `max_touches_in_possession` and
`over_three_touch_possessions`. The last one is a diagnostic: a possession of
four or more means a touch was attributed to the wrong side or a crossing was
missed. The viewer prints the sequence compactly, e.g. `touches f1-n2-f3-n2`.

## Where the ball comes from and where it goes

Both ends of a track are attached to the nearest reachable player, which gives
`player_interaction.ball_start` and `player_interaction.ball_end` with an
`origin`:

* `serve_zone` - the player stands on or behind the end line: a serve, so the
  episode is live play.
* `in_court` - the ball starts at a player standing inside the court. With few
  contacts and no exchange between the sides, this is a handover to the opponent
  for the next serve, i.e. `not_rally / technical_return`.
* `off_court`, `unattached` - no reliable origin, e.g. a fragment of a longer
  rally. Only negative evidence (many contacts, exchanges between sides) is used
  for such tracks, so real rally fragments are not mistaken for handovers.

Before a serve the ball is passed over, held and tossed - all of it within
somebody's reach. `rally_start_frame` marks where that handling ends and the
ball goes up for the serve, with `preparation_sec` measuring what came before;
the serve strike itself follows within a second. A trajectory that starts in
flight keeps its own start.

The end is read the same way: a rally ends with the ball on the sand or out of
frame, while a handover ends **in somebody's hands** in the middle of the court.
So a track that ends at a player inside the court with at most two contacts and
no exchange between the sides is a handover - the ball was thrown in, caught and
passed on to the server.

The serving side is taken from this origin first and only then from a serve
contact - a player receiving a serve deep in their own court also stands in the
serving zone. `trajectory_analysis.serve_side_source` records which rule fired.

## Output

Each `track_*.json` gains a `player_interaction` block:

```json
{
  "enabled": true,
  "ball_start": {"frame": 11320, "origin": "serve_zone", "side": "near", "player_track_id": 496},
  "ball_end": {"frame": 11600, "origin": "in_court", "side": "far", "player_track_id": 501},
  "contacts": [
    {"frame": 11386, "player_track_id": 501, "contact_type": "dig", "depth_ratio": 0.27,
     "side": "far", "in_serve_zone": false, "ball_above_net": false, "confidence": 0.93}
  ],
  "summary": {"contact_count": 5, "dig_count": 2, "overhead_count": 2, "attack_count": 1,
              "side_switch_count": 2, "starts_from_serve_zone": true, "serve_side": "near"}
}
```

## Reviewing the result

`src/show_rally.py` plays the tracks over the source video with the ball path,
player boxes, the start marker and every contact label, frame by frame:

```bash
uv run src/show_rally.py output/beach-mixt/tracks /path/to/video.mp4 \
  --players_json_path ../uploads/mix/beach-mixt_predictions.json \
  --court_json_path ../uploads/mix/beach-mixt_court.json
```

`space` play/pause, `a`/`d` one frame back/forward, `w`/`s` 15 frames,
`n`/`p` next/previous track, `b` boxes, `t` path, `h` help, `q` quit. The
timeline under the frame shows the track span, the contacts, the rally start and
where the cursor is. `--snapshot FILE` renders a single frame without a display.

`v` switches between the video and the schematic view: a blank frame with only what the analysis
works from - the court and net from the court JSON, the player boxes with their
ids and ground points, the ball and its path, and the contacts. It shows the
scene exactly as the decision code sees it, without the video underneath.

The summary is merged into `rally_features` as well, and feeds the rally score:
exchanges between the sides, several contacts and a serve start argue for a
rally; a start from a player inside the court with almost no contacts and no
exchange argues for a technical return. Run with `--verbose` for a per-track
breakdown.

## Action type of a contact: `src/classify_contacts.py`

The height-based `contact_type` above is geometry only. The pipeline
(`src/celery_worker.py`: detector -> `track_calculator_with_court.py` ->
`classify_contacts.py`) then labels every contact with a clip classifier: around
the contact frame a window of frames is cropped at the player closest to the
ball and classified as `Serve` / `Receive` / `Set` / `Attack` / `NoAction`. The
result goes back into the contact as `action: {type, score, probabilities}`.

```bash
uv run src/classify_contacts.py video.mp4 \
  --players_json_path ../uploads/mix/beach-mixt/predictions.jsonl.gz \
  --tracks_dir ../uploads/mix/beach-mixt/tracks \
  --model ov/action_clf_r2plus1d18_verified.xml
```

The classifier runs on OpenVINO (CPU). The worker picks the model by name
(`_resolve_action_model_xml`); `ov/action_clf_r2plus1d18_verified.xml` is the
script default.

## Touches from the detector: RAVEL-VB-013 (players + touch)

Everything above finds a touch **from the ball**: a break in its flight path
with a player within reach. RAVEL-VB-013 finds it **from the players**: it is
RAVEL-VB-012 (VB7 yolo26n, 640x360 RGB, 9-frame clip: players, ball, heads)
with a touch head on top, so every player box of every frame gets the
probability "this player touches the ball now". One release, two formats:

| file | runtime | script |
|---|---|---|
| `ov/RAVEL-VB-013-9f.xml` | OpenVINO, FP32 | `src/inference_player_ball_openvino.py` |
| `models/RAVEL-VB-013-9f.onnx` | ONNX Runtime (CUDA / CPU), FP32 | `src/inference_player_ball_onnx.py` |

```bash
uv run src/inference_player_ball_openvino.py video.mp4 \
  --model ov/RAVEL-VB-013-9f.xml --action-threshold 0.5 \
  --output ../uploads/mix/video/predictions.jsonl
```

**What it writes.** A player whose touch probability reaches
`--action-threshold` (default 0.5) gets `"action": "Touch"` and
`"action_score"`; in the JSONL they ride in the extra dict of the player row:

```json
{"type":"frame","frame_index":30,"players":[[3,0.952,1769.51,607.29,1964.16,1120.63,0,{"head_point_norm":[0.695,0.433],"action":"Touch","action_score":0.991}], ...],"ball":[0.423,1895.53,427.85,1952.29,484.61]}
```

Boxes the hysteresis carries over a gap (`interpolated`) never carry an
action. Any other release writes exactly what it wrote before. Behind the
fields are two model outputs: `action_logits` `[B, T, K, 2]` (log-probabilities
of `Touch` / `none`, the names are `action_names` in the sidecar `.json`) and
`action_actor_logits` `[B, T, K + 1]` (the players of a frame compete for the
role of the one touching; the last channel is "nobody").

**A touch is a peak, not a frame.** The probability rises over several frames
around the contact. On `serve_0001` (contact at frame 30) the server is above
0.5 on frames 18-33, peaking at 0.99 on frames 30-31. The event is the maximum
of `action_score` along a `track_id`; take peaks at least ~0.5 s apart.

**How good it is** (epoch 45 of
`tape_vb7_yolo_v41mask_touch_640_9f_60ep_20261101_064742`):

- validation of the training set: touch precision 0.66, recall 0.90; players
  mAP75 0.90;
- portal match 4bbc5a55 (СГЦ - Динамо, 636 portal touches, threshold 0.7,
  +-6 frames): 451 matched (359 on the same player, 92 on another), the frame
  is within 2 of the portal one in 82% of them (median 1). 185 portal touches are
  missed, 258 touches are the model's only, and 123 of those lie 7-20 frames
  after a real touch: a double peak, so merge peaks of one player closer than
  ~20 frames. Outside rallies (warm-up) it fires too, about 360 events on that
  match;
- the touch training costs the detector a little: on a different camera
  (pro-test of beach-test) detection mAP75 is 0.64 against 0.75 for the same
  network without the touch head.

**Limits.**
- It was trained and checked on consecutive frames (`--frame-step 1`). The
  pipeline runs the detector with `INFERENCE_FRAME_STEP=2` by default, which
  doubles the time a clip spans. Touches at `--frame-step 2` are not checked.
- The touch is not wired into the analysis yet: `player_contacts.py` still
  finds contacts from the ball path and ignores `action`. Natural uses are
  confirming a trajectory break with a model touch of the same player within a
  few frames, choosing between near and far players when the reach cost is
  ambiguous, and recovering touches where the ball was not seen.
- The production detector is still RAVEL-VB-011 (`INFERENCE_MODEL_XML`).
