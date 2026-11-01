import gzip
import json

from predictions_io import (
    dump_line,
    expand_frame,
    find_predictions,
    frame_record,
    iter_frames,
    pack,
    read_meta,
    read_predictions,
)


def ball(frame, x):
    return {"frame_index": frame, "class_name": "ball", "score": 0.9, "bbox_xyxy": [x, 10.0, x + 10.0, 20.0]}


def player(frame, track_id, interpolated=False, **extra):
    return {
        "class_name": "player",
        "score": 0.8,
        "bbox_xyxy": [1.0, 2.0, 30.0, 90.0],
        "track_id": track_id,
        "interpolated": interpolated,
        "frame_index": frame,
        **extra,
    }


def write_jsonl(path, frames, end=True, torn=""):
    lines = [dump_line({"type": "header", "format": "ravel-vb-predictions-v3", "fps": 30.0, "frame_step": 2})]
    for frame in frames:
        lines.append(dump_line(frame_record(frame, [player(frame, 3), ball(frame, float(frame))])))
        if frame == 4:
            lines.append(dump_line({"type": "checkpoint", "next_frame": 6}))
    if end:
        lines.append(dump_line({"type": "end", "last_frame": frames[-1] + 1, "processed_frames": frames[-1] + 2, "benchmark": {"processed_frames": frames[-1] + 2}}))
    path.write_text("".join(lines) + torn)


def test_a_score_is_cut_down_so_it_stays_on_its_side_of_a_threshold():
    line = dump_line(frame_record(0, [{**ball(0, 1.0), "score": 0.34999}, {**player(0, 1), "score": 0.35001}]))
    record = json.loads(line)
    assert record["ball"][0] == 0.349 and record["players"][0][1] == 0.35


def test_a_frame_line_holds_rows_and_reads_back_as_the_detector_dicts():
    detector = [
        {**player(8, 3), "class_id": 0, "query_index": 5, "time_sec": 0.27, "score": 0.76580035, "bbox_xyxy": [372.531, 281.18, 407.49, 371.18]},
        {**player(8, 4, interpolated=True), "class_id": 0, "query_index": None, "time_sec": 0.27},
        {**player(8, 5), "head_point_norm": [0.5, 0.25]},
        {**ball(8, 50.0), "class_id": 1, "query_index": None, "time_sec": 0.27, "score": 0.35217735},
        {"class_name": "head", "score": 0.4, "bbox_xyxy": [1.0, 1.0, 2.0, 2.0], "frame_index": 8, "time_sec": 0.27},
    ]
    line = dump_line(frame_record(8, detector))
    # Scores are cut down (0.7658 -> 0.765), boxes kept to 0.01 px.
    assert line.startswith('{"type":"frame","frame_index":8,"players":[[3,0.765,372.53,281.18,407.49,371.18],[4,0.8,1.0,2.0,30.0,90.0,1]')
    assert '"ball":[0.352,50.0,10.0,60.0,20.0]' in line

    items = expand_frame(json.loads(line))
    assert items[0] == {"class_name": "player", "score": 0.765, "bbox_xyxy": [372.53, 281.18, 407.49, 371.18], "track_id": 3, "interpolated": False, "frame_index": 8}
    assert items[1]["interpolated"] is True
    assert items[2]["head_point_norm"] == [0.5, 0.25] and items[2]["interpolated"] is False
    assert items[3] == {"class_name": "ball", "score": 0.352, "bbox_xyxy": [50.0, 10.0, 60.0, 20.0], "frame_index": 8}
    assert items[4] == {"class_name": "head", "score": 0.4, "bbox_xyxy": [1.0, 1.0, 2.0, 2.0], "frame_index": 8}


def test_a_growing_jsonl_reads_up_to_its_last_whole_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [0, 2, 4, 6, 8], end=False, torn='{"type":"frame","frame_index":10,"play')

    meta = read_meta(path)
    assert meta["complete"] is False and meta["last_frame"] == 8 and meta["fps"] == 30.0
    assert [frame for frame, _ in iter_frames(path)] == [0, 2, 4, 6, 8]
    assert [frame for frame, _ in iter_frames(path, 3, 6)] == [4, 6]


def test_the_end_line_marks_the_last_frame_and_the_packed_file_reads_the_same(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [0, 2, 4])
    meta = read_meta(path)
    assert meta["complete"] is True and meta["last_frame"] == 5 and meta["processed_frames"] == 6
    assert meta["benchmark"] == {"processed_frames": 6}
    plain_frames = list(iter_frames(path))

    packed = pack(path)
    assert packed.name == "predictions.jsonl.gz" and not path.exists()
    assert find_predictions(tmp_path) == packed
    assert read_meta(packed) == meta
    assert list(iter_frames(packed)) == plain_frames
    assert [frame for frame, _ in iter_frames(packed, 2, 2)] == [2]


def test_the_older_formats_read_the_same(tmp_path):
    predictions = [ball(0, 0.0), ball(2, 2.0), ball(4, 4.0)]
    (tmp_path / "predictions.json").write_text(json.dumps({"fps": 25.0, "frame_step": 2, "predictions": predictions}))
    path = find_predictions(tmp_path)
    assert path.name == "predictions.json"

    meta, items = read_predictions(path, frame_start=2)
    assert meta["complete"] is True and meta["fps"] == 25.0
    assert items == predictions[1:]

    # v2: the dicts of a frame as they are, spaces after the separators.
    v2 = tmp_path / "predictions.jsonl"
    v2.write_text("".join(json.dumps({"type": "frame", "frame_index": p["frame_index"], "predictions": [p]}) + "\n" for p in predictions))
    assert find_predictions(tmp_path) == v2
    assert read_predictions(v2, frame_start=1)[1] == predictions[1:]
