"""
标注工具的单元测试。

运行:  python tests/test_annotate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.data.annotate import (  # noqa: E402
    ActionAnnotation,
    VideoAnnotation,
    export_to_npz_dataset,
    load_annotations,
    save_annotations,
    validate_dataset,
)


# ===========================================================================
# 数据结构
# ===========================================================================
def test_action_annotation_roundtrip() -> None:
    """标注的存取往返。"""
    a = ActionAnnotation(label="抬右手", start_frame=10, end_frame=55)
    b = ActionAnnotation.from_dict(a.to_dict())
    assert b.label == a.label and b.start_frame == 10 and b.end_frame == 55
    assert b.duration_frames == 45


def test_video_annotation_roundtrip() -> None:
    """视频标注的存取往返。"""
    v = VideoAnnotation(
        video="s01_b.mp4", fps=25.0, num_frames=100,
        actions=[ActionAnnotation("站立", 0, 30), ActionAnnotation("抬手", 30, 100)],
        keypoints="a.npz",
    )
    v2 = VideoAnnotation.from_dict(v.to_dict())
    assert v2.video == "s01_b.mp4" and v2.fps == 25.0 and v2.num_frames == 100
    assert len(v2.actions) == 2
    assert v2.actions[1].label == "抬手"
    assert v2.keypoints == "a.npz"


# ===========================================================================
# 自检
# ===========================================================================
def test_validate_detects_overlap() -> None:
    """重叠的动作段要被发现。"""
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=100,
        actions=[ActionAnnotation("A", 0, 50), ActionAnnotation("B", 40, 80)],
    )
    problems = v.validate()
    assert any("重叠" in p for p in problems), f"没检测到重叠: {problems}"


def test_validate_detects_out_of_bounds() -> None:
    """越界的动作段要被发现。"""
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=50,
        actions=[ActionAnnotation("A", 0, 100)],
    )
    problems = v.validate()
    assert any("越界" in p for p in problems), f"没检测到越界: {problems}"


def test_validate_detects_negative_length() -> None:
    """长度非法的段要被发现。"""
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=100,
        actions=[ActionAnnotation("A", 50, 30)],
    )
    problems = v.validate()
    assert any("长度非法" in p for p in problems), f"没检测到长度问题: {problems}"


def test_validate_accepts_good_annotation() -> None:
    """正常的标注不该报问题。"""
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=100,
        actions=[ActionAnnotation("A", 0, 30), ActionAnnotation("B", 30, 70),
                 ActionAnnotation("A", 70, 100)],
    )
    assert v.validate() == [], f"正常标注不该有问题: {v.validate()}"


def test_validate_allows_gap_between_segments() -> None:
    """段与段之间留空隙是允许的(那部分可能是"无动作")。"""
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=100,
        actions=[ActionAnnotation("A", 0, 30), ActionAnnotation("B", 50, 80)],
    )
    assert v.validate() == []


# ===========================================================================
# 存取
# ===========================================================================
def test_save_and_load_annotations() -> None:
    """保存再读回来应该一致。"""
    v = VideoAnnotation(
        video="s01_b.mp4", fps=30.0, num_frames=100,
        actions=[ActionAnnotation("站立", 0, 30), ActionAnnotation("抬手", 30, 100)],
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ann.json"
        save_annotations([v], p)
        videos, labels = load_annotations(p)

    assert len(videos) == 1
    assert videos[0].num_frames == 100
    assert len(videos[0].actions) == 2
    assert set(labels) == {"站立", "抬手"}


def test_save_annotations_collects_labels() -> None:
    """标签表要自动汇总。"""
    vs = [
        VideoAnnotation(video="s01_b.mp4", actions=[ActionAnnotation("X", 0, 10)]),
        VideoAnnotation(video="b.mp4", actions=[ActionAnnotation("Y", 0, 10)]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ann.json"
        save_annotations(vs, p)
        payload = json.loads(p.read_text(encoding="utf-8"))
        _, labels = load_annotations(p)

    assert set(payload["labels"]) == {"X", "Y"}
    assert set(labels) == {"X", "Y"}


# ===========================================================================
# 数据集校验
# ===========================================================================
def test_validate_dataset_reports_counts() -> None:
    """校验报告要统计出视频数、段数、类别数。"""
    vs = [
        VideoAnnotation(video="s01_b.mp4", num_frames=100,
                        actions=[ActionAnnotation("A", 0, 50), ActionAnnotation("B", 50, 100)]),
        VideoAnnotation(video="b.mp4", num_frames=100,
                        actions=[ActionAnnotation("A", 0, 100)]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "anns"
        d.mkdir()
        save_annotations(vs, d / "ann.json")
        report = validate_dataset(d, verbose=False)

    assert report["num_videos"] == 2
    assert report["num_segments"] == 3
    assert report["num_labels"] == 2
    assert report["label_counts"]["A"] == 2
    assert report["problems"] == []


def test_validate_dataset_warns_on_imbalance() -> None:
    """类别严重不均衡要给出警告。"""
    vs = [
        VideoAnnotation(video=f"v{i}.mp4", num_frames=10,
                        actions=[ActionAnnotation("常见", 0, 10)])
        for i in range(20)
    ]
    vs.append(
        VideoAnnotation(video="rare.mp4", num_frames=10,
                        actions=[ActionAnnotation("罕见", 0, 10)])
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "anns"
        d.mkdir()
        save_annotations(vs, d / "ann.json")
        report = validate_dataset(d, verbose=False)

    assert report["warnings"], "类别不均衡应该产生警告"
    assert any("不均衡" in w for w in report["warnings"])


def test_validate_dataset_warns_on_few_samples() -> None:
    """某类别样本太少要警告。"""
    vs = [
        VideoAnnotation(video="s01_b.mp4", num_frames=10,
                        actions=[ActionAnnotation("A", 0, 10)]),
        VideoAnnotation(video="b.mp4", num_frames=10,
                        actions=[ActionAnnotation("B", 0, 10)]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "anns"
        d.mkdir()
        save_annotations(vs, d / "ann.json")
        report = validate_dataset(d, verbose=False)

    assert any("少于 10" in w for w in report["warnings"])


def test_validate_dataset_detects_missing_npz() -> None:
    """声明了关键点但文件不存在,要报错。"""
    vs = [
        VideoAnnotation(video="s01_b.mp4", num_frames=100, keypoints="missing.npz",
                        actions=[ActionAnnotation("A", 0, 100)]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "anns"
        d.mkdir()
        npz_dir = Path(tmp) / "npz"
        npz_dir.mkdir()
        save_annotations(vs, d / "ann.json")
        report = validate_dataset(d, npz_dir=npz_dir, verbose=False)

    assert any("关键点文件不存在" in p for p in report["problems"])


def test_validate_dataset_detects_bad_npz_shape() -> None:
    """关键点形状不对要报错。"""
    vs = [
        VideoAnnotation(video="s01_b.mp4", num_frames=100, keypoints="bad.npz",
                        actions=[ActionAnnotation("A", 0, 100)]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "anns"
        d.mkdir()
        npz_dir = Path(tmp) / "npz"
        npz_dir.mkdir()
        np.savez(npz_dir / "bad.npz", keypoints=np.zeros((100, 5, 3)))
        save_annotations(vs, d / "ann.json")
        report = validate_dataset(d, npz_dir=npz_dir, verbose=False)

    assert any("形状" in p for p in report["problems"])


# ===========================================================================
# 导出训练数据
# ===========================================================================
def test_export_creates_windows() -> None:
    """导出应该按窗口切成 npz。"""
    v = VideoAnnotation(
        video="s01_b.mp4", fps=30.0, num_frames=200, keypoints="a.npz",
        actions=[ActionAnnotation("A", 0, 100)],
    )
    with tempfile.TemporaryDirectory() as tmp:
        npz_dir = Path(tmp) / "npz"
        npz_dir.mkdir()
        out_dir = Path(tmp) / "out"
        np.savez(npz_dir / "a.npz", keypoints=np.zeros((200, 17, 3)))

        stats = export_to_npz_dataset([v], npz_dir, out_dir, window=32, stride=16)

        files = sorted(out_dir.glob("*.npz"))
        assert stats["num_windows"] == len(files)
        assert stats["num_windows"] > 0
        assert stats["label_counts"]["A"] == stats["num_windows"]


def test_export_windows_have_correct_shape() -> None:
    """导出的窗口形状应该是 (window, 17, 3)。"""
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=100, keypoints="a.npz",
        actions=[ActionAnnotation("A", 0, 100)],
    )
    with tempfile.TemporaryDirectory() as tmp:
        npz_dir = Path(tmp) / "npz"
        npz_dir.mkdir()
        out_dir = Path(tmp) / "out"
        np.savez(npz_dir / "a.npz", keypoints=np.ones((100, 17, 3)))

        export_to_npz_dataset([v], npz_dir, out_dir, window=32, stride=16)

        data = np.load(sorted(out_dir.glob("*.npz"))[0])
        assert data["skel"].shape == (32, 17, 3), f"窗口形状错误: {data['skel'].shape}"
        assert data["label"] == "A"


def test_export_skips_videos_without_keypoints() -> None:
    """没有关键点的视频要跳过(而不是崩掉)。"""
    v = VideoAnnotation(video="s01_b.mp4", num_frames=100,
                        actions=[ActionAnnotation("A", 0, 100)])
    with tempfile.TemporaryDirectory() as tmp:
        stats = export_to_npz_dataset(
            [v], Path(tmp), Path(tmp) / "out", window=32, stride=16
        )
    assert stats["num_windows"] == 0
    assert stats["num_skipped"] == 1


def test_export_respects_action_boundaries() -> None:
    """
    窗口不应该跨越动作段边界。

    跨界的窗口标签就有歧义了(到底算前一个动作还是后一个)。
    """
    v = VideoAnnotation(
        video="s01_b.mp4", num_frames=200, keypoints="a.npz",
        actions=[ActionAnnotation("A", 0, 50), ActionAnnotation("B", 100, 200)],
    )
    with tempfile.TemporaryDirectory() as tmp:
        npz_dir = Path(tmp) / "npz"
        npz_dir.mkdir()
        out_dir = Path(tmp) / "out"
        np.savez(npz_dir / "a.npz", keypoints=np.zeros((200, 17, 3)))

        export_to_npz_dataset([v], npz_dir, out_dir, window=32, stride=16)

        for f in out_dir.glob("*.npz"):
            data = np.load(f)
            s, e = int(data["start_frame"]), int(data["end_frame"])
            label = str(data["label"])
            # 窗口必须完全落在对应动作段内
            if label == "A":
                assert s >= 0 and e <= 50, f"A 段窗口越界: [{s},{e})"
            else:
                assert s >= 100 and e <= 200, f"B 段窗口越界: [{s},{e})"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
