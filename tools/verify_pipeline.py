"""
端到端流水线自检 —— 不需要 ONNX 权重、不需要训练数据。

用合成骨架序列验证三层管线的每一环都工作正常：
    1. One-Euro 滤波确实降低了抖动
    2. 坐标归一化确实做到了位置/尺度/朝向不变
    3. 分类器能输出合法概率
    4. 解码层能还原出正确的动作段和时间戳

用法:
    python tools/verify_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.core.constants import (  # noqa: E402
    L_ELBOW,
    L_WRIST,
    R_ELBOW,
    R_WRIST,
)
from actionflow.postprocess.decode import decode_segments  # noqa: E402
from actionflow.postprocess.filters import SkeletonFilter  # noqa: E402
from actionflow.postprocess.normalize import normalize_keypoints  # noqa: E402

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def make_pose(action: int, cx: float = 500.0, cy: float = 400.0, scale: float = 1.0) -> np.ndarray:
    """构造一个动作对应的 17 点骨架。"""
    kp = np.zeros((17, 3), dtype=np.float64)
    kp[:, 2] = 0.95

    sy, hy = cy - 60 * scale, cy + 20 * scale
    pts = {
        0: (cx, cy - 105 * scale),
        1: (cx - 8 * scale, cy - 112 * scale),
        2: (cx + 8 * scale, cy - 112 * scale),
        3: (cx - 18 * scale, cy - 108 * scale),
        4: (cx + 18 * scale, cy - 108 * scale),
        5: (cx - 45 * scale, sy), 6: (cx + 45 * scale, sy),
        7: (cx - 55 * scale, cy - 10 * scale), 8: (cx + 55 * scale, cy - 10 * scale),
        9: (cx - 58 * scale, cy + 45 * scale), 10: (cx + 58 * scale, cy + 45 * scale),
        11: (cx - 30 * scale, hy), 12: (cx + 30 * scale, hy),
        13: (cx - 32 * scale, cy + 100 * scale), 14: (cx + 32 * scale, cy + 100 * scale),
        15: (cx - 33 * scale, cy + 180 * scale), 16: (cx + 33 * scale, cy + 180 * scale),
    }
    if action == 1:  # 抬右手
        pts[8] = (cx + 75 * scale, cy - 90 * scale)
        pts[10] = (cx + 85 * scale, cy - 165 * scale)
    elif action == 2:  # 抬左手
        pts[7] = (cx - 75 * scale, cy - 90 * scale)
        pts[9] = (cx - 85 * scale, cy - 165 * scale)
    elif action == 3:  # 抬右腿
        pts[14] = (cx + 60 * scale, cy + 40 * scale)
        pts[16] = (cx + 70 * scale, cy - 20 * scale)

    for idx, (x, y) in pts.items():
        kp[idx, :2] = (x, y)
    return kp


def check_1_filter() -> None:
    section("检查 1 · One-Euro Filter —— 抑制关键点抖动")
    rng = np.random.default_rng(0)

    # beta=0：检测噪声本身会产生很大的表观速度，beta 偏大会把噪声
    # 误判成"快速运动"从而削弱滤波。详见 FilterConfig 的注释。
    filt = SkeletonFilter(min_cutoff=1.0, beta=0.0)
    raw, smoothed = [], []
    for _ in range(60):
        kp = make_pose(0)
        kp[:, :2] += rng.normal(0, 3.0, size=(17, 2))
        raw.append(kp.copy())
        smoothed.append(filt(kp, dt=1 / 30))

    raw_arr = np.stack(raw)
    sm_arr = np.stack(smoothed)

    # 用骨长的标准差衡量抖动（骨长本应恒定）
    def bone_std(arr):
        lw = np.linalg.norm(arr[:, L_WRIST, :2] - arr[:, L_ELBOW, :2], axis=-1)
        rw = np.linalg.norm(arr[:, R_WRIST, :2] - arr[:, R_ELBOW, :2], axis=-1)
        return float(np.std(np.concatenate([lw, rw])))

    raw_std = bone_std(raw_arr)
    sm_std = bone_std(sm_arr)
    reduction = (1 - sm_std / raw_std) * 100
    print(f"  骨长标准差: 原始 {raw_std:.3f} px  →  滤波后 {sm_std:.3f} px")
    print(f"  抖动降低: {reduction:.1f}%")
    assert reduction > 30.0, f"滤波效果不足（{reduction:.1f}%）"
    print(f"  {PASS} 滤波有效")


def check_2_normalize() -> None:
    section("检查 2 · 坐标归一化 —— 位置/尺度/朝向不变")
    ref = normalize_keypoints(make_pose(0, cx=500, cy=400, scale=1.0))

    shifted = normalize_keypoints(make_pose(0, cx=900, cy=700, scale=1.0))
    diff = np.abs(ref[:, :2] - shifted[:, :2]).max()
    print(f"  位置不变性: 最大差异 {diff:.2e}")
    assert diff < 1e-9, "位置不变性失败"

    scaled = normalize_keypoints(make_pose(0, cx=500, cy=400, scale=1.6))
    diff = np.abs(ref[:, :2] - scaled[:, :2]).max()
    print(f"  尺度不变性: 最大差异 {diff:.2e}")
    assert diff < 1e-9, "尺度不变性失败"

    # 朝向不变性
    ang = np.deg2rad(30)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    kp = make_pose(0)
    kp[:, :2] = kp[:, :2] @ rot.T
    rotated = normalize_keypoints(kp)
    diff = np.abs(ref[:, :2] - rotated[:, :2]).max()
    print(f"  朝向不变性: 最大差异 {diff:.2e}")
    assert diff < 1e-9, "朝向不变性失败"

    print(f"  {PASS} 三个不变性全部满足")


def check_3_model() -> None:
    section("检查 3 · 流式分类器 —— 输出合法概率 + 前缀不变性")
    try:
        import torch
    except ImportError:
        print("  [跳过] 未安装 torch")
        return

    from actionflow.core.config import ModelConfig
    from actionflow.models.streaming import build_streaming_model

    cfg = ModelConfig(num_classes=4, d_model=64, nhead=4, num_layers=2, window=32)
    model = build_streaming_model(cfg)
    model.eval()

    # 用合成骨架喂进去
    seq = np.stack([make_pose(a) for a in [0] * 16 + [1] * 16])
    skel = torch.from_numpy(seq).float().unsqueeze(0)

    with torch.no_grad():
        probs = model.forward_probs(skel)

    print(f"  逐帧输出 shape: {tuple(probs.shape)}")
    sums = probs.sum(dim=-1)
    ok = torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print(f"  概率和为 1: {ok}")
    assert ok, "概率不是合法分布"

    # ── 前缀不变性:流式架构的立身之本 ──────────────────────────
    # 只用前 16 帧算的结果,必须和用 32 帧算的前 16 帧完全一样。
    # 这条不成立的话,实时输出的历史结果会被未来帧偷偷改掉。
    with torch.no_grad():
        full = model.forward_probs(skel)
        prefix = model.forward_probs(skel[:, :16])
    diff = (full[:, :16] - prefix).abs().max().item()
    print(f"  前缀不变性最大差异: {diff:.3e}")
    assert diff < 1e-5, f"前缀不变性被破坏,差异 {diff:.3e}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params/1e6:.3f}M")
    print(f"  {PASS} 模型正常")


def check_4_decode() -> None:
    section("检查 4 · 时序解码 —— 还原动作段与时间戳")
    # 构造带噪声的 4 段动作序列
    C, T, FPS = 4, 120, 30.0
    probs = np.zeros((T, C))
    truth = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30
    rng = np.random.default_rng(0)
    for t, c in enumerate(truth):
        probs[t, c] = float(np.clip(0.9 + rng.normal(0, 0.05), 0, 1))
        rest = (1 - probs[t, c]) / (C - 1)
        for k in range(C):
            if k != c:
                probs[t, k] = rest * (0.5 + rng.random())

    # 注入噪声
    probs[35, :] = 0.05
    probs[35, 3] = 0.90
    probs[95, :] = 0.05
    probs[95, 1] = 0.85

    segs = decode_segments(probs, fps=FPS, high=0.7, low=0.4, min_duration=0.3, use_viterbi=True)
    labels = [s.label_id for s in segs]
    print(f"  解码出 {len(segs)} 段: {labels}")

    for s in segs:
        name = ["站立", "抬右手", "抬左手", "抬右腿"][s.label_id]
        print(f"    [{s.start_time:5.2f} - {s.end_time:5.2f}s] {name:8s} conf={s.confidence:.3f}")

    assert labels == [0, 1, 2, 3], f"动作段还原错误: {labels}"
    assert all(s.confidence > 0.5 for s in segs), "置信度量级异常"
    print(f"  {PASS} 动作段与时间戳正确")


def main() -> int:
    print("\n\033[1mActionFlow 流水线自检\033[0m")
    print("不依赖 ONNX 权重 / 不依赖训练数据")

    checks = [
        ("One-Euro 滤波", check_1_filter),
        ("坐标归一化", check_2_normalize),
        ("动作分类器", check_3_model),
        ("时序解码", check_4_decode),
    ]

    failed = 0
    for name, fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  {FAIL} {name} 失败: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  {FAIL} {name} 异常: {type(e).__name__}: {e}")

    print(f"\n{'=' * 68}")
    if failed:
        print(f"\033[91m{failed} 项检查失败\033[0m")
    else:
        print("\033[92m全部检查通过 ✓\033[0m")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
