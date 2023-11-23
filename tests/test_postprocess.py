"""
One-Euro Filter 与坐标归一化的单元测试。

运行:  python -m tests.test_postprocess
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.core.constants import (  # noqa: E402
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from actionflow.postprocess.filters import OneEuroFilter, SkeletonFilter  # noqa: E402
from actionflow.postprocess.normalize import (  # noqa: E402
    normalize_keypoints,
    normalize_sequence,
)


def make_skeleton(cx: float = 500.0, cy: float = 400.0, scale: float = 1.0) -> np.ndarray:
    """构造一个标准站姿骨架（17, 3），躯干长度约 100*scale 像素。"""
    kp = np.zeros((17, 3), dtype=np.float64)
    kp[:, 2] = 0.95  # 高置信度

    # 躯干：肩在 y=cy-50*scale，髋在 y=cy+50*scale（图像 y 向下，所以髋更大）
    kp[L_SHOULDER] = [cx - 30 * scale, cy - 50 * scale, 0.95]
    kp[R_SHOULDER] = [cx + 30 * scale, cy - 50 * scale, 0.95]
    kp[L_HIP] = [cx - 25 * scale, cy + 50 * scale, 0.95]
    kp[R_HIP] = [cx + 25 * scale, cy + 50 * scale, 0.95]
    kp[L_ELBOW] = [cx - 35 * scale, cy + 5 * scale, 0.95]
    kp[R_ELBOW] = [cx + 35 * scale, cy + 5 * scale, 0.95]
    kp[L_WRIST] = [cx - 38 * scale, cy + 60 * scale, 0.95]
    kp[R_WRIST] = [cx + 38 * scale, cy + 60 * scale, 0.95]
    kp[L_KNEE] = [cx - 26 * scale, cy + 130 * scale, 0.95]
    kp[R_KNEE] = [cx + 26 * scale, cy + 130 * scale, 0.95]
    kp[L_ANKLE] = [cx - 27 * scale, cy + 210 * scale, 0.95]
    kp[R_ANKLE] = [cx + 27 * scale, cy + 210 * scale, 0.95]

    # 头部
    kp[0] = [cx, cy - 80 * scale, 0.95]                       # nose
    kp[1] = [cx - 6 * scale, cy - 84 * scale, 0.95]           # left_eye
    kp[2] = [cx + 6 * scale, cy - 84 * scale, 0.95]           # right_eye
    kp[3] = [cx - 12 * scale, cy - 82 * scale, 0.95]          # left_ear
    kp[4] = [cx + 12 * scale, cy - 82 * scale, 0.95]          # right_ear
    return kp


# ===========================================================================
# One-Euro Filter
# ===========================================================================
def test_one_euro_reduces_noise_on_static_signal() -> None:
    """静止信号 + 高斯噪声 → 滤波后抖动应变小。"""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    noisy = 100.0 + rng.normal(0, 3.0, size=200)
    filtered = [f(v, dt=1 / 30) for v in noisy]

    raw_std = float(np.std(noisy))
    filt_std = float(np.std(filtered))
    assert filt_std < raw_std * 0.75, f"滤波后抖动应变小: {raw_std:.2f} → {filt_std:.2f}"


def test_one_euro_lower_cutoff_gives_more_smoothing() -> None:
    """min_cutoff 越小 → 平滑越强（抖动压得越狠）。"""
    rng = np.random.default_rng(0)
    noisy = 100.0 + rng.normal(0, 3.0, size=200)

    stds = []
    for mc in (0.5, 2.0, 8.0):
        f = OneEuroFilter(min_cutoff=mc, beta=0.0)
        out = np.array([f(v, dt=1 / 30) for v in noisy])
        stds.append(float(np.std(out)))

    assert stds[0] < stds[1] < stds[2], f"min_cutoff 越小应越平滑，实际 {stds}"


def test_one_euro_tracks_fast_motion_without_lag() -> None:
    """快速运动 → beta 生效，跟踪误差应远小于固定低通滤波。"""
    n = 100
    signal = np.linspace(0, 100, n)  # 每帧移动 1 像素

    # beta=0（几乎不滤波）
    f_fast = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out_fast = np.array([f_fast(v, dt=1 / 30) for v in signal])

    # beta 高（快速运动时少滤波）
    f_adaptive = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    out_adaptive = np.array([f_adaptive(v, dt=1 / 30) for v in signal])

    # 忽略前 20 帧的建立期
    err_fast = float(np.abs(out_fast[20:] - signal[20:]).mean())
    err_adaptive = float(np.abs(out_adaptive[20:] - signal[20:]).mean())

    assert err_adaptive < err_fast, (
        f"自适应滤波应该比固定滤波跟得更紧: adaptive={err_adaptive:.3f} fast={err_fast:.3f}"
    )


def test_one_euro_first_frame_passthrough() -> None:
    """第一帧应原样输出（没有历史可参考）。"""
    f = OneEuroFilter()
    assert f(42.0) == 42.0


def test_skeleton_filter_preserves_confidence() -> None:
    """骨架滤波不应改变置信度通道。"""
    filt = SkeletonFilter(min_cutoff=1.0, beta=0.02)
    kp = make_skeleton()
    kp[5, 2] = 0.42  # 左肩置信度特殊值
    out = filt(kp, dt=1 / 30)
    assert abs(out[5, 2] - 0.42) < 1e-9, "置信度不应被滤波"


def test_skeleton_filter_reset_clears_history() -> None:
    """reset 后第一帧应原样输出。"""
    filt = SkeletonFilter()
    kp = make_skeleton()
    filt(kp, dt=1 / 30)
    filt.reset()
    out = filt(kp, dt=1 / 30)
    assert np.allclose(out[:, :2], kp[:, :2]), "reset 后应原样输出"


# ===========================================================================
# 坐标归一化
# ===========================================================================
def test_normalize_is_translation_invariant() -> None:
    """人在画面不同位置，归一化后应完全一致。"""
    kp_a = make_skeleton(cx=200, cy=300)
    kp_b = make_skeleton(cx=800, cy=700)

    n_a = normalize_keypoints(kp_a)
    n_b = normalize_keypoints(kp_b)

    assert np.allclose(n_a[:, :2], n_b[:, :2], atol=1e-6), "平移不变性失败"


def test_normalize_is_scale_invariant() -> None:
    """人的体型大小不同，归一化后应一致。"""
    kp_small = make_skeleton(cx=500, cy=400, scale=0.7)
    kp_large = make_skeleton(cx=500, cy=400, scale=1.4)

    n_small = normalize_keypoints(kp_small)
    n_large = normalize_keypoints(kp_large)

    assert np.allclose(n_small[:, :2], n_large[:, :2], atol=1e-6), "尺度不变性失败"


def test_normalize_puts_hip_center_at_origin() -> None:
    """归一化后髋中心应在原点。"""
    kp = make_skeleton(cx=333, cy=555)
    n = normalize_keypoints(kp)
    hip_center = (n[L_HIP, :2] + n[R_HIP, :2]) / 2
    assert np.allclose(hip_center, [0, 0], atol=1e-6), f"髋中心应在原点，实际 {hip_center}"


def test_normalize_orients_body_upward() -> None:
    """归一化后躯干方向应朝 +y（头顶为正）。"""
    kp = make_skeleton(cx=500, cy=400)
    n = normalize_keypoints(kp)
    # 肩应该在髋的负 y 方向（因为标准坐标系 y 朝上）
    shoulder_y = (n[L_SHOULDER, 1] + n[R_SHOULDER, 1]) / 2
    hip_y = (n[L_HIP, 1] + n[R_HIP, 1]) / 2
    assert shoulder_y < hip_y, f"肩应在髋的上方，实际 shoulder_y={shoulder_y:.3f} hip_y={hip_y:.3f}"


def test_normalize_handles_rotation() -> None:
    """人转 30° 后，归一化应旋转回标准朝向。"""
    kp = make_skeleton(cx=500, cy=400)
    angle = np.deg2rad(30)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])

    kp_rot = kp.copy()
    kp_rot[:, :2] = kp_rot[:, :2] @ rot.T

    n_rot = normalize_keypoints(kp_rot)
    n_ref = normalize_keypoints(kp)
    assert np.allclose(n_rot[:, :2], n_ref[:, :2], atol=1e-6), "朝向不变性失败"


def test_normalize_torso_length_is_one() -> None:
    """归一化后躯干长度应为 1.0。"""
    kp = make_skeleton(cx=500, cy=400, scale=1.0)
    n = normalize_keypoints(kp)
    torso = float(np.linalg.norm(n[L_SHOULDER, :2] - n[L_HIP, :2]))
    assert abs(torso - 1.0) < 1e-6, f"躯干长度应为 1.0，实际 {torso}"


def test_normalize_preserves_confidence() -> None:
    """归一化不应改变置信度。"""
    kp = make_skeleton()
    kp[9, 2] = 0.15  # 左腕低置信度
    n = normalize_keypoints(kp)
    assert abs(n[9, 2] - 0.15) < 1e-9, "置信度不应改变"


def test_normalize_sequence_shape() -> None:
    """整段序列归一化后形状不变。"""
    seq = np.stack([make_skeleton(cx=500 + i) for i in range(10)])
    out = normalize_sequence(seq)
    assert out.shape == seq.shape, f"形状应保持，实际 {out.shape}"


def test_normalize_handles_degenerate_torso() -> None:
    """躯干退化（所有点重合）时不应产生 NaN。"""
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    n = normalize_keypoints(kp)
    assert not np.any(np.isnan(n)), "不应产生 NaN"


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
            print(f"  ✗ {name}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}\n      {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
