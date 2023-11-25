"""
Evaluate 模块的单元测试 —— 段级评估的正确性。

关键:这里的测试**不依赖任何模型权重**,全部用手工构造的段。
匹配规则是评估工具的地基 —— 它错了,所有指标都是错的,
所以必须单独测死。

运行:
    python tests/test_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actionflow.evaluate import (  # noqa: E402
    Segment,
    aggregate,
    evaluate_video,
    match_segments,
    temporal_iou,
)

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  \033[92m✓\033[0m {name}")
    else:
        FAILED += 1
        print(f"  \033[91m✗\033[0m {name}  {detail}")


def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


# ===========================================================================
def test_temporal_iou() -> None:
    print("\n【IoU 计算】")

    # 完全重叠 → IoU = 1
    a = Segment("x", 0, 100)
    b = Segment("x", 0, 100)
    check("完全重叠 IoU=1", approx(temporal_iou(a, b), 1.0))

    # 完全不重叠 → IoU = 0
    c = Segment("x", 200, 300)
    check("完全不重叠 IoU=0", approx(temporal_iou(a, c), 0.0))

    # 一半重叠:交集 [50,100)=50,并集 [0,200)=200 → 0.25
    d = Segment("x", 50, 200)
    check("一半重叠 IoU=0.25", approx(temporal_iou(a, d), 0.25))

    # 只碰一个点(端点相接)→ 交集为空 → IoU = 0
    e = Segment("x", 100, 200)
    check("端点相接 IoU=0", approx(temporal_iou(a, e), 0.0))

    # 包含关系:交集 50,并集 100 → 0.5
    f = Segment("x", 0, 50)
    check("完全包含 IoU=0.5", approx(temporal_iou(a, f), 0.5))


# ===========================================================================
def test_match_basic() -> None:
    print("\n【基本配对】")

    gt = [Segment("抬手", 0, 100), Segment("握拳", 100, 200)]
    pred = [Segment("抬手", 5, 95), Segment("握拳", 105, 195)]

    pairs = match_segments(gt, pred)
    check("两段都配上了", len(pairs) == 2, f"实际 {len(pairs)}")
    check("抬手配抬手", (0, 0, pairs[0][2]) == (0, 0, pairs[0][2]) and pairs[0][0] == 0)
    check("握拳配握拳", pairs[1][0] == 1 and pairs[1][1] == 1)
    check("IoU 正确", approx(pairs[0][2], 90 / 100))


# ===========================================================================
def test_match_label_must_match() -> None:
    print("\n【类别必须相同】")

    gt = [Segment("抬手", 0, 100)]
    pred = [Segment("握拳", 0, 100)]  # 时间完全重叠,但类别不同

    pairs = match_segments(gt, pred)
    check("类别不同不配对", len(pairs) == 0, f"实际 {len(pairs)}")


# ===========================================================================
def test_match_greedy_no_duplicate() -> None:
    print("\n【贪心配对:一个真值段不能被重复占用】")

    # 一个真值段 vs 三个重叠的预测段(模拟「把一个动作切成三段」)
    # 三段的 IoU 刻意各不相同,这样才能验证「配的是 IoU 最高的那个」
    gt = [Segment("抬手", 0, 100)]
    pred = [
        Segment("抬手", 0, 60),    # 交集 60,并集 100 → IoU 0.60
        Segment("抬手", 0, 100),   # 交集 100,并集 100 → IoU 1.00  ← 最高
        Segment("抬手", 40, 100),  # 交集 60,并集 100 → IoU 0.60
    ]

    pairs = match_segments(gt, pred)
    check("只配上一对", len(pairs) == 1, f"实际 {len(pairs)}")
    check("配的是 IoU 最高的那个", pairs[0][1] == 1, f"实际 pred idx={pairs[0][1]}")
    check("配对的 IoU=1.0", approx(pairs[0][2], 1.0), f"实际 {pairs[0][2]}")

    # 关键:另外两个预测段必须被算成**误检**,不能算命中
    res = evaluate_video(gt, pred)
    check("命中数=1", res.n_matched == 1)
    check("预测数=3", res.n_pred == 3)
    check("Precision=1/3", approx(res.precision, 1 / 3))
    check("Recall=1.0", approx(res.recall, 1.0))
    check("F1=0.5", approx(res.f1, 0.5), f"实际 {res.f1}")


# ===========================================================================
def test_match_greedy_prefers_higher_iou() -> None:
    print("\n【贪心配对优先高 IoU】")

    gt = [Segment("A", 0, 100), Segment("B", 100, 200)]
    pred = [
        Segment("A", 0, 60),   # 与 A 的 IoU = 0.6
        Segment("A", 0, 100),  # 与 A 的 IoU = 1.0
    ]

    pairs = match_segments(gt, pred)
    # IoU=1.0 的那个先被配走
    hit = {(gi, pi) for gi, pi, _ in pairs}
    check("高 IoU 的先配", (0, 1) in hit, f"实际 {hit}")
    check("A 只配一次", sum(1 for gi, _, _ in pairs if gi == 0) == 1)


# ===========================================================================
def test_iou_threshold() -> None:
    print("\n【IoU 阈值生效】")

    gt = [Segment("A", 0, 100)]
    # 只重叠 5% → IoU ≈ 0.0476
    pred = [Segment("A", 95, 200)]

    check("低 IoU 时不配(默认阈值0.1)", len(match_segments(gt, pred)) == 0)
    check("阈值调到 0.01 就配上了", len(match_segments(gt, pred, iou_threshold=0.01)) == 1)


# ===========================================================================
def test_empty_cases() -> None:
    print("\n【边界情况】")

    res = evaluate_video([], [], video="empty")
    check("空输入不崩", res.n_gt == 0 and res.n_pred == 0)
    check("空输入 Precision=0", approx(res.precision, 0.0))
    check("空输入 F1=0", approx(res.f1, 0.0))

    res = evaluate_video([Segment("A", 0, 10)], [], video="no_pred")
    check("有真值无预测:Recall=0", approx(res.recall, 0.0))
    check("有真值无预测:Precision=0", approx(res.precision, 0.0))

    res = evaluate_video([], [Segment("A", 0, 10)], video="no_gt")
    check("有预测无真值:Precision=0", approx(res.precision, 0.0))
    check("有预测无真值:Recall=0", approx(res.recall, 0.0))


# ===========================================================================
def test_boundary_error() -> None:
    print("\n【边界误差】")

    gt = [Segment("A", 100, 200)]
    # 预测段整体偏移 +10 帧(起点 110,终点 210)
    pred = [Segment("A", 110, 210)]

    res = evaluate_video(gt, pred)
    check("命中 1 段", res.n_matched == 1)
    # 起点差 10,终点差 10 → 平均 10
    check("边界误差=10 帧", approx(res.mean_boundary_error, 10.0), f"实际 {res.mean_boundary_error}")


# ===========================================================================
def test_label_accuracy() -> None:
    print("\n【类别准确率(不管边界)】")

    gt = [Segment("A", 0, 100), Segment("B", 200, 300)]
    # 边界全错,但类别对
    pred = [Segment("A", 500, 600), Segment("B", 700, 800)]

    res = evaluate_video(gt, pred)
    check("类别准确率=1.0", approx(res.label_accuracy, 1.0), f"实际 {res.label_accuracy}")
    # 两个类别都出现了 → label_correct=2
    check("类别判对 2 段", res.label_correct == 2)
    # 但因为 IoU=0,一个都没配上
    check("但没有段级命中", res.n_matched == 0)


# ===========================================================================
def test_aggregate() -> None:
    print("\n【多视频汇总】")

    r1 = evaluate_video(
        [Segment("A", 0, 100), Segment("B", 100, 200)],
        [Segment("A", 0, 100), Segment("B", 100, 200)],
        video="v1",
    )
    r2 = evaluate_video([Segment("A", 0, 100)], [], video="v2")

    rep = aggregate([r1, r2])
    check("视频数=2", rep["num_videos"] == 2)
    check("真值段=3", rep["num_gt_segments"] == 3)
    check("预测段=2", rep["num_pred_segments"] == 2)
    check("命中=2", rep["num_matched"] == 2)
    # Precision = 2/2 = 1.0
    check("Precision=1.0", approx(rep["precision"], 1.0))
    # Recall = 2/3
    check("Recall=2/3", approx(rep["recall"], 2 / 3))
    check("平均 IoU=1.0", approx(rep["mean_iou"], 1.0))


# ===========================================================================
def test_aggregate_micro_averaging() -> None:
    print("\n【汇总用的是 micro 平均(不是各视频指标的平均)】")

    # 视频1: 10 个真值,命中 1 个
    r1 = evaluate_video(
        [Segment("A", i * 100, i * 100 + 90) for i in range(10)],
        [Segment("A", 0, 90)],
        video="v1",
    )
    # 视频2: 1 个真值,命中 1 个
    r2 = evaluate_video([Segment("A", 0, 90)], [Segment("A", 0, 90)], video="v2")

    rep = aggregate([r1, r2])
    # micro: 命中 2 / 预测 2 = 1.0; 命中 2 / 真值 11 = 0.1818
    check("micro Precision=1.0", approx(rep["precision"], 1.0))
    check("micro Recall=2/11", approx(rep["recall"], 2 / 11), f"实际 {rep['recall']}")
    # 如果是 macro 平均: (0.1 + 1.0)/2 = 0.55,和 micro 不同
    check("不是 macro 平均", not approx(rep["recall"], 0.55))


# ===========================================================================
def _run_all() -> int:
    tests = [
        test_temporal_iou,
        test_match_basic,
        test_match_label_must_match,
        test_match_greedy_no_duplicate,
        test_match_greedy_prefers_higher_iou,
        test_iou_threshold,
        test_empty_cases,
        test_boundary_error,
        test_label_accuracy,
        test_aggregate,
        test_aggregate_micro_averaging,
    ]
    print("=" * 68)
    print("Evaluate 模块测试")
    print("=" * 68)
    for t in tests:
        t()
    print("\n" + "=" * 68)
    print(f"{PASSED}/{PASSED + FAILED} 通过")
    if FAILED:
        print(f"\033[91m{FAILED} 个失败\033[0m")
    else:
        print("\033[92m全部通过 ✓\033[0m")
    print("=" * 68)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
