"""
标注工具的图形界面 (tkinter)

为什么要有 GUI
--------------
标动作段最耗时的不是"点两下选类别",而是**来回拖时间轴找边界**。
命令行里你得记住"现在是第几帧"、反复用 ← → 键微调,效率极低。

GUI 提供的核心效率提升:
    ① 画面上直接画出关键点,**错标一眼就能看出来**
    ② 拖动滑块 → 立刻看到那一帧的画面和关键点
    ③ 键盘快捷键:空格播放/暂停,← → 逐帧,Shift+← → 跳 10 帧
    ④ 设好起止点后按一个键就能存一段

快捷键
------
    空格       播放 / 暂停
    ← →       上一帧 / 下一帧
    Shift+← →  跳 10 帧
    Home/End   跳到开头 / 结尾
    A / D      把当前帧设为「动作段起点」/「终点」
    Enter      保存当前动作段
    N          新建一个动作段
    Delete     删除当前动作段
    L          切换标签(在候选列表里循环)
    Ctrl+S     保存标注文件
    Ctrl+Q     退出

依赖
----
    macOS: 系统自带的 python3 有 tkinter
    Linux: sudo apt install python3-tk
    Windows: 官方安装包自带
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from actionflow.core.constants import SKELETON_EDGES
from actionflow.data.annotate import (
    ActionAnnotation,
    VideoAnnotation,
    load_annotations,
    save_annotations,
)


class AnnotationGUI:
    """
    标注界面。

    ⚠️ 用 tkinter 是因为它**不需要装任何东西**(macOS/Linux 通常自带),
       而且对这种"画点 + 拖滑块"的工具足够用。
       换成 PyQt 也能做,但要多一个依赖,对开源项目不友好。
    """

    def __init__(
        self,
        video: str,
        npz_path: str | None,
        out_path: str,
        labels: list[str] | None = None,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.video = video
        self.out_path = Path(out_path)
        self.labels = labels or []

        # ── 读关键点 ────────────────────────────────────────────
        self.kps: np.ndarray | None = None
        if npz_path and Path(npz_path).exists():
            self.kps = np.load(npz_path)["keypoints"]

        # ── 读视频帧 ────────────────────────────────────────────
        import cv2

        self.cap = cv2.VideoCapture(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.current_frame = None

        # ── 标注状态 ────────────────────────────────────────────
        self.annotation = VideoAnnotation(video=str(video), fps=self.fps, num_frames=self.total)
        self.segments: list[ActionAnnotation] = []
        self.pending_start: int | None = None
        self.pending_label: str = self.labels[0] if self.labels else "action"
        self.playing = False

        self._build_ui()
        self._load_existing()
        self._show_frame(0)
        self.root.mainloop()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        tk = self.tk
        self.root = tk.Tk()
        self.root.title(f"ActionFlow 标注工具 — {Path(self.video).name}")
        self.root.configure(bg="#1a1a1a")

        # 画面区
        self.canvas = tk.Canvas(self.root, bg="#000000", height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 时间轴滑块
        self.slider = tk.Scale(
            self.root, from_=0, to=max(self.total - 1, 1), orient=tk.HORIZONTAL,
            command=self._on_slider, bg="#1a1a1a", fg="#ffffff",
            troughcolor="#333333", highlightthickness=0, length=900,
        )
        self.slider.pack(fill=tk.X, padx=10)

        # 状态栏 + 标签选择
        bottom = tk.Frame(self.root, bg="#1a1a1a")
        bottom.pack(fill=tk.X, padx=10, pady=6)

        self.status = tk.Label(
            bottom, text="", bg="#1a1a1a", fg="#00ff88", font=("Menlo", 12)
        )
        self.status.pack(side=tk.LEFT)

        self.label_var = tk.StringVar(value=self.pending_label)
        lbl_frame = tk.Frame(bottom, bg="#1a1a1a")
        lbl_frame.pack(side=tk.RIGHT)
        tk.Label(lbl_frame, text="标签:", bg="#1a1a1a", fg="#ffffff").pack(side=tk.LEFT)
        self.label_combo = tk.Combobox(
            lbl_frame, textvariable=self.label_var, values=self.labels or [],
            width=18, bg="#333333", fg="#ffffff",
        )
        self.label_combo.pack(side=tk.LEFT, padx=4)

        # 动作段列表
        self.seg_list = tk.Listbox(self.root, bg="#1a1a1a", fg="#ffffff", height=6)
        self.seg_list.pack(fill=tk.X, padx=10, pady=4)
        self.seg_list.bind("<<ListboxSelect>>", self._on_select_segment)

        # 按钮行
        btn_row = tk.Frame(self.root, bg="#1a1a1a")
        btn_row.pack(fill=tk.X, padx=10, pady=4)
        for text, cmd in [
            ("▶ 播放 (空格)", self._toggle_play),
            ("设起点 (A)", self._set_start),
            ("设终点 (D)", self._set_end),
            ("保存本段 (Enter)", self._save_segment),
            ("删除本段 (Del)", self._delete_segment),
            ("保存文件 (Ctrl+S)", self._save_file),
        ]:
            tk.Button(
                btn_row, text=text, command=cmd, bg="#333333", fg="#ffffff",
                activebackground="#555555", relief=tk.FLAT, padx=8, pady=4,
            ).pack(side=tk.LEFT, padx=2)

        # 快捷键
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Left>", lambda e: self._step(-1))
        self.root.bind("<Right>", lambda e: self._step(1))
        self.root.bind("<Shift-Left>", lambda e: self._step(-10))
        self.root.bind("<Shift-Right>", lambda e: self._step(10))
        self.root.bind("<Home>", lambda e: self._goto(0))
        self.root.bind("<End>", lambda e: self._goto(self.total - 1))
        self.root.bind("a", lambda e: self._set_start())
        self.root.bind("d", lambda e: self._set_end())
        self.root.bind("<Return>", lambda e: self._save_segment())
        self.root.bind("n", lambda e: self._set_start())
        self.root.bind("<Delete>", lambda e: self._delete_segment())
        self.root.bind("<Control-s>", lambda e: self._save_file())
        self.root.bind("<Control-q>", lambda e: self.root.destroy())

    # ------------------------------------------------------------------
    def _load_existing(self) -> None:
        """如果标注文件已存在,加载进来继续标。"""
        if not self.out_path.exists():
            return
        try:
            videos, labels = load_annotations(self.out_path)
        except Exception:  # noqa: BLE001
            return

        for v in videos:
            if v.video == str(self.video):
                self.segments = list(v.actions)
                if not self.labels:
                    self.labels = list(labels)
                    self.label_combo["values"] = self.labels
                if self.labels:
                    self.pending_label = self.labels[0]
                    self.label_var.set(self.pending_label)
                break
        self._refresh_list()

    # ------------------------------------------------------------------
    def _show_frame(self, idx: int) -> None:
        import cv2

        idx = max(0, min(idx, self.total - 1))
        self.frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame

        # 画关键点
        if self.kps is not None and idx < len(self.kps):
            self._draw_keypoints(frame, self.kps[idx])

        # PIL 转成 tkinter 能显示的格式
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((900, 480))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.tk_img)

        self.slider.set(idx)
        self._update_status()

    # ------------------------------------------------------------------
    def _draw_keypoints(self, frame: np.ndarray, kp: np.ndarray) -> None:
        """把骨架画到画面上(OpenCV 画,比 tkinter 画方便得多)。"""
        import cv2

        h, w = frame.shape[:2]
        scale = max(h, w) / 640.0

        for a, b in SKELETON_EDGES:
            if kp[a][2] < 0.3 or kp[b][2] < 0.3:
                continue
            pa = (int(kp[a][0]), int(kp[a][1]))
            pb = (int(kp[b][0]), int(kp[b][1]))
            cv2.line(frame, pa, pb, (0, 255, 120), int(2 * scale))

        for i in range(len(kp)):
            if kp[i][2] < 0.3:
                continue
            pt = (int(kp[i][0]), int(kp[i][1]))
            cv2.circle(frame, pt, int(4 * scale), (0, 200, 255), -1)

    # ------------------------------------------------------------------
    def _update_status(self) -> None:
        t = self.frame_idx / self.fps if self.fps else 0
        parts = [
            f"帧 {self.frame_idx}/{self.total - 1}",
            f"  时间 {t:.2f}s",
            f"  标签 {self.pending_label}",
        ]
        if self.pending_start is not None:
            parts.append(f"  起点 {self.pending_start}")
        self.status.config(text="  |  ".join(parts))

    def _refresh_list(self) -> None:
        self.seg_list.delete(0, "end")
        for i, s in enumerate(self.segments):
            self.seg_list.insert(
                "end", f"{i+1}. {s.label}  [{s.start_frame}, {s.end_frame})  "
                f"({s.duration_frames} 帧)"
            )

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _on_slider(self, val: str) -> None:
        self._show_frame(int(float(val)))

    def _step(self, delta: int) -> None:
        self._show_frame(self.frame_idx + delta)

    def _goto(self, idx: int) -> None:
        self._show_frame(idx)

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.playing:
            return
        self._step(1)
        if self.frame_idx >= self.total - 1:
            self.playing = False
            return
        self.root.after(33, self._play_loop)

    def _set_start(self) -> None:
        self.pending_start = self.frame_idx
        self._update_status()

    def _set_end(self) -> None:
        if self.pending_start is None:
            self.pending_start = self.frame_idx
            self._update_status()
            return
        self._save_segment()

    def _save_segment(self) -> None:
        if self.pending_start is None:
            self.pending_start = self.frame_idx
            self._update_status()
            return
        s, e = sorted([self.pending_start, self.frame_idx])
        if e <= s:
            return
        self.segments.append(
            ActionAnnotation(label=self.pending_label, start_frame=s, end_frame=e)
        )
        self.pending_start = None
        self._refresh_list()
        self._update_status()

    def _delete_segment(self) -> None:
        sel = self.seg_list.curselection()
        if sel:
            self.segments.pop(sel[0])
            self._refresh_list()

    def _on_select_segment(self, _event=None) -> None:
        sel = self.seg_list.curselection()
        if sel:
            s = self.segments[sel[0]]
            self.pending_label = s.label
            self.label_var.set(s.label)
            self._goto(s.start_frame)

    def _save_file(self) -> None:
        self.annotation.actions = list(self.segments)
        videos: list[VideoAnnotation] = []
        if self.out_path.exists():
            try:
                videos, _ = load_annotations(self.out_path)
            except Exception:  # noqa: BLE001
                videos = []
        videos = [v for v in videos if v.video != str(self.video)]
        videos.append(self.annotation)
        save_annotations(videos, self.out_path)
        self.status.config(text=f"  ✓ 已保存 {self.out_path}  (共 {len(self.segments)} 段)")


def run_gui(args) -> int:
    """GUI 启动入口。"""
    labels = None
    if getattr(args, "labels", None):
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    try:
        AnnotationGUI(args.video, args.npz, args.out, labels)
    except ImportError as e:
        print(f"tkinter 不可用: {e}")
        return 1
    return 0
