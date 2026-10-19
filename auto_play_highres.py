#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-play Gaussians · 高分辨率版
====================================================
支持高分辨率实时预览，方便屏幕录制。

参数:
  --resolution WxH   渲染分辨率（宽×高），任意整数皆可。
  --port P           viser 服务端口 (默认 8080)。

常用分辨率速查表:
  1920x1080   # Full-HD  16:9
  2560x1440   # 2K      16:9
  3840x2160   # 4K      16:9
  4096x2160   # DCI-4K  256:135
  5120x2880   # 5K      16:9
  7680x4320   # 8K      16:9

示例终端命令:
  # 4K 60fps 播放（使用自定义 ckpt 与位移）
  python auto_play_highres.py \
      --static_ckpt /path/to/static_checkpoint.pt \
      --moving_ckpt /path/to/moving_checkpoint.pt \
      --vib_path   /path/to/displacements.dat \
      --scale_disp 50 \
      --resolution 3840x2160 --port 8080

  # 2K 30fps，端口 9000
  python auto_play_highres.py --resolution 2560x1440 --port 9000 --scale_disp 30

注意: 渲染分辨率越高，显存/算力占用越大；如遇掉帧请减小分辨率或降低 GUI "帧率" 滑块数值。

背景默认纯白，可在浏览器中二次合成。
"""

import argparse, threading, time, math, json, os, statistics
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import viser, nerfview
from gsplat.rendering import rasterization

# ---------- 1. 读取高斯 ----------
def load_splats(path, dev):
    ck = torch.load(path, map_location=dev)["splats"]
    return (
        ck["means"].cpu().numpy(),
        F.normalize(ck["quats"], p=2).cpu().numpy(),
        torch.exp(ck["scales"]).cpu().numpy(),
        torch.sigmoid(ck["opacities"]).cpu().numpy(),
        ck["sh0"].cpu().numpy(),
        ck["shN"].cpu().numpy(),
    )


def load_vibration(path, moving_count, scale=1.0):
    """Read (frames, moving Gaussians, xyz); raw .dat uses float32 in C order."""
    path = Path(path)
    if moving_count <= 0:
        raise ValueError("The moving checkpoint must contain at least one Gaussian.")
    if not np.isfinite(scale):
        raise ValueError("Displacement scale must be finite.")
    frame_size = moving_count * 3
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
        if raw.ndim == 1:
            if raw.size == 0 or raw.size % frame_size:
                raise ValueError("Flat vibration .npy must contain complete non-empty frames.")
            raw = raw.reshape(-1, moving_count, 3)
        elif raw.ndim != 3 or raw.shape[1:] != (moving_count, 3) or raw.shape[0] == 0:
            raise ValueError(f"Expected vibration shape (T, {moving_count}, 3), T > 0; got {raw.shape}.")
    elif path.suffix.lower() == ".dat":
        frame_bytes = frame_size * np.dtype(np.float32).itemsize
        if path.stat().st_size == 0 or path.stat().st_size % frame_bytes:
            raise ValueError("Raw vibration .dat must contain complete non-empty float32 frames.")
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, moving_count, 3)
    else:
        raise ValueError("Vibration input must be a .npy or raw float32 .dat file.")
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("Vibration values must be finite real numbers.")
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = np.asarray(raw, dtype=np.float32) * np.float32(scale)
    if not np.isfinite(displacement).all():
        raise ValueError("Scaled displacement is outside the finite float32 range.")
    return np.ascontiguousarray(displacement)

# ---------- 2. 全局 ----------
cur = 0           # 当前帧
iter_id = 0       # 迭代计数
T = None          # 总帧数
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ms_s = qs_s = ss_s = os_s = cs_s = None
ms_m = qs_m = ss_m = os_m = cs_m = None
disp_t = None
deg = 0           # 球谐阶
bench_stats = None


def initialize_scene(static_ckpt, moving_ckpt, vib_path, scale_disp=1.0, device=None):
    """Load the scene without opening a server or invoking the CUDA renderer."""
    global cur, iter_id, T, dev, deg, disp_t
    global ms_s, qs_s, ss_s, os_s, cs_s, ms_m, qs_m, ss_m, os_m, cs_m
    if device is not None:
        dev = torch.device(device)
    ms, qs, ss, os_, sh0s, shNs = load_splats(static_ckpt, dev)
    mm, qm, sm, om, sh0m, shNm = load_splats(moving_ckpt, dev)
    cs = np.concatenate([sh0s, shNs], axis=1)
    cm = np.concatenate([sh0m, shNm], axis=1)
    if cs.shape[1:] != cm.shape[1:] or math.isqrt(cs.shape[1]) ** 2 != cs.shape[1]:
        raise ValueError("Static and moving checkpoints must use the same complete SH degree.")
    disp = load_vibration(vib_path, mm.shape[0], scale_disp)
    to = lambda value: torch.from_numpy(value).float().to(dev)
    ms_s, qs_s, ss_s, os_s, cs_s = map(to, (ms, qs, ss, os_, cs))
    ms_m, qs_m, ss_m, os_m, cs_m = map(to, (mm, qm, sm, om, cm))
    disp_t = to(disp)
    deg = math.isqrt(cs.shape[1]) - 1
    T = disp.shape[0]
    if T <= 0:
        raise ValueError("At least one vibration frame is required.")
    cur = iter_id = 0


class RenderBenchmark:
    """记录 viewer 实际回调耗时，用于论文中的 FPS/ms-frame 指标。"""

    def __init__(self, log_path=None, window=30):
        self.log_path = log_path
        self.window = max(1, int(window))
        self.samples = []
        self.last_written = 0
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    def add(self, width, height, frame_id, ms_frame):
        fps = 1000.0 / ms_frame if ms_frame > 0 else float("inf")
        sample = {
            "time": time.time(),
            "frame": int(frame_id),
            "width": int(width),
            "height": int(height),
            "ms_frame": float(ms_frame),
            "actual_fps": float(fps),
        }
        self.samples.append(sample)
        if len(self.samples) % self.window == 0:
            self.report()

    def report(self):
        recent = self.samples[-self.window:]
        if not recent:
            return
        ms_vals = [s["ms_frame"] for s in recent]
        fps_vals = [s["actual_fps"] for s in recent]
        summary = {
            "n_recent": len(recent),
            "width": recent[-1]["width"],
            "height": recent[-1]["height"],
            "ms_frame_mean": statistics.fmean(ms_vals),
            "ms_frame_median": statistics.median(ms_vals),
            "ms_frame_min": min(ms_vals),
            "ms_frame_max": max(ms_vals),
            "actual_fps_mean": statistics.fmean(fps_vals),
            "actual_fps_median": statistics.median(fps_vals),
        }
        print(
            "[benchmark] "
            f"{summary['width']}x{summary['height']} | "
            f"Actual FPS mean/median = {summary['actual_fps_mean']:.2f}/"
            f"{summary['actual_fps_median']:.2f} | "
            f"ms/frame mean/median = {summary['ms_frame_mean']:.2f}/"
            f"{summary['ms_frame_median']:.2f}",
            flush=True,
        )
        if self.log_path and len(self.samples) != self.last_written:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary_recent": summary,
                        "samples": self.samples,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            self.last_written = len(self.samples)

# ---------- 3. 渲染回调生成器 ----------
def make_render_fn(width: int, height: int):
    white = torch.tensor([1.0, 1.0, 1.0], device=dev)  # 常量白色
    @torch.no_grad()
    def render_fn(cam_state: nerfview.CameraState, render_request):
        global cur, bench_stats
        # nerfview >= 0.0.3 passes RenderTabState; older versions pass (W, H).
        if isinstance(render_request, (tuple, list)):
            render_width, render_height = render_request
        elif render_request is not None and hasattr(render_request, "viewer_width"):
            render_width = render_request.viewer_width
            render_height = render_request.viewer_height
        else:
            render_width, render_height = width, height
        render_width, render_height = int(render_width), int(render_height)
        if render_width <= 0 or render_height <= 0:
            raise ValueError("Render width and height must be positive.")
        t0 = time.perf_counter()
        view = torch.from_numpy(cam_state.c2w).float().to(dev).inverse()
        Kmat = torch.from_numpy(cam_state.get_K((render_width, render_height))).float().to(dev)

        pos_m = ms_m + disp_t[cur]

        means  = torch.cat([ms_s, pos_m], dim=0)
        quats  = torch.cat([qs_s, qs_m], dim=0)
        scales = torch.cat([ss_s, ss_m], dim=0)
        opacs  = torch.cat([os_s, os_m], dim=0)
        colors = torch.cat([cs_s, cs_m], dim=0)

        rgb, alpha, _ = rasterization(
            means, quats, scales, opacs,
            colors.unsqueeze(0),
            view[None], Kmat[None],
            render_width, render_height,
            sh_degree=deg,
            render_mode="RGB",
            radius_clip=3.0,
        )

        rgb_out = rgb + (1.0 - alpha) * white           # 合成白底
        if dev.type == "cuda":
            torch.cuda.synchronize()
        out_np = (rgb_out[0].clamp(0.0, 1.0).flip(0) * 255).byte().cpu().numpy()
        if bench_stats is not None:
            bench_stats.add(render_width, render_height, cur, (time.perf_counter() - t0) * 1000.0)
        return out_np
    return render_fn


def create_viewer(width, height, host="127.0.0.1", port=8080):
    """Create the same real Viser/nerfview service used by the CLI and smoke check."""
    if width <= 0 or height <= 0:
        raise ValueError("Render width and height must be positive.")
    server = viser.ViserServer(host=host, port=port, verbose=False)
    try:
        server.scene.set_background_image(np.full((1, 1, 3), 255, dtype=np.uint8))
        fps_control = server.gui.add_slider("帧率", min=1, max=60, step=1, initial_value=30)
        frame_text = server.gui.add_text("当前帧", "0")
        viewer = nerfview.Viewer(server=server, render_fn=make_render_fn(width, height), mode="training")
        # nerfview adapts preview dimensions to the browser aspect ratio.
        # This public state is the maximum image dimension, including 4K/8K.
        viewer.render_tab_state.viewer_res = max(width, height)
        viewer.render_tab_state.num_train_rays_per_sec = 1.0
        return server, viewer, fps_control, frame_text
    except Exception:
        server.stop()
        raise

# ---------- 4. 主程序 ----------
def main():
    global bench_stats
    parser = argparse.ArgumentParser(description="Auto-play Gaussians 高分辨率 Viewer")
    parser.add_argument("--resolution", type=str, default="1920x1080",
                        help="渲染分辨率，如 1920x1080 / 3840x2160 / 任意 WxH")
    parser.add_argument("--port", type=int, default=8080, help="Viser 端口")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址，默认仅本机访问")
    parser.add_argument("--static_ckpt", type=str, required=True,
                        help="静态高斯 checkpoint (.pt)")
    parser.add_argument("--moving_ckpt", type=str, required=True,
                        help="动态高斯 checkpoint (.pt)")
    parser.add_argument("--vib_path", type=str, required=True,
                        help="位移文件 (.dat 或 .npy)")
    parser.add_argument("--scale_disp", type=float, default=50.0,
                        help="位移缩放因子")
    parser.add_argument("--benchmark_log", type=str, default="",
                        help="保存实际 FPS 与 ms/frame 的 JSON 文件路径")
    parser.add_argument("--benchmark_window", type=int, default=30,
                        help="每多少个实际渲染回调统计一次均值/中位数")
    args = parser.parse_args()
    W, H = map(int, args.resolution.lower().split('x'))
    if W <= 0 or H <= 0:
        parser.error("--resolution must have positive width and height.")

    ck_static  = args.static_ckpt
    ck_moving  = args.moving_ckpt
    vib_path   = args.vib_path
    scale_disp = args.scale_disp
    bench_stats = RenderBenchmark(args.benchmark_log or None, args.benchmark_window)

    # 触发前端重渲
    def force_render():
        global iter_id
        iter_id += 1
        viewer.render_tab_state.num_train_rays_per_sec = 1.0
        viewer.update(iter_id, 1)
        frame_text.value = f"{cur}/{T-1}"
        server.flush()

    initialize_scene(ck_static, ck_moving, vib_path, scale_disp)
    server, viewer, fps_control, frame_text = create_viewer(W, H, args.host, args.port)

    print(f"🌐 打开  http://{args.host}:{server.get_port()}   (Ctrl+C 退出)")
    print(f"🎬 渲染最大边长: {max(W, H)}，实际宽高随浏览器窗口调整")
    print(f"📊 Benchmark: 每 {args.benchmark_window} 个实际渲染回调输出 Actual FPS 和 ms/frame")

    # 5. 循环播放
    stop_playback = threading.Event()
    def tick():
        global cur
        while not stop_playback.wait(1.0 / fps_control.value):
            cur = (cur + 1) % T
            force_render()
    playback_thread = threading.Thread(target=tick, daemon=True)
    playback_thread.start()

    # 6. 阻塞主线程
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        stop_playback.set()
        playback_thread.join(timeout=2.0)
        server.stop()
        if bench_stats is not None:
            bench_stats.report()


if __name__ == "__main__":
    main()
