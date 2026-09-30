"""Encode policy evidence directly from retained RGB, with explicit provenance."""
import bisect
import math
import subprocess
import numpy as np


def write_mp4(frames, path, step_dt, caption):
    from PIL import Image, ImageDraw, ImageFont
    steps = sorted(frames)
    first = frames[steps[0]]
    height, width = first["top_rgb"].shape[:2]
    width *= 3
    banner = 88
    fps = 30
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{width}x{height + banner}", "-framerate", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-n", str(path)]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in range(math.ceil(steps[-1] * step_dt * fps) + 1):
            simulation_step = index / fps / step_dt
            step = steps[max(0, bisect.bisect_right(steps, simulation_step) - 1)]
            pixels = np.concatenate([frames[step][key] for key in ("left_rgb", "top_rgb", "right_rgb")], axis=1)
            canvas = Image.new("RGB", (width, height + banner), (12, 12, 12))
            canvas.paste(Image.fromarray(pixels), (0, banner))
            ImageDraw.Draw(canvas).multiline_text((8, 6), caption, fill="white", font=font, spacing=3)
            proc.stdin.write(np.asarray(canvas).tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        proc.stdin.close()
        proc.stderr.close()
    return {"path": str(path), "fps": fps, "codec": "mpeg4", "source": "retained policy RGB; sample-and-hold at original simulation timestamps",
            "caption": caption, "predicted_final_frame_step": steps[-1]}
