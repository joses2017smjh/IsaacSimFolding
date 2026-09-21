"""Fresh Storm file acquisition and an actual cloth-visibility difference pass."""
import os
import time
import numpy as np
from lehome_fold.storm_obs import StormObserver


def read_rgb(path):
    from PIL import Image
    with Image.open(path) as source:
        if source.mode == "RGBA":
            source = Image.alpha_composite(Image.new("RGBA", source.size, (150, 150, 150, 255)), source)
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


class StrictStormObserver(StormObserver):
    def __init__(self, config):
        super().__init__(config)
        self.update_serial = 0
        self.render_serial = 0
        self.last_acquisition = None

    def update(self, particles, link_poses):
        super().update(particles, link_poses)
        self.update_serial += 1

    def render(self):
        from pxr import Usd, UsdGeom
        started = time.monotonic()
        # If Record reports success without writing, there is no old file to read.
        for key in self._cams:
            path = os.path.join(self.workdir, key + ".png")
            if os.path.exists(path):
                os.unlink(path)
        images = super().render()
        self.render_serial += 1
        # Exact rendered garment contribution, not a red-pixel heuristic.
        imageable = UsdGeom.Imageable(self._points.GetPrim())
        attr = imageable.CreateVisibilityAttr()
        original_visibility = attr.Get()
        background = os.path.join(self.workdir, "cloth_hidden_visibility.png")
        if os.path.exists(background):
            os.unlink(background)
        try:
            attr.Set(UsdGeom.Tokens.invisible)
            if not self._rec.Record(self._stage, self._cams["top_rgb"], Usd.TimeCode.Default(), background):
                raise RuntimeError("Garment visibility render failed")
            without_cloth = read_rgb(background)
        finally:
            attr.Set(original_visibility or UsdGeom.Tokens.inherited)
        delta = np.abs(images["top_rgb"].astype(np.int16) - without_cloth.astype(np.int16))
        garment_pixels = int(np.any(delta > 8, axis=2).sum())
        if garment_pixels < 16:
            raise ValueError(f"Garment not visible in top RGB: {garment_pixels} changed pixels")
        self.last_acquisition = {"render_serial": self.render_serial,
            "update_serial": self.update_serial, "fresh_camera_count": 3,
            "old_files_removed_before_record": True, "top_garment_pixels": garment_pixels,
            "visibility_method": "same-state top RGB minus cloth-hidden render, channel difference >8",
            "wall_seconds": time.monotonic() - started}
        return images
