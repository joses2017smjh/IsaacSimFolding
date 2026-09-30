"""Bounded diagnostic tracing for the unmodified upstream garment-switch path."""
import faulthandler
import json
import linecache
from pathlib import Path
import sys
import time


def trace_switch(env, garment, path, observer=None):
    with open(path, "w", buffering=1) as stream:
        def event(name, **kwargs):
            stream.write(json.dumps({"event": name, "monotonic": time.monotonic(),
                                     "unix_time": time.time(), **kwargs}) + "\n")
            stream.flush()

        functions = {"switch_garment", "_delete_garment_object", "_create_garment_object",
                     "load_garment_config", "initialize_obs", "initialize", "retarget",
                     "render", "step", "pause", "play", "collect"}
        def tracer(frame, kind, arg):
            name = frame.f_code.co_name
            if name in functions and kind in ("call", "line", "return", "exception"):
                filename = frame.f_code.co_filename
                event(kind, function=name, file=filename, line=frame.f_lineno,
                      source=linecache.getline(filename, frame.f_lineno).strip())
            return tracer

        old = sys.gettrace()
        faulthandler.dump_traceback_later(45, repeat=True, file=stream)
        event("switch_begin", target=garment)
        sys.settrace(tracer)
        try:
            env.switch_garment(garment)
            event("upstream_switch_returned")
            if observer is not None:
                event("camera_retarget_begin")
                category = "_".join(garment.split("_")[:2])
                directory = Path(observer.cfg.assets) / "objects/Challenge_Garment/Release" / category / garment
                observer.retarget(str(directory))
                event("camera_retarget_end")
        finally:
            sys.settrace(old)
            faulthandler.cancel_dump_traceback_later()
            event("switch_exit")
