"""Run the unchanged simulator while enforcing the user's no-new-documents rule."""
import builtins
import io
import os
from pathlib import Path
import runpy
import sys

def deny_new_doc(path,writing):
    if isinstance(path,(str,bytes,os.PathLike)) and writing:
        p=Path(os.fsdecode(path))
        if p.suffix.lower() in {'.md','.txt'} and not p.exists():
            raise PermissionError(f'User prohibits creating new documentation files: {p}')

def install_guard():
    ordinary_open=builtins.open
    io_open=io.open
    os_open=os.open
    def checked_open(file,mode='r',*args,**kwargs):
        deny_new_doc(file,any(x in mode for x in 'wax+'))
        return ordinary_open(file,mode,*args,**kwargs)
    def checked_io(file,mode='r',*args,**kwargs):
        deny_new_doc(file,any(x in mode for x in 'wax+'))
        return io_open(file,mode,*args,**kwargs)
    def checked_os(file,flags,*args,**kwargs):
        deny_new_doc(file,bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT)))
        return os_open(file,flags,*args,**kwargs)
    builtins.open=checked_open
    io.open=checked_io
    os.open=checked_os

if __name__=='__main__':
    runner=Path(sys.argv[1]).resolve()
    sys.argv=sys.argv[1:]
    sys.path.insert(0,str(runner.parent))
    install_guard()
    runpy.run_path(str(runner),run_name='__main__')
