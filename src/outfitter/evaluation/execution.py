"""Local process budget and exclusive artifact locks; no environment state."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gzip
import json
import os
from pathlib import Path
import tempfile
import time


def default_cpu_budget():
    return min(8, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'evaluation workspace already in use: {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def execution_budget(size=None, directory=None):
    size = default_cpu_budget() if size is None else int(size)
    if size < 1:
        raise ValueError('cpu_budget must be positive')
    if directory is not None:
        root = Path(directory)
        config = json.loads((root/'budget.json').read_text())
        if config['size'] != size:
            raise ValueError('shared CPU budget size mismatch')
        yield str(root), size
        return
    with tempfile.TemporaryDirectory(prefix='outfitter-eval-cpu-') as name:
        root = Path(name)
        (root/'budget.json').write_text(json.dumps(dict(size=size)))
        for index in range(size):
            (root/f'slot-{index}').touch()
        yield name, size


class CpuLease:
    """fcntl locks are automatically released on exit/crash; children own no FD."""
    def __init__(self, directory, size):
        self.directory = Path(directory)
        self.size = size
        self.handle = None

    def acquire(self):
        if self.handle is not None:
            return self
        while True:
            for index in range(self.size):
                handle = (self.directory/f'slot-{index}').open('r+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                self.handle = handle
                return self
            time.sleep(.05)

    def release(self):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()

    @contextmanager
    def suspended(self):
        self.release()
        try:
            yield
        finally:
            self.acquire()


def write_state(path, value):
    """Preserve insertion order: state dictionaries are NOT report dictionaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(gzip.compress(payload, compresslevel=1, mtime=0))
    temporary.replace(path)


def read_state(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def worker_environment():
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def occupied_cpu_slots(directory,size):
    """Instantaneous diagnostic; locks, not stale owner text, are authoritative."""
    occupied=0
    for index in range(size):
        with (Path(directory)/f'slot-{index}').open('r+') as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                occupied+=1
            else:
                fcntl.flock(handle,fcntl.LOCK_UN)
    return occupied
