"""Bootstrap new evaluator code alongside, but not over, a frozen environment."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    root=args.source_root or args.evaluation_root.parents[1]
    if (root/'src/outfitter/runner.py').is_file():root=root/'src'
    if not (root/'outfitter/runner.py').is_file():raise ValueError('source-root must contain outfitter or src/outfitter')
    sys.path.insert(0,str(root.resolve()))
    import outfitter
    spec=importlib.util.spec_from_file_location('outfitter.evaluation',args.evaluation_root/'__init__.py',
                                              submodule_search_locations=[str(args.evaluation_root)])
    module=importlib.util.module_from_spec(spec)
    sys.modules['outfitter.evaluation']=module
    spec.loader.exec_module(module)
    def no_network(*a,**kw):raise RuntimeError('network access forbidden in offline evaluator')
    socket.socket.connect=no_network
    socket.create_connection=no_network
    request=json.loads(args.request.read_text())
    kind=request.pop('task','replay')
    if kind=='relax':
        from outfitter.evaluation.parallel import run_branch
        run_branch(**request)
    elif kind=='replay':
        from outfitter.evaluation.extended import replay
        request['source_root']=str(root.resolve())
        replay(**request)
    else:
        raise ValueError(f'unknown offline worker task {kind}')


if __name__=='__main__':main()
