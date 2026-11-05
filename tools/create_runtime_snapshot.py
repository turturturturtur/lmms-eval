"""Copy executable evaluation sources so queued/running jobs cannot change underneath them."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def create_snapshot(source, destination, worker=None, judge=None):
    destination.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', 'node_modules', '.git', 'web')
    for name in ('lmms_eval', 'run_scripts', 'tools'):
        shutil.copytree(source / name, destination / name, ignore=ignore)
    for name in ('pyproject.toml',):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    for override, name in ((worker, 'qwen35_worker.sh'), (judge, 'run_judge.sh')):
        if override is not None:
            shutil.copy2(override, destination / 'run_scripts' / name)
    manifest = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            manifest[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--judge', type=Path)
    args = parser.parse_args()
    create_snapshot(args.source, args.destination, args.worker, args.judge)


if __name__ == '__main__':
    main()
