"""Fetch the private release and install only manifest-listed, verified files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_relative(value: str) -> Path:
    part = PurePosixPath(value)
    if not value or not part.parts or part.is_absolute() or any(p in ('..', '.') or ':' in p or '\\' in p for p in part.parts):
        raise ValueError(f'Invalid asset path: {value}')
    return Path(*part.parts)


def destination(entry: dict, project: Path, face_home: Path) -> Path:
    roots = {'project': project, 'face-home': face_home}
    root = roots[entry['root']].resolve()
    target = (root / checked_relative(entry['target'])).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Asset destination escapes its root: {entry["target"]}')
    return target


def verify_archive(archive: Path, asset: dict) -> None:
    if sha256(archive) != asset['sha256']:
        raise ValueError(f'Archive checksum mismatch: {archive.name}')
    expected = {item['member']: item for item in asset['files']}
    if len(expected) != len(asset['files']):
        raise ValueError('Duplicate manifest members')
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(f'Unexpected archive members: {archive.name}')
        for name, item in expected.items():
            checked_relative(name)
            info = bundle.getinfo(name)
            if info.file_size != item['size']:
                raise ValueError(f'File size mismatch: {name}')
            digest = hashlib.sha256()
            with bundle.open(name) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise ValueError(f'File checksum mismatch: {name}')


def install_archive(archive: Path, asset: dict, project: Path, face_home: Path) -> None:
    # Verify the ENTIRE archive before writing any file.
    verify_archive(archive, asset)
    with zipfile.ZipFile(archive) as bundle:
        for item in asset['files']:
            target = destination(item, project, face_home)
            if target.exists():
                if item.get('preserve_existing') or sha256(target) == item['sha256']:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='.detect-', dir=target.parent)
            try:
                with os.fdopen(descriptor, 'wb') as output, bundle.open(item['member']) as stream:
                    shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', default=['core', 'config'])
    parser.add_argument('--repo', default='')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.downloads')
    parser.add_argument('--face-home', type=Path, default=Path.home() / '.insightface')
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets-manifest.json').read_text(encoding='utf-8'))
    valid = {asset['group'] for asset in manifest['assets']}
    if set(args.groups) - valid:
        parser.error(f'Unknown groups: {set(args.groups) - valid}')
    repository = args.repo or manifest['repository']
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        if asset['group'] not in args.groups:
            continue
        archive = args.cache_dir / checked_relative(asset['name'])
        if not archive.exists():
            gh = shutil.which('gh')
            if not gh:
                raise SystemExit('Install GitHub CLI (https://cli.github.com/) and run gh auth login, or copy all release ZIPs into .downloads/.')
            subprocess.run([gh, 'release', 'download', manifest['tag'], '--repo', repository,
                            '--pattern', asset['name'], '--dir', str(args.cache_dir)], check=True)
        if args.verify_only:
            verify_archive(archive, asset)
        else:
            install_archive(archive, asset, args.project, args.face_home)
        print(f'OK: {asset["name"]}', flush=True)


if __name__ == '__main__':
    main()

