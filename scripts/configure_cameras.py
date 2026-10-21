"""Set local camera connections without publishing credentials or logging in."""
from __future__ import annotations

import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'camera/config/cameras.json'
DOCUMENTATION_NETS = [ipaddress.ip_network(value) for value in ('192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24')]


def valid_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
        return (address.version == 4 and not address.is_unspecified and not address.is_loopback
                and not address.is_multicast and not any(address in net for net in DOCUMENTATION_NETS))
    except ValueError:
        return False


def issues(config: dict) -> list[str]:
    problems = []
    seen = set()
    if not config.get('cameras'):
        problems.append('No cameras configured.')
    for camera in config.get('cameras', []):
        camera_id = camera.get('id', '')
        if not camera_id or camera_id in seen:
            problems.append('Camera IDs must be nonempty and unique.')
        seen.add(camera_id)
        if not valid_ip(str(camera.get('ip', ''))):
            problems.append(f'{camera_id}: replace the example IP with the actual camera IPv4 address.')
        if not str(camera.get('username', '')).strip() or not camera.get('password'):
            problems.append(f'{camera_id}: fill in the local camera username and password.')
        try:
            if not 1 <= int(camera.get('port', 0)) <= 65535 or int(camera.get('channel', 0)) < 1:
                raise ValueError()
        except (ValueError, TypeError):
            problems.append(f'{camera_id}: invalid SDK port or channel.')
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate local settings without connecting.')
    args = parser.parse_args()
    if not CONFIG.exists():
        raise SystemExit('Run install.ps1 to restore the configuration first.')
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    if args.check:
        problems = issues(config)
        if problems:
            raise SystemExit('\n'.join(problems))
        print('Camera configuration OK. No login was attempted.')
        return
    print('Settings stay in the ignored local camera/config/cameras.json. No camera login is attempted.')
    print('Keep camera IDs unchanged to preserve their floor-plan calibration. Enter keeps existing values.')
    for camera in config['cameras']:
        print(f'\n{camera["id"]}: {camera.get("name", "")}')
        camera['ip'] = input(f'Camera IPv4 [{camera.get("ip", "")}]: ').strip() or camera.get('ip', '')
        camera['username'] = input(f'Username [{camera.get("username", "")}]: ').strip() or camera.get('username', '')
        camera['password'] = getpass.getpass('Password (hidden; Enter keeps existing): ') or camera.get('password', '')
        try:
            camera['port'] = int(input(f'SDK port [{camera.get("port", 8000)}]: ').strip() or camera.get('port', 8000))
            camera['channel'] = int(input(f'Channel [{camera.get("channel", 1)}]: ').strip() or camera.get('channel', 1))
        except ValueError:
            raise SystemExit('Port and channel must be integers; settings were not saved.')
    problems = issues(config)
    if problems:
        raise SystemExit('\n'.join(problems) + '\nSettings were not saved; run this command again.')
    descriptor, temporary = tempfile.mkstemp(prefix='.cameras-', dir=CONFIG.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            json.dump(config, output, ensure_ascii=False, indent=2)
        os.replace(temporary, CONFIG)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print('Local configuration saved. Start Detect with start.ps1.')


if __name__ == '__main__':
    main()
