"""Install the Bybit watcher and retire only the named legacy services on this Mac."""
from pathlib import Path
import argparse
import json
import os
import plistlib
import shlex
import shutil
import signal
import subprocess
import time

BASE = Path(__file__).resolve().parent
LABEL = 'com.crypto.near-momentum'
LEGACY = {'surge-alert', 'crypto-monitor'}


def legacy_command(command):
    try:
        words = shlex.split(command) if isinstance(command, str) else command
    except ValueError:
        return False
    return any(isinstance(w, str) and w.endswith('.py') and LEGACY.intersection(Path(w).parts) for w in words)


def inventory(home=Path.home()):
    jobs = []
    for path in (home / 'Library/LaunchAgents').glob('*.plist'):
        try:
            d = plistlib.loads(path.read_bytes())
            if legacy_command(d.get('ProgramArguments', [])):
                jobs.append({'path': str(path), 'label': d['Label']})
        except (OSError, ValueError, KeyError):
            continue
    ps = subprocess.run(['ps', '-axo', 'pid=,args='], capture_output=True, text=True, check=True)
    processes = []
    for line in ps.stdout.splitlines():
        pid, _, command = line.strip().partition(' ')
        if pid.isdigit() and int(pid) != os.getpid() and legacy_command(command):
            processes.append(int(pid))
    cron = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if cron.returncode not in (0, 1):
        raise RuntimeError('Could not inspect crontab')
    if cron.returncode == 1 and 'no crontab' not in cron.stderr.lower():
        raise RuntimeError('Crontab inspection failed; no changes made')
    lines = cron.stdout.splitlines()
    indexes = [i for i, line in enumerate(lines) if not line.lstrip().startswith('#') and legacy_command(line)]
    return jobs, processes, lines, indexes


def retire(jobs, processes, cron_lines, indexes):
    backup = BASE / 'output' / ('retired_' + str(time.time_ns()))
    backup.mkdir(parents=True, mode=0o700)
    domain = f'gui/{os.getuid()}'
    for job in jobs:
        loaded = subprocess.run(['launchctl', 'print', domain + '/' + job['label']], capture_output=True)
        if loaded.returncode == 0:
            subprocess.run(['launchctl', 'bootout', domain, job['path']], check=True)
        subprocess.run(['launchctl', 'disable', domain + '/' + job['label']], check=True)
        shutil.copy2(job['path'], backup / Path(job['path']).name)
    for pid in processes:
        # Recheck identity before signalling: PID could have been reused after inventory.
        p = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
        if p.returncode == 0 and legacy_command(p.stdout.strip()):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if indexes:
        original = '\n'.join(cron_lines) + '\n'
        saved = backup / 'crontab.txt'
        saved.write_text(original)
        saved.chmod(0o600)
        for i in indexes:
            cron_lines[i] = '# retired by near-momentum: ' + cron_lines[i]
        subprocess.run(['crontab', '-'], input='\n'.join(cron_lines) + '\n', text=True, check=True)
    return str(backup)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--install', action='store_true')
    action.add_argument('--retire-only', action='store_true')
    args = parser.parse_args()
    jobs, processes, cron, indexes = inventory()
    print(json.dumps({'legacy_jobs': jobs, 'legacy_processes': processes, 'legacy_cron_lines': [i + 1 for i in indexes],
                      'new_label': LABEL, 'apply': args.install or args.retire_only}, ensure_ascii=False, indent=2))
    if not (args.install or args.retire_only):
        return
    if args.install:
        node = shutil.which('node')
        if not node:
            raise RuntimeError('Node22+ required')
        # Require durable credentials; shell-only exports do not survive launchd login/restart.
        credential_files = [Path.home() / '.openclaw' / name for name in ('.env.slack-surge', '.env.slack-crypto')]
        if not any(p.is_file() for p in credential_files):
            raise RuntimeError('Configure SLACK_WEBHOOK_URL in ~/.openclaw/.env.slack-surge before installing')
        env = {k: v for k, v in os.environ.items() if 'SLACK_WEBHOOK' not in k}
        subprocess.run([node, str(BASE / 'scan.mjs'), '--slack', '--check'], env=env, check=True)
        output = BASE / 'output'
        output.mkdir(exist_ok=True)
        target = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')
        target.parent.mkdir(parents=True, exist_ok=True)
        spec = {'Label': LABEL, 'ProgramArguments': [node, str(BASE / 'scan.mjs'), '--slack'],
                'WorkingDirectory': str(BASE.parent), 'StartInterval': 300, 'RunAtLoad': True,
                'StandardOutPath': str(output / 'service.log'), 'StandardErrorPath': str(output / 'service-error.log')}
        # Do not overwrite an unrelated plist that happens to share our filename.
        if target.exists():
            old = plistlib.loads(target.read_bytes())
            if old.get('Label') != LABEL or str(BASE / 'scan.mjs') not in old.get('ProgramArguments', []):
                raise RuntimeError('Existing service path differs; inspect before replacing')
        domain = f'gui/{os.getuid()}'
        if subprocess.run(['launchctl', 'print', domain + '/' + LABEL], capture_output=True).returncode == 0:
            subprocess.run(['launchctl', 'bootout', domain, str(target)], check=True)
        target.write_bytes(plistlib.dumps(spec))
        subprocess.run(['launchctl', 'enable', domain + '/' + LABEL], check=True)
        subprocess.run(['launchctl', 'bootstrap', domain, str(target)], check=True)
        subprocess.run(['launchctl', 'print', domain + '/' + LABEL], capture_output=True, check=True)
    backup = retire(jobs, processes, cron, indexes)
    print(json.dumps({'installed': args.install, 'backup': backup, 'note': 'Only this Mac was changed. No SSH or remote deployment.'}))


if __name__ == '__main__':
    main()
