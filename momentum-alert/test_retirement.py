import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import service


class Retirement(unittest.TestCase):
    def test_match_only_named_legacy_directories(self):
        self.assertTrue(service.legacy_command('python3 /Users/a/.openclaw/workspace/surge-alert/surge_alert.py'))
        self.assertTrue(service.legacy_command(['python3', '/apps/crypto-tools/crypto-monitor/market_monitor.py']))
        self.assertFalse(service.legacy_command('python3 /apps/crypto-paper-trading/crypto_monitor.py'))
        self.assertFalse(service.legacy_command('node /apps/momentum-alert/scan.mjs --slack'))

    def test_all_legacy_senders_return_without_network(self):
        root = Path(__file__).resolve().parent.parent
        found = 0
        for folder in ('surge-alert', 'crypto-monitor'):
            for p in (root / folder).glob('*.py'):
                tree = ast.parse(p.read_text())
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == 'send_slack':
                        isolated = ast.Module(body=[node], type_ignores=[])
                        namespace = {}
                        exec(compile(isolated, str(p), 'exec'), namespace)
                        self.assertIsNone(namespace['send_slack']('test'))
                        found += 1
        self.assertEqual(found, 15)

    def test_live_entrypoints_exit_before_work(self):
        root = Path(__file__).resolve().parent.parent
        files = ['surge-alert/' + x for x in ('surge_alert.py', 'ml_scanner.py', 'ml_scanner_v2.py',
                  'ignition_scanner.py', 'prebreakout_scanner.py', 'surge_detector_v3.py')]
        files.append('crypto-monitor/market_monitor.py')
        for name in files:
            tree = ast.parse((root / name).read_text())
            guard = next(x for x in tree.body if isinstance(x, ast.If) and isinstance(x.test, ast.Compare)
                         and isinstance(x.test.left, ast.Name) and x.test.left.id == '__name__')
            self.assertIsInstance(guard.body[0], ast.Raise, name)

    def test_retirement_does_not_kill_reused_pid(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(service, 'BASE', Path(tmp)), \
             patch.object(service.subprocess, 'run') as run, patch.object(service.os, 'kill') as kill:
            run.return_value.returncode = 0
            run.return_value.stdout = 'node /apps/momentum-alert/scan.mjs'
            service.retire([], [1234], [], [])
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
