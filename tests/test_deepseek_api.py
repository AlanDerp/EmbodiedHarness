import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_harness.drivers.deepseek_api import DeepSeekApiDriver
from test_contracts import scene


class DeepSeekApiTests(unittest.TestCase):
    def test_explicit_mode_images_and_fresh_messages(self):
        with tempfile.TemporaryDirectory() as folder:
            keys = Path(folder)/'keys.json'
            keys.write_text('{"DEEPSEEK_API_KEY":"fixture-key"}'); keys.chmod(0o600)
            calls = []
            envelope = {'id':'request', 'model':'deepseek-flash', 'stop_reason':'end_turn',
                        'content':[{'type':'text','text':'{"ready":true}'}]}
            def respond(request, timeout):
                calls.append(json.loads(request.data))
                return io.BytesIO(json.dumps(envelope).encode())
            with patch('embodied_harness.drivers.deepseek_api.urllib.request.build_opener') as build:
                build.return_value.open.side_effect = respond
                driver = DeepSeekApiDriver(keys, audit_dir=Path(folder)/'audit')
                self.assertEqual(driver.respond('generate', 'contract', {'observation':scene()}), {'ready':True})
                driver.respond('verify', 'contract', {})
                self.assertEqual(calls[0]['thinking'], {'type':'disabled'})
                self.assertNotIn('tools', calls[0])
                self.assertTrue(any(x['type']=='image' for x in calls[0]['messages'][0]['content']))
                self.assertEqual(len(calls[1]['messages']), 1)
                self.assertNotIn('fixture-key', json.dumps(calls))
                envelope['stop_reason'] = 'max_tokens'
                with self.assertRaisesRegex(RuntimeError, 'incomplete response'):
                    driver.respond('repair', 'contract', {})
                envelope['stop_reason'] = 'end_turn'
                envelope['content'] = [{'type':'thinking','thinking':'not allowed'}]
                with self.assertRaisesRegex(RuntimeError, 'text-only'):
                    driver.respond('repair', 'contract', {})
                envelope['content'].append({'type':'text','text':'{"repaired":true}'})
                thinking_driver = DeepSeekApiDriver(keys, thinking='enabled', max_tokens=131072, timeout_s=1800)
                self.assertEqual(thinking_driver.respond('repair', 'contract', {}), {'repaired':True})
                self.assertEqual(calls[-1]['thinking'], {'type':'enabled'})
                self.assertEqual(calls[-1]['max_tokens'], 131072)
                envelope['stop_reason'] = 'max_tokens'
                with self.assertRaisesRegex(RuntimeError, 'incomplete response'):
                    thinking_driver.respond('repair', 'contract', {})
            for p in (Path(folder)/'audit').glob('*.json'):
                self.assertEqual(p.stat().st_mode & 0o777, 0o600)
                self.assertNotIn('fixture-key', p.read_text())
