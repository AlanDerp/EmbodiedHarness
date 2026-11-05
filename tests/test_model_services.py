from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from embodied_harness.agents import AgentService
from embodied_harness.credentials import read_key
from embodied_harness.drivers.claude_code import ClaudeCodeDriver, public_message
from embodied_harness.rpc import RpcClient, RpcServer
from embodied_harness.runtime.selectors import JevSelector, RemoteSelector
from embodied_harness.services.system1 import System1Service
from test_contracts import scene


class ModelServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.keys = self.root/'system2.json'
        self.keys.write_text(json.dumps({'DEEPSEEK_API_KEY': 'fixture-private-key'}))
        self.keys.chmod(0o600)
        self.cli = self.root/'claude'
        self.cli.write_text('#!'+sys.executable+'\n'+'''
import json, os, sys, time
message=json.load(sys.stdin)
content=message['message']['content']
context=json.loads(content[0]['text'])
if context.get('hang'):
    print(os.environ['ANTHROPIC_AUTH_TOKEN'],file=sys.stderr,flush=True)
    time.sleep(30)
if context.get('fail'):
    print(os.environ['ANTHROPIC_AUTH_TOKEN'],file=sys.stderr)
    sys.exit(3)
reply=context.get('fixture_reply', {'cwd':os.getcwd(), 'config':os.environ['CLAUDE_CONFIG_DIR'],
      'api_url':os.environ['ANTHROPIC_BASE_URL'], 'token_present':bool(os.environ['ANTHROPIC_AUTH_TOKEN']),
      'other_secret_present':'TYPESAFE_API_KEY' in os.environ, 'arguments':sys.argv[1:],
      'image_count':sum(item['type']=='image' for item in content)})
print(json.dumps({'type':'system','subtype':'init'}))
print(json.dumps({'type':'result','subtype':'success','is_error':False,'result':json.dumps(reply), 'session_id':'fixture'}))
''')
        self.cli.chmod(0o700)

    def driver(self, **kwargs):
        return ClaudeCodeDriver(self.cli, self.keys, workspace_root=self.root, **kwargs)

    def serve(self, service):
        server = RpcServer(('127.0.0.1', 0), 'rpc', service.methods())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.shutdown();server.server_close();thread.join(timeout=3)
        self.addCleanup(close)
        return RpcClient('http://127.0.0.1:'+str(server.server_port), 'rpc')

    def test_real_child_has_images_and_fresh_context_but_no_system1_secret_or_tools(self):
        with patch.dict(os.environ, {'TYPESAFE_API_KEY':'must-not-forward'}):
            first=self.driver().respond('generate','Return JSON',{'public': {'observation':scene()}})
            second=self.driver().respond('generate','Return JSON',{'public': {'observation':scene()}})
        self.assertEqual(first['image_count'],1)
        self.assertFalse(first['other_secret_present'])
        self.assertTrue(first['token_present'])
        self.assertNotEqual(first['cwd'],second['cwd'])
        self.assertNotEqual(first['config'],second['config'])
        self.assertFalse(Path(first['cwd']).exists())
        self.assertEqual(first['api_url'],'https://api.deepseek.com/anthropic')
        args=first['arguments']
        self.assertEqual(args[args.index('--tools')+1],'')
        self.assertIn('--restricted',args)
        self.assertEqual(args[args.index('--output-format')+1],'stream-json')
        self.assertIn('--verbose',args)
        self.assertIn('--no-session-persistence',args)
        self.assertNotIn('--resume',args)
        self.assertNotIn('fixture-private-key',' '.join(args))

    def test_private_file_reload_empty_placeholder_mode_and_symlink(self):
        self.assertTrue(self.driver().configured())
        self.keys.write_text(json.dumps({'DEEPSEEK_API_KEY':'PASTE_DEEPSEEK_API_KEY_HERE'}))
        self.assertFalse(self.driver().configured())
        with self.assertRaises(ValueError):
            self.driver().respond('generate','',{})
        self.keys.write_text(json.dumps({'DEEPSEEK_API_KEY':'new-fixture-key'}))
        self.assertTrue(self.driver().configured())
        self.keys.chmod(0o644)
        with self.assertRaises(PermissionError):
            self.driver().configured()
        self.keys.chmod(0o600)
        link=self.root/'link.json';link.symlink_to(self.keys)
        with self.assertRaises(PermissionError):
            read_key(link,'DEEPSEEK_API_KEY')

    def test_cli_errors_do_not_expose_credentials_and_timeout_kills_child(self):
        with self.assertRaises(RuntimeError) as caught:
            self.driver().respond('generate','',{'fail':True})
        self.assertNotIn('fixture-private-key',str(caught.exception))
        with self.assertRaises(TimeoutError):
            self.driver(timeout_s=.3).respond('generate','',{'hang':True})
        self.assertFalse(list(self.root.glob('role-*')))

    def test_rejected_cli_response_is_audited_without_credentials(self):
        audit = self.root/'audit'
        with self.assertRaises(RuntimeError):
            self.driver(audit_dir=audit).respond('generate', '', {'fail': True})
        path = next(audit.glob('*.json'))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('fixture-private-key', path.read_text())
        record = json.loads(path.read_text())
        self.assertEqual(record['returncode'], 3)
        self.assertIn('[REDACTED]', record['stderr'])

    def test_timeout_preserves_partial_output_and_rpc_error_type(self):
        audit = self.root/'timeout-audit'
        driver = self.driver(timeout_s=.3, audit_dir=audit)
        service = AgentService(lambda:driver, lambda:{})
        with self.assertRaisesRegex(RuntimeError, 'RPC failed: TimeoutError'):
            self.serve(service).call('respond', role='generate', context={'hang':True})
        record = json.loads(next(audit.glob('*.json')).read_text())
        self.assertEqual(record['failure'], 'TimeoutExpired')
        self.assertIn('[REDACTED]', record['stderr'])
        self.assertNotIn('fixture-private-key', str(record))

    def test_image_request_is_real_rgb_and_preserves_calibration_metadata(self):
        message=json.loads(public_message({'observation':scene()}))
        context=json.loads(message['message']['content'][0]['text'])
        self.assertEqual(context['observation']['cameras']['head']['intrinsics'],scene().cameras['head'].intrinsics.tolist())
        image=message['message']['content'][2]
        self.assertEqual(image['type'],'image')
        self.assertEqual(image['source']['media_type'],'image/png')

    def test_remote_system1_calls_provider_and_enforces_pinned_response_model(self):
        keys=self.root/'system1.json';keys.write_text('{"TYPESAFE_API_KEY":"fixture-key"}');keys.chmod(0o600)
        def provider(payload):
            return {'model':'jev-1.13.0','answers':{'execution_parameter':{'type':'choice','choice':'fine','confidence':.9,
                    'probabilities':{'fine':.9,'normal':.1}}}}
        selector=JevSelector('jev-1.13.0',credentials_file=str(keys),transport=provider)
        rpc=self.serve(System1Service(selector))
        self.assertTrue(rpc.call('describe')['credentials_configured'])
        remote=RemoteSelector(rpc,'jev-1.13.0')
        result=remote.select({'position_error_m':.01},{'fine':'0.001 m','normal':'0.01 m'})
        self.assertEqual(result.option,'fine')
        with self.assertRaises(ValueError):
            RemoteSelector(rpc,'jev-1.14.0').select({}, {'fine':'0.001 m','normal':'0.01 m'})
        keys.write_text('{"TYPESAFE_API_KEY":""}')
        self.assertFalse(rpc.call('describe')['credentials_configured'])

    def test_claude_driver_is_wired_to_role_rpc_without_resuming_proposer(self):
        service=AgentService(lambda:self.driver(), lambda:{'credentials_configured':self.driver().configured()})
        rpc=self.serve(service)
        plan={'goal':'fixture goal','subtasks':[{'task_id':'one','goal':'fixture goal','source':'value = 1','parameters':{},'relation':None}]}
        context={'goal':'fixture goal','fixture_reply':plan}
        first=rpc.call('respond',role='generate',context=context)
        second=rpc.call('respond',role='generate',context=context)
        self.assertEqual(first['reply']['goal'],'fixture goal')
        self.assertNotEqual(first['agent_session_id'],second['agent_session_id'])
        self.assertTrue(rpc.call('describe')['credentials_configured'])
