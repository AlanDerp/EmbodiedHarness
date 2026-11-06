from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import unittest

from embodied_harness import wire
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.artifacts.events import EventLog
from embodied_harness.control import Executor
from embodied_harness.runtime import Session
from embodied_harness.runtime.selectors import Selection, prior_with_jev_override
from test_contracts import scene
from test_runtime import DirectRemote, NativeFixture, RoboDojoWorker, profile


class PriorTests(unittest.TestCase):
    def test_threshold_invalid_and_error_fallback_and_stale_prior(self):
        source=scene().source
        prior={'option':'fine','reason':'small observed pose error','source':source}
        for confidence,expected in [(0,'fine'),(.8,'fine'),(.80001,'normal')]:
            selector=Mock();selector.select.return_value=Selection('normal',confidence)
            result=prior_with_jev_override(selector,{}, {'fine':'1mm','normal':'10mm'},prior,source,.8)
            self.assertEqual(result['option'],expected)
            self.assertEqual(result['jev_confidence'],confidence)
        selector.select.side_effect=RuntimeError('unavailable')
        result=prior_with_jev_override(selector,{}, {'fine':'1mm'},prior,source,.8)
        self.assertEqual(result['option'],'fine');self.assertEqual(result['jev_error_type'],'RuntimeError')
        selector.reset_mock();selector.select.side_effect=None
        selector.select.return_value=Selection('missing',.99)
        self.assertEqual(prior_with_jev_override(selector,{}, {'fine':'1mm'},prior,source,.8)['option'],'fine')
        selector.reset_mock()
        with self.assertRaises(ValueError):
            prior_with_jev_override(selector,{}, {'fine':'1mm'}, {**prior,'source':replace(source,state_version=source.state_version+1)},source,.8)
        selector.select.assert_not_called()

    def test_real_skill_prior_changes_motion_without_low_confidence_stop(self):
        for confidence,step in [(.1,.001),(.8,.001),(.81,.01)]:
            with self.subTest(confidence=confidence), TemporaryDirectory() as folder:
                selector=Mock();selector.select.return_value=Selection('normal',confidence)
                native=NativeFixture();worker=RoboDojoWorker(native,profile(),'episode',native_action_limit=100)
                session=Session(Executor(DirectRemote(worker)),SnapshotStore(Path(folder)/'snapshots'),
                    EventLog(Path(folder)/'events.jsonl'),selector=selector,parameter_selection_mode='system2_prior_jev_override')
                source=(Path(__file__).parents[1]/'examples/ee_skill.py').read_text()
                source += '''
base_context = skill_context
def skill_context(state):
    context = base_context(state)
    context['parameter_prior'] = {'option':'fine', 'reason':'conservative observed displacement', 'source':state['observation'].source}
    return context
'''
                session.delegate_python('prior','move',source,{'arm':'left','target':{'position':[.03,0,0],'quaternion_wxyz':[1,0,0,0],'frame':'world'}})
                events=[wire.loads(x) for x in (Path(folder)/'events.jsonl').read_bytes().splitlines()]
                action=next(x['action'] for x in events if x['kind']=='action_intent')
                self.assertAlmostEqual(action.arms['left'].ee_target.position[0],step)
                decision=next(x for x in events if x['kind']=='parameter_arbitration')
                self.assertEqual(decision['prior']['option'],'fine')
                self.assertEqual(decision['jev_confidence'],confidence)
                state=selector.select.call_args.args[0]
                self.assertNotIn('images',state);self.assertNotIn('cameras',state);self.assertNotIn('instruction',state)
