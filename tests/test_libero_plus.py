from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from embodied_harness.adapters.libero_plus import layout_catalog, LiberoPlusNative
from embodied_harness.campaign import partition
from test_runtime import profile

class LiberoTests(unittest.TestCase):
    def test_single_panda_arm_does_not_relax_arx_contract(self):
        base=profile()
        panda=replace(base,embodiment='libero_panda',arms={'right':base.arms['right']})
        self.assertEqual(set(panda.arms),{'right'})
        with self.assertRaises(ValueError): replace(base,arms={'right':base.arms['right']})
    def test_catalog_fingerprints_content_and_rejects_duplicate_partitions(self):
        with TemporaryDirectory() as folder:
            bddl=Path(folder)/'task.bddl'; bddl.write_text('goal')
            states=np.array([[1.,2.],[2.,3.],[1.,2.]])
            with patch('embodied_harness.adapters.libero_plus.task_data',return_value=(None,states,bddl)):
                catalog=layout_catalog(folder,'libero_spatial:0')
            self.assertEqual(catalog[0]['layout_sha256'],catalog[2]['layout_sha256'])
            with self.assertRaises(ValueError): partition(catalog,[0],[1],[2])
    def test_scene_outside_bank_fails_before_simulator_start(self):
        with patch('embodied_harness.adapters.libero_plus.task_data',return_value=(None,np.zeros((1,3)),None)):
            with self.assertRaises(ValueError): LiberoPlusNative('/unused','libero_spatial:0',1,'/unused')

    def test_virtual_camera_variant_uses_base_bddl_and_distinct_identity(self):
        with TemporaryDirectory() as folder:
            base=Path(folder)/'pick.bddl'; base.write_text('goal')
            first=Path(folder)/'pick_view_0_0_100_0_0_initstate_1.bddl'
            second=Path(folder)/'pick_view_10_0_100_0_0_initstate_1.bddl'
            hashes=[]
            for path in (first,second):
                with patch('embodied_harness.adapters.libero_plus.task_data',return_value=(None,np.ones((1,3)),path)):
                    hashes.append(layout_catalog(folder,'libero_spatial:0')[0]['layout_sha256'])
            self.assertNotEqual(*hashes)

if __name__=='__main__': unittest.main()
