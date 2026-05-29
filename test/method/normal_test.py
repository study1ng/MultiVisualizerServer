import unittest
from pathlib import Path
from method.normal import Normal


class TestNormal(unittest.TestCase):
    def take(self, base, gt, fn):
        normal = Normal()
        processed = normal.process(base, gt, fn)
        base = processed["base"]
        gt = processed["gt"]
        fn = processed["fn"]
        return base, gt, fn

    def test_process(self):
        base, gt, fn = self.take(
            Path("samples/base.nii.gz"),
            Path("samples/gt.nii.gz"),
            [Path("samples/out.nii.gz")],
        )
        self.assertEqual()
