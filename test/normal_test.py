import unittest
from method.normal import Normal
import tempfile
import numpy as np
import nibabel as nib
from upath import UPath


class TestNormal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmps: list[UPath] = []

        def make_random_file(dtype):
            data = (np.random.rand(64, 64, 32) * 100).astype(dtype)
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp_path = UPath(tmp.name)
                cls.tmps.append(tmp_path)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                nib.save(img, tmp_path)
            return data, tmp_path

        cls.base_data, cls.base_tmp_path = make_random_file(np.float32)
        cls.gt_data, cls.gt_tmp_path = make_random_file(np.uint32)
        cls.fn0_data, cls.fn0_tmp_path = make_random_file(np.uint32)

    @classmethod
    def tearDownClass(cls):
        for tmp in cls.tmps:
            tmp.unlink()

    def take(self, base, gt, fn):
        normal = Normal()
        processed = normal.process(base, gt, fn)
        base = processed["base"]
        gt = processed["gt"]
        fn = [processed[f"fn{i}"] for i in range(len(fn))]
        return base, gt, fn

    def test_process(self):
        base, gt, fn = self.take(
            UPath(self.base_tmp_path),
            UPath(self.gt_tmp_path),
            [UPath(self.fn0_tmp_path)],
        )
        np.testing.assert_array_equal(base, self.base_data)
        np.testing.assert_array_equal(gt, self.gt_data)
        np.testing.assert_array_equal(fn, [self.fn0_data])
