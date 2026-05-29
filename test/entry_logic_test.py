from io import BytesIO
import unittest
import tempfile
import numpy as np
import nibabel as nib
from upath import UPath
from errors import InternalError
from main import entry_logic
import zlib
from utils import map_dict


class TestEntryLogic(unittest.TestCase):
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

    @staticmethod
    def take(data):
        def _take(v: bytes | None):
            if v is None:
                return None
            v = zlib.decompress(v)
            with BytesIO(v) as buffer:
                buffer.seek(0)
                arr = np.load(buffer)
            return arr

        data = map_dict(_take, data)
        base = data["base"]
        gt = data["gt"]
        fn = [data[f"fn{i}"] for i in range(len(data) - 2)]
        return base, gt, fn

    def test_axial_normal(self):
        data, payload = entry_logic(
            "ax=axial,process=normal",
            self.base_tmp_path,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = self.take(data)
        np.testing.assert_equal(self.base_data, base)
        np.testing.assert_equal(self.gt_data, gt)
        np.testing.assert_equal([self.fn0_data], fn)

        data, payload = entry_logic(
            "ax=axial,process=normal",
            None,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = self.take(data)
        self.assertIsNone(base)
        np.testing.assert_equal(self.gt_data, gt)
        np.testing.assert_equal([self.fn0_data], fn)

        data, payload = entry_logic(
            "ax=axial,process=normal",
            self.base_tmp_path,
            None,
            [self.fn0_tmp_path],
        )
        base, gt, fn = self.take(data)
        np.testing.assert_equal(self.base_data, base)
        self.assertIsNone(gt)
        np.testing.assert_equal([self.fn0_data], fn)

        with self.assertRaises(InternalError):
            data, payload = entry_logic(
                "ax=axial,process=normal",
                None,
                None,
                None,
            )

    def test_saggital_normal(self):
        data, payload = entry_logic(
            "ax=saggital,process=normal",
            self.base_tmp_path,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = self.take(data)

        def _saggital(data):
            return np.transpose(data, (1, 2, 0))

        np.testing.assert_equal(_saggital(self.base_data), base)
        np.testing.assert_equal(_saggital(self.gt_data), gt)
        np.testing.assert_equal([_saggital(self.fn0_data)], fn)

    def test_coronal_normal(self):
        data, payload = entry_logic(
            "ax=coronal,process=normal",
            self.base_tmp_path,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = self.take(data)

        def _coronal(data):
            return np.transpose(data, (0, 2, 1))

        np.testing.assert_equal(_coronal(self.base_data), base)
        np.testing.assert_equal(_coronal(self.gt_data), gt)
        np.testing.assert_equal([_coronal(self.fn0_data)], fn)
