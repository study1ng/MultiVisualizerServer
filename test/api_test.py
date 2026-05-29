from io import BytesIO
import unittest
import tempfile
import numpy as np
import nibabel as nib
from upath import UPath
import zlib
from utils import map_dict
from fastapi.testclient import TestClient
from main import app
from urllib.parse import quote
from zipfile import ZipFile

client = TestClient(app)


class TestAPI(unittest.TestCase):
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

    def get(
        self,
        view_method: str,
        base: UPath | None,
        gt: UPath | None,
        fn: list[UPath] | None,
    ):
        view_method = quote(view_method)
        base = "" if base is None else quote(str(base))
        gt = "" if gt is None else quote(str(gt))
        fn = "" if fn is None else [quote(str(f)) for f in fn]
        url = f"/{view_method}/?base={base}&gt={gt}"
        for i in range(len(fn)):
            url += f"&fn={fn[i]}"
        response = client.get(url)
        self.assertEqual(response.status_code, 200)
        return response.text.strip('"')

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
        if "base" not in data:
            data["base"] = None
        if "gt" not in data:
            data["gt"] = None

        base = data["base"]
        gt = data["gt"]
        fn = [data[f"fn{i}"] for i in range(len(data) - 2)]
        return base, gt, fn

    def download(self, uid):
        response = client.get(f"/{uid}.zip")
        self.assertEqual(response.status_code, 200)
        zipfile = ZipFile(BytesIO(response.content))
        out = {}
        for file in zipfile.namelist():
            out[file] = zipfile.read(file)
        payload = out.pop("payload.json")
        data = {}
        for k, v in out.items():
            self.assertTrue(k.endswith(".bin"))
            data[k[:-4]] = v
        out = data
        out = self.take(out)
        return out, payload

    def follow(
        self,
        view_method: str,
        base: UPath | None,
        gt: UPath | None,
        fn: list[UPath] | None,
    ):
        return self.download(self.get(view_method, base, gt, fn))

    def test_axial_normal(self):
        data, payload = self.follow(
            "ax=axial,process=normal",
            self.base_tmp_path,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = data
        np.testing.assert_equal(self.base_data, base)
        np.testing.assert_equal(self.gt_data, gt)
        np.testing.assert_equal([self.fn0_data], fn)

        data, payload = self.follow(
            "ax=axial,process=normal",
            None,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = data
        self.assertIsNone(base)
        np.testing.assert_equal(self.gt_data, gt)
        np.testing.assert_equal([self.fn0_data], fn)

        data, payload = self.follow(
            "ax=axial,process=normal",
            None,
            self.gt_tmp_path,
            [self.fn0_tmp_path],
        )
        base, gt, fn = data
        self.assertIsNone(base)
        np.testing.assert_equal(gt, self.gt_data)
        np.testing.assert_equal(fn, [self.fn0_data])
