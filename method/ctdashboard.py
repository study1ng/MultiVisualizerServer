from method.normal import Normal
from methods_types import DashboardMethod
import numpy as np
import scipy.stats


class CTDashboardMethod(DashboardMethod):
    def __init__(self):
        pass

    def process(self, base, gt, fn):
        # base: 画素値ごとのヒストグラム
        # base: 画素値の平均, 分散, 標準偏差, 中央値, エントロピー
        # gt&fn: diff, hd
        # gt&fn: label: diff, hd
        # gt|fn: label: 面積
        # (gt|fn)&base: label: 平均輝度, 分散, 標準偏差, 中央値, エントロピー, ヒストグラム
        out = {}
        loaded = Normal().process(base, gt, fn)
        base = loaded.pop("base")
        gt = loaded.pop("gt")
        fn = [loaded[f"fn{i}"] for i in range(len(fn))]
        if base is not None:
            hist = {}
            hist[self.TYPE] = self.CONTINUOUS_GRAPH
            hist[self.RELATED] = ["base"]
            hist[self.VALUE] = {}
            counts, bins = np.histogram(base, bins=1000)
            bins = bins[:-1]
            for count, b in zip(counts, bins):
                hist[self.VALUE][float(b)] = int(count)
            out["hist"] = hist
        if base is not None:

            def _(v):
                return {
                    self.TYPE: self.NUMBER,
                    self.RELATED: ["base"],
                    self.VALUE: float(v),
                }

            mean = base.mean()
            var = base.var()
            std = base.std()
            mid = np.median(base)
            ent = scipy.stats.entropy(base, axis=None)
            out["mean"] = _(mean)
            out["var"] = _(var)
            out["std"] = _(std)
            out["mid"] = _(mid)
            out["entropy"] = _(ent)

        return out
