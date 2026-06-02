from errors import InternalError
from method.normal import Normal
from methods_types import DashboardMethod
import numpy as np
import scipy.stats

from utils import dice
from medpy.metric.binary import hd


class CTDashboardMethod(DashboardMethod):
    def __init__(self):
        pass

    def process(self, base, gt, fn):
        # base: 画素値ごとのヒストグラム
        # base: 画素値の平均, 分散, 標準偏差, 中央値, エントロピー
        # gt&fn: diff, hd
        # gt&fn: label: dice, hd
        # gt|fn: label: 面積
        # (gt|fn)&base: label: 平均輝度, 分散, 標準偏差, 中央値, エントロピー, ヒストグラム
        out = {}
        loaded = Normal().process(base, gt, fn)
        bl = loaded.pop("base")
        gl = loaded.pop("gt")
        fnl = [loaded[f"fn{i}"] for i in range(len(fn))]
        if bl is not None:
            hist = {}
            hist[self.TYPE] = self.CONTINUOUS_GRAPH
            hist[self.RELATED] = "base"
            hist[self.VALUE] = {}
            counts, bins = np.histogram(bl, bins=1000)
            bins = bins[:-1]
            for count, b in zip(counts, bins):
                hist[self.VALUE][float(b)] = int(count)
            out["hist"] = hist
        if bl is not None:

            def _(v):
                return {
                    self.TYPE: self.NUMBER,
                    self.RELATED: ["base"],
                    self.VALUE: float(v),
                }

            mean = bl.mean()
            var = bl.var()
            std = bl.std()
            mid = np.median(bl)
            ent = scipy.stats.entropy(bl, axis=None)
            out["mean"] = _(mean)
            out["var"] = _(var)
            out["std"] = _(std)
            out["mid"] = _(mid)
            out["entropy"] = _(ent)

        if gl is not None:
            for i in range(len(fnl)):
                f = fnl[i]
                related = f"gt&fn{i}"

                labels = np.unique(list(np.unique(gl)) + list(np.unique(f)))
                dlabels = {}
                hlabels = {}
                for label in labels:
                    gf = gl == label
                    ff = f == label
                    dlabels[int(label)] = dice(gf, ff)
                    if gf.any() and ff.any():
                        hlabels[int(label)] = hd(gf, ff)
                    else:
                        hlabels[int(label)] = np.nan
                dlabels["mean"] = np.nanmean(list(dlabels.values())).item()
                dlabels["mean_nobg"] = np.nanmean(
                    [v for k, v in dlabels.items() if k != 0]
                ).item()
                hlabels["mean"] = np.nanmean(list(hlabels.values())).item()
                hlabels["mean_nobg"] = np.nanmean(
                    [v for k, v in hlabels.items() if k != 0],
                ).item()
                for k, v in hlabels.items():
                    if v is np.nan:
                        hlabels[k] = None
                out[f"dice-fn{i}"] = {
                    self.TYPE: self.SCATTER_GRAPH,
                    self.RELATED: related,
                    self.VALUE: dlabels,
                }
                out[f"hausdorff-fn{i}"] = {
                    self.TYPE: self.SCATTER_GRAPH,
                    self.RELATED: related,
                    self.VALUE: hlabels,
                }
        return out
