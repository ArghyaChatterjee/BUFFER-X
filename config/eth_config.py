from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class ETHConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "ETH"
        self._C.data.root = root_dir / "ETH"
        self._C.test.experiment_id = "threedmatch"

        self._C.match.dist_th = 0.20
        self._C.match.inlier_th = 1.5
        self._C.match.similar_th = 0.9
        self._C.match.confidence = 1.0
        self._C.match.iter_n = 50000

        self._C.test.rte_thresh = 0.3
        self._C.test.rre_thresh = 2.0


def make_cfg(root_dir):
    return ETHConfig(root_dir).get_cfg()
