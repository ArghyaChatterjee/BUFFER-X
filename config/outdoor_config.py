from easydict import EasyDict as edict


class OutdoorBaseConfig:
    def __init__(self):
        self._C = edict()

        # Data
        self._C.data = edict()
        self._C.data.dataset = ""
        self._C.data.root = ""
        self._C.data.downsample = 0.05
        self._C.data.voxel_size_0 = 0.30
        self._C.data.voxel_size_1 = self._C.data.voxel_size_0
        self._C.data.max_numPts = 30000
        self._C.data.manual_seed = 123

        # Training
        self._C.train = edict()
        self._C.train.epoch = 50
        self._C.train.max_iter = 50000
        self._C.train.batch_size = 1
        self._C.train.num_workers = 0
        self._C.train.pos_num = 512
        self._C.train.augmentation_noise = 0.01
        self._C.train.pretrain_model = ""
        self._C.train.all_stage = ["Desc", "Pose"]

        # Test
        self._C.test = edict()
        self._C.test.experiment_id = "threedmatch"
        self._C.test.pose_refine = False

        # Evaluation thresholds
        self._C.test.rte_thresh = 2.0  # RTE threshold for outdoor datasets
        self._C.test.rre_thresh = 5.0  # RRE threshold for outdoor datasets

        # Optimizer
        self._C.optim = edict()
        self._C.optim.lr = {"Desc": 0.001, "Pose": 0.001}
        self._C.optim.lr_decay = 0.50
        self._C.optim.weight_decay = 1e-6
        self._C.optim.scheduler_interval = {"Desc": 10, "Pose": 5}

        # Multi-scale patch embedder
        self._C.patch = edict()
        self._C.patch.des_r = 3.0  # For training
        self._C.patch.num_points_per_patch = 512
        self._C.patch.num_fps = 1500
        self._C.patch.rad_n = 3
        self._C.patch.azi_n = 20
        self._C.patch.ele_n = 7
        self._C.patch.delta = 0.8
        self._C.patch.voxel_sample = 10
        self._C.patch.num_scales = 3
        self._C.patch.is_aligned_to_global_z = True

        # Threshold should be setted in decreasing order
        self._C.patch.search_radius_thresholds = [5, 2, 0.5]
        self._C.patch.num_points_radius_estimate = 2000

        # Hierarchical inlier search & RANSAC
        self._C.match = edict()
        self._C.match.dist_th = 0.30
        self._C.match.inlier_th = 2.0
        self._C.match.similar_th = 0.9
        self._C.match.confidence = 1.0
        self._C.match.iter_n = 50000

    def get_cfg(self):
        return self._C
