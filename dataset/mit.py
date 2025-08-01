import torch.utils.data as Data
import os
import open3d as o3d
import glob
import numpy as np
from utils.SE3 import rotation_matrix, integrate_trans
from utils.common import make_open3d_point_cloud
from utils.tools import sphericity_based_voxel_analysis

curr_path = os.path.dirname(os.path.realpath(__file__))
split_path = curr_path + "/../config/splits"


class MITDataset(Data.Dataset):
    DATA_FILES = {"train": "train_mit.txt", "val": "val_mit.txt", "test": "test_mit.txt"}

    def __init__(self, split, config=None):
        self.config = config
        self.pc_path = config.data.root
        self.split = split
        self.files = {"train": [], "val": [], "test": []}
        self.poses = []
        self.length = 0
        self.pdist = config.test.pdist
        self.mit_cache = {}
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        subset_names = open(os.path.join(split_path, self.DATA_FILES[split])).read().split()
        for dirname in subset_names:
            drive_id = str(dirname)
            fnames = glob.glob(str(self.pc_path / f"{drive_id}" / "scans" / "*.pcd"))
            assert len(fnames) > 0, f"Make sure that the path {self.pc_path} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))

            # set valid pair threshold to 5
            valid_pairs = pdist > self.pdist
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(valid_pairs[curr_time][curr_time : curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        self.length = len(self.files[split])

    def __getitem__(self, index):
        # load meta data
        drive = self.files[self.split][index][0]
        t0, t1 = self.files[self.split][index][1], self.files[self.split][index][2]

        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        o3d_cloud0 = o3d.io.read_point_cloud(fname0)
        o3d_cloud1 = o3d.io.read_point_cloud(fname1)
        xyz0 = np.asarray(o3d_cloud0.points, dtype=np.float32)
        xyz1 = np.asarray(o3d_cloud1.points, dtype=np.float32)

        # Note (Minkyun Seo):
        # Above code is commented out because it does not work well for the kimera-multi dataset.
        trans = np.linalg.inv(positions[1]) @ positions[0]
        if self.split != "test":
            xyz0 += (np.random.rand(xyz0.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
            xyz1 += (np.random.rand(xyz1.shape[0], 3) - 0.5) * self.config.train.augmentation_noise

        # process point clouds
        src_pcd = make_open3d_point_cloud(xyz0, [1, 0.706, 0])
        tgt_pcd = make_open3d_point_cloud(xyz1, [0, 0.651, 0.929])

        self.config.data.downsample, sphericity, _ = sphericity_based_voxel_analysis(
            src_pcd, tgt_pcd
        )
        is_aligned_to_global_z = self.config.patch.is_aligned_to_global_z

        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            src_pcd, voxel_size=self.config.data.downsample
        )
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            tgt_pcd, voxel_size=self.config.data.downsample
        )

        if self.split != "test":
            # SO(2) augmentation
            R = rotation_matrix(1, 1)
            t = np.zeros([3, 1])
            aug_trans = integrate_trans(R, t)
            tgt_pcd.transform(aug_trans)
            relt_pose = aug_trans @ trans
        else:
            relt_pose = trans

        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)

        # second sample
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt)

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt)

        # if we get too many points, we do random downsampling
        if src_kpt.shape[0] > self.config.data.max_numPts:
            idx = np.random.choice(
                range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False
            )
            src_kpt = src_kpt[idx]

        if tgt_kpt.shape[0] > self.config.data.max_numPts:
            idx = np.random.choice(
                range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False
            )
            tgt_kpt = tgt_kpt[idx]

        return {
            "src_fds_pts": src_pts,  # first downsampling
            "tgt_fds_pts": tgt_pts,
            "relt_pose": relt_pose,
            "src_sds_pts": src_kpt,  # second downsampling
            "tgt_sds_pts": tgt_kpt,
            "voxel_size": ds_size,
            "src_id": "%s_%d" % (drive, t0),
            "tgt_id": "%s_%d" % (drive, t1),
            "dataset_name": self.config.data.dataset,
            "sphericity": sphericity,
            "is_aligned_to_global_z": is_aligned_to_global_z,
        }

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def get_video_odometry(self, drive, indices=None, ext=".txt", return_all=False):
        data_path = self.pc_path / f"{drive}" / "poses_kitti.txt"
        if data_path not in self.mit_cache:
            self.mit_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return self.mit_cache[data_path]
        else:
            return self.mit_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        fname = self.pc_path / f"{drive}" / "scans" / f"{t:06d}.pcd"
        return str(fname)

    def __len__(self):
        return self.length
