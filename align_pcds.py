import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

from config import make_cfg
from models.BUFFERX import BufferX

# utility to mimic collate_fn_descriptor for one sample
def make_data_source(src_pcd_path, tgt_pcd_path, voxel_size, device):
    def load_pcd(path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts

    def voxel_downsample(pts, voxel_size):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        down = pcd.voxel_down_sample(voxel_size)
        return np.asarray(down.points, dtype=np.float32)

    src_pts = load_pcd(src_pcd_path)
    tgt_pts = load_pcd(tgt_pcd_path)

    # approximate first-level / second-level downsampling
    src_fds_pts = voxel_downsample(src_pts, voxel_size)
    tgt_fds_pts = voxel_downsample(tgt_pts, voxel_size)
    src_sds_pts = voxel_downsample(src_fds_pts, voxel_size * 2)
    tgt_sds_pts = voxel_downsample(tgt_fds_pts, voxel_size * 2)

    data_source = {
        "src_fds_pcd": torch.tensor(src_fds_pts, dtype=torch.float32, device=device),
        "tgt_fds_pcd": torch.tensor(tgt_fds_pts, dtype=torch.float32, device=device),
        "src_sds_pcd": torch.tensor(src_sds_pts[:, :3], dtype=torch.float32, device=device),
        "tgt_sds_pcd": torch.tensor(tgt_sds_pts[:, :3], dtype=torch.float32, device=device),
        "relt_pose": torch.eye(4, dtype=torch.float32, device=device),  # dummy since no GT
        "src_id": "src",
        "tgt_id": "tgt",
        "voxel_sizes": torch.tensor([voxel_size], device=device),
        "dataset_names": ["CustomPCD"],  # string list like the collate would produce
        "sphericity": torch.tensor([1.0], dtype=torch.float32, device=device),
        "is_aligned_to_global_z": True,
    }
    return data_source


def apply_transform_to_cloud(pcd: o3d.geometry.PointCloud, transform: np.ndarray) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    # homogeneous
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts, ones], axis=1)  # (N,4)
    transformed = (transform @ hom.T).T[:, :3]
    new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed))
    return new_pcd

def main():
    parser = argparse.ArgumentParser(description="Run BufferX on two .pcd files without ground truth evaluation")
    parser.add_argument("--src", required=True, help="Source .pcd path")
    parser.add_argument("--tgt", required=True, help="Target .pcd path")
    parser.add_argument("--experiment_id", required=True, help="Experiment ID (used in snapshot/<exp>/<stage>/best.pth)")
    parser.add_argument("--root_dir", type=str, default="../datasets", help="Root dir passed to make_cfg (used to instantiate cfg)")
    parser.add_argument("--cfg_dataset", type=str, default="3DMatch", help="Base dataset name for config (affects make_cfg)")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling (tune for your scale)")
    parser.add_argument("--output_dir", type=str, default="out_align", help="Where to write aligned pointclouds")
    parser.add_argument("--cuda", action="store_true", help="Force use of CUDA if available")
    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config (expects your existing config system to work)
    cfg = make_cfg(args.cfg_dataset, args.root_dir)
    cfg.stage = "test"

    # Instantiate model
    model = BufferX(cfg)

    # Load checkpoint(s) for all stages
    if not hasattr(cfg.train, "all_stage"):
        raise RuntimeError("cfg.train.all_stage missing; ensure your config defines training stages")
    for stage in cfg.train.all_stage:
        model_path = f"snapshot/{args.experiment_id}/{stage}/best.pth"
        if not os.path.isfile(model_path):
            print(f"[ERROR] checkpoint not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        state_dict = torch.load(model_path, map_location=device)
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {stage} model from {model_path}")

    model = nn.DataParallel(model, device_ids=[0]) if device.type == "cuda" else model
    model = model.to(device)
    model.eval()

    # Build a single data_source
    sample = make_data_source(args.src, args.tgt, args.voxel_size, device)

    # The original script passes a dict directly to model; replicate that
    with torch.no_grad():
        output = model(sample)
        # model returns (trans_est, times) per original script
        if isinstance(output, tuple) and len(output) >= 1:
            trans_est = output[0]
        else:
            trans_est = output

        if isinstance(trans_est, torch.Tensor):
            trans_est = trans_est.detach().cpu().numpy()

    # Expected 4x4; if batch dimension present, squeeze
    if trans_est.ndim == 3 and trans_est.shape[0] == 1:
        trans_est = trans_est[0]
    assert trans_est.shape == (4, 4), f"Unexpected transform shape: {trans_est.shape}"

    print("\n=== Estimated transformation matrix (source → target) ===")
    np.set_printoptions(precision=6, suppress=True)
    print(trans_est)

    # Apply transform to source cloud and save
    src_pcd = o3d.io.read_point_cloud(args.src)
    tgt_pcd = o3d.io.read_point_cloud(args.tgt)
    aligned_src = apply_transform_to_cloud(src_pcd, trans_est)

    aligned_path = os.path.join(args.output_dir, "source_aligned.pcd")
    o3d.io.write_point_cloud(aligned_path, aligned_src)
    print(f"Aligned source saved to: {aligned_path}")

    # Also save a merged cloud (aligned source + target) for quick inspection
    merged = aligned_src + tgt_pcd
    merged_path = os.path.join(args.output_dir, "merged.pcd")
    o3d.io.write_point_cloud(merged_path, merged)
    print(f"Merged point cloud saved to: {merged_path}")

    # Optional: dump transform to text
    np.savetxt(os.path.join(args.output_dir, "estimated_transform.txt"), trans_est, fmt="%.6f")
    print(f"Transform matrix written to estimated_transform.txt in {args.output_dir}")

if __name__ == "__main__":
    main()
