# BUFFER-X 

## Requirements
This work has been tested on Ubuntu 22.04, Cuda 12.1, Nvidia RTX 4070 and Python 3.10.

## Install

Clone and create a virtual environment:

```
git clone https://github.com/ArghyaChatterjee/BUFFER-X 
cd BUFFER-X
python3 -m venv buffer_x_venv
source buffer_x_venv/bin/activate
pip3 install --upgrade pip setuptools wheel scikit-build-core ninja cmake build
```

Setup your **own virtual environment** (e.g., `conda create -n bufferx python=3.x` or setting your Nvidia Docker env.) and then install the required libraries. We present some shellscripts as follows.
```
chmod +x scripts/install_py3_10_cuda12_1.sh
./scripts/install_py3_10_cuda12_1.sh
```

______________________________________________________________________

## Quick Start

### Training and Test

#### Test on Our Generalization Benchmark

You can easily run our **generalization benchmark** with BUFFER-X. First, download the model using the following script:

```
./scripts/download_pretrained_models.sh
```

<details>
  <summary><strong>Detailed explanation about file directory</a></strong></summary>

The structure should be as follows:

- `BUFFER-X`
  - `snapshot` # \<- this directory is generated by the command above
    - `threedmatch`
      - `Desc`
      - `Pose`
    - `kitti`
      - `Desc`
      - `Pose`
  - `config`
  - `dataset`
  - ...

</details>
<br>

#### Test on 2 PointClouds
If you have 2 point clouds in `.pcd` format, run the following script:
```
python align_pcds.py \
  --src /home/arghya/BUFFER-X/input_data/mug_mesh_orig_77.pcd \
  --tgt /home/arghya/BUFFER-X/input_data/mug_mesh_roi_77.pcd \
  --experiment_id threedmatch \
  --root_dir . \
  --cfg_dataset 3DMatch \
  --voxel_size 0.05 \
  --output_dir aligned_output \
  --cuda
```
The output will be saved inside `aligned_output` directory. The outputs are:
```
estimated_transform.txt
merged.pcd
source_aligned.pcd
```