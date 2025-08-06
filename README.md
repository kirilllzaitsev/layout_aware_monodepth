# Layout-aware monocular depth estimation

This project explores improving monocular depth estimation from a single RGB image by regularizing deep learning models based on structural cues from 2D lines and vanishing points. The model leverages geometric priors to enhance depth quality and generalization. We considered supervised and self-supervised training setups with a UNet architecture for scenes in public outdoor (KITTI) and indoor (NYUv2) datasets.

## Installation

### Setup

```bash
conda create -n monodepth python=3.8
conda activate monodepth
pip install git+https://github.com/kirilllzaitsev/how-do-vits-work.git@transformer
pip install git+https://github.com/kujason/ip_basic.git@master
pip install git+https://github.com/kirilllzaitsev/calibrated-backprojection-network.git@master
pip install -r requirements.txt
pip install -e .
```

(Optional) Fetch artifacts:

```bash
git lfs pull
```

## Datasets

Data splits are provided in `data/data_splits` folder.

### NYU Depth V2

Download the dataset from [here](https://github.com/zzangjinsun/NLSPN_ECCV20/tree/master). This is a preprocessed version of the original NYU Depth V2 dataset and does not have prerequisite data preparation steps.

### KITTI Depth

Download the dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction). Follow instructions in [this repo](https://github.com/zzangjinsun/NLSPN_ECCV20/tree/master) to prepare the data. At this point, to use Eigen split one needs to download raw dataset as well from the [official website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

## Running the code

### Training

See the available CLI arguments in the `train_prototyping.py` file.

An example of a training command for NYUv2 dataset (`--ds nyu`) is provided below:

```bash
python train_prototyping.py --exp_tags vp_filtering --num_epochs 20 --line_op concat_binary --do_overfit --use_eigen --line_filter vanishing_point --use_attn --ds nyu --max_depth_eval 10.0 --exp_disabled
```

Metrics, plots, and checkpoints are logged to the [Comet ML project](https://www.comet.com/kirilllzaitsev/layout-aware-monodepth/view/new/experiments).