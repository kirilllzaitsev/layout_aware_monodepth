# Layout-aware monoocular depth estimation

## Installation

### Requirements

- Python >=3.8

### Setup

```bash
pip install git+https://github.com/kirilllzaitsev/how-do-vits-work.git@transformer
pip install -r requirements.txt
```

## Datasets

Data splits are provided in `data/data_splits` folder.

### NYU Depth V2

Download the dataset from [here](https://github.com/zzangjinsun/NLSPN_ECCV20/tree/master). This is a preprocessed version of the original NYU Depth V2 dataset and does not have prerequisite data preparation steps.

### KITTI Depth

Download the dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction). Follow instructions in [this repo](https://github.com/zzangjinsun/NLSPN_ECCV20/tree/master) to prepare the data. At this point, to use Eigen split one needs to download raw dataset as well from the [official website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

## Running the code

### Training

See the available CLI arguments in the train_prototyping.py file.

An example of a training command is provided below:

```bash
python train_prototyping.py --exp_tags vp_filtering --num_epochs 20 --line_op concat_binary --do_overfit --use_eigen_test --line_filter vanishing_point --use_attn --ds nyu --max_depth_eval 10.0 --exp_disabled
```

Metrics, plots, and checkpoints are saved to the [Comet ML project](https://www.comet.com/kirilllzaitsev/layout-aware-monodepth/view/new/experiments).