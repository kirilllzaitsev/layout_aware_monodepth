import glob

import imageio.v2 as imageio
from torch.utils.data import Dataset
from torchvision import transforms


class WaymoDataset(Dataset):
    def __init__(self, save_dir):
        self.depth_paths = []
        self.rgb_paths = []

        for context in glob.glob(f"{save_dir}/*"):
            for depth_path, rgb_path in zip(
                sorted(glob.glob(f"{context}/depth_images/*")),
                sorted(glob.glob(f"{context}/rgb_images/*")),
            ):
                self.depth_paths.append(depth_path)
                self.rgb_paths.append(rgb_path)

        assert len(self.depth_paths) == len(self.rgb_paths)

        self.depth_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((375, 1242), antialias=False),
            ]
        )

        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((192, 640), antialias=True),
            ]
        )

    def __len__(self):
        return len(self.depth_paths)

    def __getitem__(self, idx):
        depth_path = self.depth_paths[idx]
        rgb_path = self.rgb_paths[idx]

        depth = (imageio.imread(depth_path) / 255).astype("float32")
        rgb = (imageio.imread(rgb_path) / 255).astype("float32")

        depth = self.depth_transform(depth)
        rgb = self.rgb_transform(rgb)

        return {"depth": depth, "rgb": rgb}


if __name__ == "__main__":
    # Create a dataset instance
    dataset = WaymoDataset(save_dir="/waymo/v1/validation/extracted")

    # Get a sample from the dataset
    depth_sample, rgb_sample = dataset[0]
    print(depth_sample.size(), rgb_sample.size())
