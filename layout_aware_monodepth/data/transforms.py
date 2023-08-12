import random

import cv2
import numpy as np
import torch
from ip_basic import depth_map_utils
from PIL import Image
from torchvision import transforms


def kb_crop(image: Image.Image, depth_gt=None):
    height = image.height
    width = image.width
    top_margin = int(height - 352)
    left_margin = int((width - 1216) / 2)
    image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

    if depth_gt is not None:
        depth_gt = depth_gt.crop(
            (left_margin, top_margin, left_margin + 1216, top_margin + 352)
        )
        return image, depth_gt

    return image


def rotate_image(image, angle, flag=Image.BILINEAR):
    result = image.rotate(angle, resample=flag)
    return result


def resize_inputs(*args, target_shape):
    def _resize(x):
        x = cv2.resize(
            x,
            target_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        return x

    return [_resize(x) for x in args] if len(args) > 1 else _resize(args[0])


def random_crop(img, depth, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == depth.shape[0]
    assert img.shape[1] == depth.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y : y + height, x : x + width, :]
    depth = depth[y : y + height, x : x + width, :]
    return img, depth


def train_preprocess(image, depth_gt):
    # Random flipping
    do_flip = random.random()
    if do_flip > 0.5:
        image = (image[:, ::-1, :]).copy()
        depth_gt = (depth_gt[:, ::-1, :]).copy()

    # Random gamma, brightness, color augmentation
    do_augment = random.random()
    if do_augment > 0.5:
        image = augment_image(image)

    return image, depth_gt


def augment_image(image, ds_name="kitti"):
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image**gamma

    # brightness augmentation
    if ds_name == "nyu":
        brightness = random.uniform(0.75, 1.25)
    else:
        brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        image = self.to_tensor(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def interpolate_depth(depth, do_multiscale=False, *args, **kwargs):
    """See depth_map_utils.fill_in_fast"""
    if do_multiscale:
        ddm, _ = depth_map_utils.fill_in_multiscale(
            depth.astype("float32"), *args, **kwargs
        )
    else:
        ddm = depth_map_utils.fill_in_fast(depth.astype("float32"), *args, **kwargs)
    return ddm


train_transform = transforms.Compose(
    [
        ToTensor(mode="train"),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        ToTensor(mode="test"),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
