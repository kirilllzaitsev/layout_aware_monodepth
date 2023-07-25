import numpy as np
from .vis_utils import depth_colorize, feature_colorize, validcrop, cmap
import torch
import cv2
from PIL import Image


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


def save_image_torch(rgb, filename):
    # torch2numpy
    rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    # print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype("uint8")
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


def save_depth_as_uint16png(img, filename):
    # from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype("uint16")
    cv2.imwrite(filename, img)


def save_depth_as_uint16png_upload(img, filename):
    # from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype("uint16")
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, "raw", "I;16")
    imgsave.save(filename)


def save_depth_as_uint8colored(sparse, gt, pred, filename):
    # from tensor
    # img = validcrop(img)
    img_list = []

    sparse = np.squeeze(sparse[0, ...]).data.cpu().numpy()
    sparse = depth_colorize(sparse)
    pred = np.squeeze(pred[0, ...]).data.cpu().numpy()
    pred = depth_colorize(pred)

    gt = np.squeeze(gt[0, ...]).data.cpu().numpy()
    gt = depth_colorize(gt)

    img_list.append(sparse)
    img_list.append(gt)
    img_list.append(pred)

    img_merge1 = np.hstack(img_list)
    img_merge1 = img_merge1.astype("uint8")

    img = cv2.cvtColor(img_merge1, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if normalized == False:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if colored == True:
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
