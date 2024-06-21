import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision
from PIL import ImageEnhance


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = (imgarr - mean[0]) / std[0]
    return proc_img


def random_noise(pre_img, post_im):
    np.random.seed()
    noise_map = np.random.random(pre_img.size)
    return pre_img


def random_scaling(pre_img, post_img, loc_label, dam_label, size_range, scale_range):
    h, w, = dam_label.shape

    min_ratio, max_ratio = scale_range
    assert min_ratio <= max_ratio

    ratio = random.uniform(min_ratio, max_ratio)

    new_scale = int(size_range[0] * ratio), int(size_range[1] * ratio)

    max_long_edge = max(new_scale)
    max_short_edge = min(new_scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

    return _img_rescaling(pre_img, post_img, loc_label, dam_label, scale=ratio)


def _img_rescaling(pre_img, post_img, loc_label, dam_label, scale=None):
    # scale = random.uniform(scales)
    h, w, = dam_label.shape

    new_scale = [int(scale * w), int(scale * h)]

    new_pre_img = Image.fromarray(pre_img.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_pre_img = np.asarray(new_pre_img).astype(np.float32)

    new_post_img = Image.fromarray(post_img.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_post_img = np.asarray(new_post_img).astype(np.float32)

    if dam_label is None:
        return new_pre_img, new_post_img

    new_dam_label = Image.fromarray(dam_label).resize(new_scale, resample=Image.NEAREST)
    new_dam_label = np.asarray(new_dam_label)
    new_loc_label = Image.fromarray(loc_label).resize(new_scale, resample=Image.NEAREST)
    new_loc_label = np.asarray(new_loc_label)

    return new_pre_img, new_post_img, new_loc_label, new_dam_label


def img_resize_short(image, min_size=512):
    h, w, _ = image.shape
    if min(h, w) >= min_size:
        return image

    scale = float(min_size) / min(h, w)
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    return new_image


def random_resize(image, label, size_range=None):
    _new_size = random.randint(size_range[0], size_range[1])

    h, w, = label.shape
    scale = _new_size / float(max(h, w))
    new_scale = [int(scale * w), int(scale * h)]

    new_image, new_label = _img_rescaling(image, label, scale=new_scale)

    return new_image, new_label


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label


def random_fliplr_multicd(pre_img, post_img, pre_lc_label, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
        pre_lc_label = np.fliplr(pre_lc_label)

    return pre_img, post_img, pre_lc_label, label


def random_fliplr_with_object(pre_img, post_img, object_map, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
        object_map = np.fliplr(object_map)

    return pre_img, post_img, object_map, label


def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label


def random_flipud_multicd(pre_img, post_img, pre_lc_label, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
        pre_lc_label = np.flipud(pre_lc_label)

    return pre_img, post_img, pre_lc_label, label


def random_flipud_with_object(pre_img, post_img, object_map, label):
    if random.random() > 0.5:
        object_map = np.flipud(object_map)
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, object_map, label


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_multicd(pre_img, post_img, pre_lc_label, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()
    pre_lc_label = np.rot90(pre_lc_label, k).copy()

    return pre_img, post_img, pre_lc_label, label


def random_rot_with_object(pre_img, post_img, object_map, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    object_map = np.rot90(object_map, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, object_map, label


def random_crop(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # pad_pre_image = np.zeros((H, W), dtype=np.float32)
    pad_pre_image = np.zeros((H, W, pre_img.shape[-1]), dtype=np.float32)

    pad_post_image = np.zeros((H, W, pre_img.shape[-1]), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    # pad_pre_image[:, :, 0] = mean_rgb[0]
    # pad_pre_image[:, :, 1] = mean_rgb[1]
    # pad_pre_image[:, :, 2] = mean_rgb[2]
    #
    # pad_post_image[:, :, 0] = mean_rgb[0]
    # pad_post_image[:, :, 1] = mean_rgb[1]
    # pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w)] = pre_img
    # pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end :]
    # pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(label))
    return pre_img, post_img, label


def random_crop_multicd(pre_img, post_img, pre_lc_label, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_pre_lc_label = np.zeros((H, W), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_pre_lc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = pre_lc_label
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    pre_lc_label = pad_pre_lc_label[H_start:H_end, W_start:W_end]
    label = pad_label[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(label))
    return pre_img, post_img, pre_lc_label, label


def random_crop_with_object(pre_img, post_img, object_map, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # pad_pre_image = np.zeros((H, W), dtype=np.float32)
    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_object_map = np.zeros((H, W), dtype=np.long)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    # pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w)] = pre_img
    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_object_map[H_pad:(H_pad + h), W_pad:(W_pad + w)] = object_map
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    # pre_img = pad_pre_image[H_start:H_end, W_start:W_end]
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    object_map = pad_object_map[H_start:H_end, W_start:W_end]
    label = pad_label[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(label))
    return pre_img, post_img, object_map, label


def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16), :]


def tensorboard_image(inputs=None, outputs=None, labels=None, bgr=None):
    ## images
    inputs[:, 0, :, :] = inputs[:, 0, :, :] + bgr[0]
    inputs[:, 1, :, :] = inputs[:, 1, :, :] + bgr[1]
    inputs[:, 2, :, :] = inputs[:, 2, :, :] + bgr[2]
    inputs = inputs[:, [2, 1, 0], :, :].type(torch.uint8)
    grid_inputs = torchvision.utils.make_grid(tensor=inputs, nrow=2)

    ## preds
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    preds_cmap = encode_cmap(preds)
    preds_cmap = torch.from_numpy(preds_cmap).permute([0, 3, 1, 2])
    grid_outputs = torchvision.utils.make_grid(tensor=preds_cmap, nrow=2)

    ## labels
    labels_cmap = encode_cmap(labels.cpu().numpy())
    labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_inputs, grid_outputs, grid_labels


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.fromarray(np.uint8(image))

    random_factor = np.random.randint(5, 21) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度

    random_factor = np.random.randint(5, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度

    random_factor = np.random.randint(5, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    final = np.asarray(sharp_image)
    return final


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
