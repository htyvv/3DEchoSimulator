import numpy as np
import cv2
import imutils

import mesh_lib.image_operations as image_operations


def pad_to_size(img, size, extra_size, axis=0):
    pad = size - img.shape[axis]

    lr_pad = pad // 2

    if axis == 0:
        if pad % 2 == 1:
            img = np.pad(
                img,
                ((lr_pad + 1 + extra_size[0], lr_pad + extra_size[1]), (0 + extra_size[2], 0 + extra_size[3])),
                "constant",
                constant_values=0,
            )
        else:

            img = np.pad(
                img,
                ((lr_pad + extra_size[0], lr_pad + extra_size[1]), (0 + extra_size[2], 0 + extra_size[3])),
                "constant",
                constant_values=0,
            )
    else:
        if pad % 2 == 1:
            img = np.pad(
                img,
                ((0 + extra_size[0], 0 + extra_size[1]), (lr_pad + extra_size[2] + 1, lr_pad + extra_size[3])),
                "constant",
                constant_values=0,
            )
        else:
            img = np.pad(
                img,
                ((0 + extra_size[0], 0 + extra_size[1]), (lr_pad + extra_size[2], lr_pad + extra_size[3])),
                "constant",
                constant_values=0,
            )
    return img


def get_circle_mask(xx, yy, origin, radius):
    """ Gets a circular mask at the specified origin and radius.
     xx and yy define the coordinate grid for the mask so origin should be in reference to these two.
     """
    mask = np.zeros(xx.shape)
    mask[np.sqrt((yy - origin[0]) ** 2 + (xx - origin[1]) ** 2) < radius] = 1
    return mask


def get_angle_mask(xx, yy, origin, width, tilt):
    """ Gets a triangular mask of the specified origin, width, and tilt.
    xx and yy provide the grid for the mask and the parameters should be listed in reference to this.
    """
    mask = np.zeros(xx.shape)
    half_width = width / 2
    lower_bound = -np.pi / 2 - half_width + tilt
    upper_bound = -np.pi / 2 + half_width + tilt
    mask[
        (np.arctan2(origin[0] - yy, xx - origin[1]) < upper_bound)
        & (np.arctan2(origin[0] - yy, xx - origin[1]) > lower_bound)
    ] = 1
    return mask


def get_full_mask(inp_size, origin, radius, width, tilt, ax=None):
    """ calls both circle mask and angle mask to get a mask of an ultrasound cone.
    if ax is not None than the mask will be plotted.
    returns a mask with 0 being the region outside the cone and 1 the region inside it
    """
    assert radius < inp_size, "radius should be smaller than image"
    max_side_pt = max([radius * np.sin(width / 2 + tilt), radius * np.sin(-width / 2 - tilt)])
    diff = max_side_pt - inp_size / 2
    if diff > 0:
        inp_size = int(np.ceil(max_side_pt * 2))
        origin[1] += int(diff)
    xx, yy = np.meshgrid(range(inp_size), range(inp_size))
    circle_mask = get_circle_mask(xx, yy, origin, radius)
    angle_mask = get_angle_mask(xx, yy, origin, width, tilt)
    full_mask = circle_mask + angle_mask
    if ax is not None:
        ax.imshow(full_mask)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
    full_mask /= 2
    full_mask = full_mask.astype(int)
    return full_mask


def generate_US_cones(num_cones, inp_size=1024):
    """ randomly samples radius, width, tilt, origin, and rotations to generate a ultrasound mask. """
    radii = np.random.uniform(0.93 * inp_size, 1.0 * inp_size, size=num_cones)  # 976
    widths = np.random.normal(6 * np.pi / 12, np.pi / 30, size=num_cones)  # TODO make this adjustable with view
    # tilts = np.random.normal(0, np.pi / 32, size=num_cones)
    tilts = np.zeros(num_cones)
    origin_ys = np.ones(shape=(num_cones,))
    origin_xs = inp_size / 2 * np.ones(shape=(num_cones,))
    # origin_ys = np.random.normal(16, 4, size=num_cones)
    # origin_xs = np.random.normal(512, 16, size=num_cones)
    # rotations = np.random.normal(-90, 1, size=num_cones)
    for radius, width, tilt, origin_y, origin_x in zip(radii, widths, tilts, origin_ys, origin_xs):
        params = dict(radius=radius, width=width, tilt=tilt, origin_x=origin_x, origin_y=origin_y)
        mask = get_full_mask(inp_size, [origin_y, origin_x], radius, width, tilt)
        mask = image_operations.crop_to_mask(mask, mask)  # crop to mask and reshape
        mask = image_operations.resize(mask, (inp_size, inp_size))
        yield np.array(mask), params


def image_padding(img, extra_size):
    img = np.pad(img, ((extra_size[0], extra_size[1]), (extra_size[2], extra_size[3])), "constant", constant_values=0,)
    return img


def padding_bounds(original_bounds, padding):
    x0, x1, y0, y1 = original_bounds
    pad_y0, pad_y1, pad_x0, pad_x1 = padding

    # Calculate new bounds
    new_x0 = x0 - pad_x0
    new_x1 = x1 + pad_x1
    new_y0 = y0 - pad_y0
    new_y1 = y1 + pad_y1

    return [new_x0, new_x1, new_y0, new_y1]


def random_pad_size(mask, view):
    """
    Randomly generate padding size for the mask. Output mask with padding is square.
    :param mask: numpy array
    :param view: view name
    :return: padding size
    """
    image_hw_max = np.max(mask.shape)

    # TODO: make this adjustable with view
    # if view == "PSAX apex":
    #     extra_pad_size = np.array(image_hw_max / 2 + np.random.random(4) * image_hw_max / 4, dtype=int)
    if view == "PSAX apex":
        extra_pad_size = np.array(image_hw_max / 4 + np.random.random(4) * image_hw_max / 3, dtype=int)
    elif view == "A4CH":
        extra_pad_size = np.array(image_hw_max / 8 + np.random.random(4) * image_hw_max / 4, dtype=int)
    elif view != "PLAX":
        extra_pad_size = np.array(image_hw_max / 8 + np.random.random(4) * image_hw_max / 6, dtype=int)
    else:
        extra_pad_size = np.array(np.random.random(4) * image_hw_max / 8, dtype=int)

    if view == "PSAX apex":
        extra_pad_size[1] += image_hw_max / 3 + np.random.random(1) * image_hw_max / 3
    if view == "A4CH":
        extra_pad_size[1] += image_hw_max / 10 + np.random.random(1) * image_hw_max / 6
    mask_shape_w_padding = np.array(mask.shape) + np.array(
        [extra_pad_size[0] + extra_pad_size[1], extra_pad_size[2] + extra_pad_size[3]]
    )
    axis = np.argmin(mask_shape_w_padding)
    pad_for_square = np.max(mask_shape_w_padding) - mask_shape_w_padding[axis]
    lr_pad = pad_for_square // 2

    if axis == 0:
        if pad_for_square % 2 == 1:
            pad_x0 = lr_pad + 1
            pad_x1 = lr_pad
            pad_y0 = 0
            pad_y1 = 0
        else:
            pad_x0 = lr_pad
            pad_x1 = lr_pad
            pad_y0 = 0
            pad_y1 = 0
    else:
        if pad_for_square % 2 == 1:
            pad_x0 = 0
            pad_x1 = 0
            pad_y0 = lr_pad + 1
            pad_y1 = lr_pad
        else:
            pad_x0 = 0
            pad_x1 = 0
            pad_y0 = lr_pad
            pad_y1 = lr_pad
    pad_size = np.array([pad_x0, pad_x1, pad_y0, pad_y1])
    pad_size += extra_pad_size
    return pad_size


def mask_augmentation(mask, view):
    pad_size = random_pad_size(mask, view)
    # TODO: add random rotation, fine padding size
    mask = image_padding(mask, pad_size)
    return mask, pad_size


def apply_us_cone(mask):
    cone = generate_US_cones(1, mask.shape[np.argmax(mask.shape)])
    cone = next(cone)
    mask = mask * cone[0]
    mask[mask > 0] += 1
    mask[(mask == 0) & (cone[0] == 1.0)] = 1
    return mask


# def us_style_mask(mask, view, projected_slice_bounds):
#     mask, pad_size = mask_augmentation(mask, view)
#     padded_projected_slice_bounds = padding_bounds(projected_slice_bounds[:4], pad_size)
#     if view == "A4CH":
#         mask = cv2.flip(mask, 0)
#     elif view == "A3CH":
#         mask = imutils.rotate(mask, -45)
#     mask = apply_us_cone(mask)
#     if view == "A4CH":
#         mask = cv2.flip(mask, 0)
#     elif view == "A3CH":
#         mask = imutils.rotate(mask, 45)
#     return mask, pad_size, padded_projected_slice_bounds
