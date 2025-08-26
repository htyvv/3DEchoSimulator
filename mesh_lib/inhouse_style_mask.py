import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

import mesh_lib.image_operations as image_operations
from mesh_lib.utils import Tags
from mesh_lib.important_class_dict import important_class_dict


TissueTagNames = [t for t in Tags if any([name in t for name in ["myocardium", "border", "aorta"]])]
ValveTagNames = [t for t in Tags if t not in TissueTagNames and "valve" in t]
OtherTagNames = [
    t for t in Tags if t not in TissueTagNames and any([name in t for name in ["appendage", "vein", "cava"]])
]
BloodTagNames = [t for t in Tags if any([name in t for name in ["pool"]])]


def _merge_tissue_and_change_values(img, include_valves=True):
    """
    First this function merges all tissue tags into a single value. Only elements with
    TissueTags (as definited in mesh_utils.__init__.py) and the pericardium are included in the resulting final
    image. if include_valves is set, ValveTagNames will also be included.

    Second, in original image the values are all right next to each other. This function helps differentiate the
    tissue from the background clearly. It also differentiates between the tissue and the pericardium.

    It makes sense to use a single function here since the same masks are created.

    Will also return a mask for the area inside the heart so that area can be treated differently from the
    area outside later if desired

    """
    tissue_mask = np.zeros_like(img).astype(bool)
    for tissue_tag in TissueTagNames:
        tissue_mask += img == Tags[tissue_tag]
    if include_valves:
        for valve_tag in ValveTagNames:
            tissue_mask += img == Tags[valve_tag]
    pericardium_mask = img == Tags["pericardium"]
    outside_mask = img == 0  # all other values will have a tag
    inside_mask = outside_mask + pericardium_mask + tissue_mask == 0
    # make a new image to automatically remove all other tissue types
    INSIDE_VAL = 50
    TISSUE_VAL = 175
    PERICARDIUM_VAL = 250
    img = np.zeros_like(img) + INSIDE_VAL
    img[tissue_mask] = TISSUE_VAL
    img[pericardium_mask] = PERICARDIUM_VAL
    return img, inside_mask


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


def post_adjust_pseudo(self, pseudo, cone):
    # have to remove the padded pix to get back to original
    pseudo = image_operations.resize(pseudo, (cone.shape[0], cone.shape[0]), Image.BILINEAR)
    pseudo *= cone.astype(np.uint8)  # add the cone
    # pseudo, transform_params = post_cone_pseudo(pseudo, self.transform_params)
    # if DEBUG_WITH_IMAGES:
    #     self.show_img(pseudo, title="after post adjustment")
    return pseudo


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


def fill_in(mask):
    # _, mask_contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = mask_contour[np.argmax([len(c) for c in mask_contour])]

    new_mask = np.zeros_like(mask)
    new_mask = cv2.drawContours(new_mask.astype(np.uint8), [mask_contour], -1, 1, -1)
    return new_mask


def process_a2c(image):

    lv_myo_mask = fill_in(image == Tags["lv_myocardium"])
    la_myo_mask = fill_in(image == Tags["la_myocardium"])
    mv_mask = image == Tags["mitral_valve"]
    la_myo_mask += mv_mask
    lv_myo_mask += mv_mask
    # _, la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[la_myo_contour[0]], color=(255, 255, 255))
    cv2.fillPoly(new_mask, pts=[la_myo_contour[1]], color=(255, 255, 255))
    # _, la_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    la_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[1]], color=(255, 255, 255))
    # _, lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[0]], color=(255, 255, 255))
    # _, lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)

    label_list = important_class_dict["A2CH"]
    label_dict = {label: i + 1 for i, label in enumerate(label_list)}

    new_mask = cv2.drawContours(new_mask, lv_contour, -1, label_dict["LV cavity"], -1)
    new_mask = cv2.drawContours(new_mask, la_contour, -1, label_dict["LA cavity"], -1)
    new_mask = cv2.drawContours(new_mask, lv_wall_contour, -1, label_dict["LV wall"], -1)
    # plt.imshow(new_mask)
    # plt.show()
    new_mask[mv_mask] = label_dict["LV cavity"]

    return new_mask


def process_a4c(image):
    lv_myo_mask = fill_in(image == Tags["lv_myocardium"])

    mv_mask = image == Tags["mitral_valve"]
    rv_myo_mask = image == Tags["rv_myocardium"]
    # _, rv_myo_contour_tmp, _ = cv2.findContours(rv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_myo_contour_tmp, _ = cv2.findContours(rv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_myo_contour_tmp_left = rv_myo_contour_tmp[np.argmin([len(contour) for contour in rv_myo_contour_tmp])]
    rv_myo_contour_tmp_right = rv_myo_contour_tmp[np.argmax([len(contour) for contour in rv_myo_contour_tmp])]
    rv_wall_mask_left = np.zeros_like(image)
    rv_wall_mask_right = np.zeros_like(image)

    cv2.drawContours(rv_wall_mask_left, [rv_myo_contour_tmp_left], -1, 1, -1)
    cv2.drawContours(rv_wall_mask_right, [rv_myo_contour_tmp_right], -1, 1, -1)

    # ra_myo_mask = fill_in(image == Tags["ra_myocardium"])
    ra_myo_mask = image == Tags["ra_myocardium"]
    tv_mask = image == Tags["tricuspid_valve"]
    la_myo_mask = image == Tags["la_myocardium"]

    la_myo_mask += mv_mask
    lv_myo_mask += mv_mask
    lv_myo_mask += rv_wall_mask_left.astype(np.bool_)
    ra_myo_mask += tv_mask
    rv_myo_mask += tv_mask

    ra_myo_mask = fill_in(ra_myo_mask)

    # apply kerenel to ra_myo_mask
    kernel = np.ones((3, 3), np.uint8)
    ra_myo_mask = cv2.dilate(ra_myo_mask.astype(np.uint8), kernel, iterations=1)

    # apply kerenel to ra_myo_mask
    kernel = np.ones((2, 2), np.uint8)
    la_myo_mask = cv2.dilate(la_myo_mask.astype(np.uint8), kernel, iterations=1)

    # _, la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, ra_myo_contour, _ = cv2.findContours(ra_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ra_myo_contour, _ = cv2.findContours(ra_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # plt.imshow(ra_myo_mask)
    # plt.show()

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[la_myo_contour[0]], color=(255, 255, 255))
    cv2.fillPoly(new_mask, pts=[la_myo_contour[1]], color=(255, 255, 255))
    # _, la_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    la_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[0]], color=(255, 255, 255))
    # plt.imshow(new_mask)
    # plt.show()
    # _, lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[0]], color=(255, 255, 255))
    lv_mask = new_mask.copy()
    # _, lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    for i in range(len(ra_myo_contour)):
        cv2.fillPoly(new_mask, pts=[ra_myo_contour[i]], color=(255, 255, 255))
    # cv2.fillPoly(new_mask, pts=[ra_myo_contour[1]], color=(255, 255, 255))
    # _, ra_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ra_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rv_mask = (rv_myo_mask + lv_myo_mask).astype(np.float32)
    # _, rv_myo_contour, _ = cv2.findContours((rv_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_myo_contour, _ = cv2.findContours((rv_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(rv_mask, pts=[rv_myo_contour[0]], color=(255, 255, 255))
    cv2.fillPoly(rv_mask, pts=[rv_myo_contour[1]], color=(255, 255, 255))
    rv_mask -= lv_mask
    """
    exclude rv wall
    """
    rv_mask *= -(rv_wall_mask_right - 1)
    # _, rv_contour, _ = cv2.findContours(rv_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_contour, _ = cv2.findContours(rv_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)

    label_list = important_class_dict["A4CH"]
    label_dict = {label: i + 1 for i, label in enumerate(label_list)}

    new_mask = cv2.drawContours(new_mask, lv_contour, -1, label_dict["LV cavity"], -1)
    new_mask = cv2.drawContours(new_mask, la_contour, -1, label_dict["LA cavity"], -1)
    new_mask = cv2.drawContours(new_mask, lv_wall_contour, -1, label_dict["LV wall"], -1)
    new_mask = cv2.drawContours(new_mask, ra_contour, -1, label_dict["RA cavity"], -1)
    new_mask = cv2.drawContours(new_mask, rv_contour, -1, label_dict["RV cavity"], -1)

    new_mask[mv_mask] = label_dict["LV cavity"]
    new_mask[tv_mask] = label_dict["RV cavity"]
    # plt.imshow(new_mask)
    # plt.show()
    return new_mask


def process_a3c(image):
    lv_myo_mask = image == Tags["lv_myocardium"]
    la_myo_mask = image == Tags["la_myocardium"]
    mv_mask = image == Tags["mitral_valve"]
    av_mask = image == Tags["aortic_valve"]
    aorta_mask = image == Tags["aorta"]
    # _, small_lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_lv_myo_contour = small_lv_myo_contour[np.argmin([len(c) for c in small_lv_myo_contour])]
    small_lv_myo = np.zeros_like(lv_myo_mask)
    small_lv_myo = cv2.drawContours(small_lv_myo.astype(np.uint8), [small_lv_myo_contour], -1, 1, -1)

    la_myo_mask += mv_mask
    lv_myo_mask += mv_mask
    lv_myo_mask += av_mask
    aorta_mask += av_mask

    # _, la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, ra_myo_contour, _ = cv2.findContours(ra_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, aorta_contour, _ = cv2.findContours(aorta_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    aorta_contour, _ = cv2.findContours(aorta_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[la_myo_contour[0]], color=(255, 255, 255))
    cv2.fillPoly(new_mask, pts=[la_myo_contour[1]], color=(255, 255, 255))
    # _, la_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    la_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[1]], color=(255, 255, 255))
    # _, lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[0]], color=(255, 255, 255))
    lv_mask = new_mask.copy()
    # _, lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[aorta_contour[1]], color=(255, 255, 255))
    # _, aorta_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    aorta_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)

    label_list = important_class_dict["A3CH"]
    label_dict = {label: i + 1 for i, label in enumerate(label_list)}

    new_mask = cv2.drawContours(new_mask, lv_contour, -1, label_dict["LV cavity"], -1)
    new_mask = cv2.drawContours(new_mask, la_contour, -1, label_dict["LA cavity"], -1)
    new_mask = cv2.drawContours(new_mask, lv_wall_contour, -1, label_dict["LV wall"], -1)
    new_mask = cv2.drawContours(new_mask, aorta_contour, -1, label_dict["Aorta"], -1)

    new_mask[mv_mask] = label_dict["LV cavity"]
    new_mask[av_mask] = label_dict["LV cavity"]
    new_mask *= -(small_lv_myo - 1)

    return new_mask


def process_psax(image):
    lv_myo_mask = image == Tags["lv_myocardium"]
    # la_myo_mask = fill_in(image == Tags["la_myocardium"])
    # mv_mask = fill_in(image == Tags["mitral_valve"])
    # av_mask = fill_in(image == Tags["aortic_valve"])
    rv_myo_mask = image == Tags["rv_myocardium"]
    # aorta_mask = fill_in(image == Tags["aorta"])
    # _, small_lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # la_myo_mask += mv_mask
    # lv_myo_mask += mv_mask
    # lv_myo_mask += av_mask
    # aorta_mask += av_mask

    # _, la_myo_contour, _ = cv2.findContours(la_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_myo_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, rv_myo_contour, _ = cv2.findContours((rv_myo_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_myo_contour, _ = cv2.findContours((rv_myo_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[1]], color=(255, 255, 255))
    # _, lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_contour, _ = cv2.findContours(new_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)
    cv2.fillPoly(new_mask, pts=[lv_myo_contour[0]], color=(255, 255, 255))
    lv_mask = new_mask.copy()
    # _, lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lv_wall_contour, _ = cv2.findContours(lv_myo_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rv_mask = (rv_myo_mask + lv_myo_mask).astype(np.float32)
    # _, rv_myo_contour, _ = cv2.findContours((rv_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_myo_contour, _ = cv2.findContours((rv_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(rv_mask, pts=[rv_myo_contour[0]], color=(255, 255, 255))
    cv2.fillPoly(rv_mask, pts=[rv_myo_contour[1]], color=(255, 255, 255))
    rv_mask -= lv_mask
    rv_mask *= -(rv_myo_mask - 1)
    # _, rv_contour, _ = cv2.findContours(rv_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv_contour, _ = cv2.findContours(rv_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_mask = np.zeros_like(image)

    label_list = important_class_dict["PSAX mid"]
    label_dict = {label: i + 1 for i, label in enumerate(label_list)}

    new_mask = cv2.drawContours(new_mask, lv_contour, -1, label_dict["LV cavity"], -1)
    new_mask = cv2.drawContours(new_mask, lv_wall_contour, -1, label_dict["LV wall"], -1)
    new_mask = cv2.drawContours(new_mask, rv_contour, -1, label_dict["RV cavity"], -1)

    return new_mask


def fill_mask(mask, view):
    if view == "A2CH":
        mask[mask == Tags["appendage"]] = Tags["la_myocardium"]
        mask[mask == Tags["appendage_border"]] = Tags["la_myocardium"]
        mask[mask == Tags["left_superior_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_inferior_pulmonary_vein_border"]] = Tags["la_myocardium"]
    if view == "A4CH":
        mask[mask == Tags["left_inferior_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["left_inferior_pulmonary_vein_border"]] = Tags["la_myocardium"]
        mask[mask == Tags["left_superior_pulmonary_vein_border"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_inferior_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_superior_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_superior_pulmonary_vein_border"]] = Tags["la_myocardium"]
        mask[mask == Tags["inferior_vena_cava"]] = Tags["ra_myocardium"]
        mask[mask == Tags["inferior_vena_cava_border"]] = Tags["ra_myocardium"]
    if view == "A3CH":
        mask[mask == Tags["left_inferiror_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_superior_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["left_inferior_pulmonary_vein_border"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_superior_pulmonary_vein_border"]] = Tags["la_myocardium"]
    if view == "PLAX":
        mask[mask == Tags["left_inferiror_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_superior_pulmonary_vein"]] = Tags["la_myocardium"]
        mask[mask == Tags["left_inferior_pulmonary_vein_border"]] = Tags["la_myocardium"]
        mask[mask == Tags["right_superior_pulmonary_vein_border"]] = Tags["la_myocardium"]

    return mask


process_view = {
    "A2CH": process_a2c,
    "A3CH": process_a3c,
    "A4CH": process_a4c,
    # ("lv_plax"= process_lv_plax),
    "PSAX apex": process_psax,
    "PSAX basal": process_psax,
    "PSAX mid": process_psax,
}
