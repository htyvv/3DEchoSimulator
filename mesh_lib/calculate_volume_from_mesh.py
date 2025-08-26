import glob
import os

import numpy as np
import pandas as pd
from natsort import natsorted
from pyntcloud import PyntCloud

import sys

sys.path.append(os.getcwd())

from Epix2vox_reconstruction.data_extraction.slicing.config import cfg
from Epix2vox_reconstruction.data_extraction.slicing.plane_extract import prepare_meshes

from mesh.mesh_handler import MeshHandler

import matplotlib.pyplot as plt
import cv2
import pyvista as pv
from tqdm import tqdm


def convert_vtk2np_voxels(in_mesh, vol_resolution_x, vol_resolution_y, vol_resolution_z):
    df = pd.DataFrame(data=in_mesh.points, columns=["x", "y", "z"])
    cloud = PyntCloud(df)
    kwargs = {"regular_bounding_box": False}
    voxelgrid_id = cloud.add_structure(
        "voxelgrid", n_x=vol_resolution_x, n_y=vol_resolution_y, n_z=vol_resolution_z, regular_bounding_box=False
    )
    voxelgrid = cloud.structures[voxelgrid_id]

    binary_voxel_array = voxelgrid.get_feature_vector(mode="binary")
    # binary_voxel_array = voxelgrid.get_feature_vector(mode="_mean")
    # binary_voxel_array = closing(Binary_voxel_array, cube(2))
    return binary_voxel_array


# main()
# Change these values
# data_folder = r"/mnt/NAS01/jaeik/echo4d/Epix2vox_reconstruction/DB/heart_vtk/"
# data_folder = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/"
# data_folder = "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/"
# data_folder = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned_lv"
data_folder = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned_a2c_plane_projected"
# save_folder = r"/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned_a2c_plane_projected"
vol_resolution = 200
#
# os.makedirs(save_folder, exist_ok=True)

all_data_paths = natsorted(glob.glob(os.path.join(data_folder, "*.vtk")))

for file in tqdm(all_data_paths[:]):
    mesh_handler = MeshHandler(cfg, file)
    # mesh_handler.translate_mesh_to_a2c_plane()
    case_name = file.split(os.sep)[-1].split(".")[0]
    # mesh_handler.mesh_origin.save(os.path.join(save_folder, case_name + ".vtk"))
    # loaded_mesh = mesh_handler.mesh_origin.copy()
    loaded_mesh = pv.wrap(mesh_handler.mesh_origin).threshold(
        (1, 1), invert=False, scalars="elemTag", preference="cell"
    )
    loaded_mesh_mv = pv.wrap(mesh_handler.mesh_origin).threshold(
        (7, 7), invert=False, scalars="elemTag", preference="cell"
    )
    loaded_mesh_av = pv.wrap(mesh_handler.mesh_origin).threshold(
        (9, 9), invert=False, scalars="elemTag", preference="cell"
    )
    loaded_mesh = loaded_mesh + loaded_mesh_mv + loaded_mesh_av
    # loaded_mesh = pv.voxelize(loaded_mesh, density=1.0, check_surface=False)
    # vol = pv.UnstructuredGrid(loaded_mesh.points, loaded_mesh.cells)
    # vol.volume
    # loaded_mesh, __ = prepare_meshes(cfg, file)
    x_resol = int(abs(loaded_mesh.bounds[1] - loaded_mesh.bounds[0]))
    y_resol = int(abs(loaded_mesh.bounds[3] - loaded_mesh.bounds[2]))
    z_resol = int(abs(loaded_mesh.bounds[5] - loaded_mesh.bounds[4]))
    print(x_resol, y_resol, z_resol)
    out_voxel_array = convert_vtk2np_voxels(loaded_mesh, x_resol, y_resol, z_resol)
    out_voxel_array = np.pad(out_voxel_array, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=0)

    lv_voxel = np.zeros(out_voxel_array.shape, dtype=np.uint8)
    for i in range(0, out_voxel_array.shape[0], 1):
        h, w = out_voxel_array[0, :, :].shape[0], out_voxel_array[0, :, :].shape[1]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        holes = cv2.floodFill(out_voxel_array[i, :, :].copy().astype(np.uint8), mask, (0, 0), 1)[1]

        holes = abs(1 - holes)

        _, contours, hierarchy = cv2.findContours(holes.astype(np.uint8) * 100, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i_cnt, cnt in enumerate(contours):
            if len(cnt) < 10:
                # Fill the holes in the original image
                cv2.drawContours(out_voxel_array[i, :, :], [cnt], 0, (1), -1)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        holes = cv2.floodFill(out_voxel_array[i, :, :].copy().astype(np.uint8), mask, (0, 0), 1)[1]
        holes = abs(1 - holes)

        lv_voxel[i, :, :] = holes

    lv_voxel_cleaned = np.zeros(lv_voxel.shape, dtype=np.uint8)
    for i, slice in enumerate(lv_voxel):
        _, contours, hierarchy = cv2.findContours(
            slice.astype(np.uint8) * 100, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )  # Use cv2.CCOMP for two level hierarchy

        if len(contours) != 0:
            # find the max length contour
            max_cnt = contours[np.argmax([len(cnt) for cnt in contours])]
        if max_cnt is not None:
            cv2.drawContours(lv_voxel_cleaned[i, :, :], [max_cnt], 0, (1), -1)

        # blur
        blur = cv2.GaussianBlur(lv_voxel_cleaned[i, :, :], (0, 0), sigmaX=2, sigmaY=2)
        # plt.imshow(blur)
        # plt.show()

        # divide
        # divide = cv2.divide(lv_voxel_cleaned[i, :, :], blur, scale=1)
        # thresh = cv2.threshold(divide, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        lv_voxel_cleaned[i, :, :] = blur

    for i in range(lv_voxel_cleaned.shape[0]):
        os.makedirs(f"./tmp/mesh_slice", exist_ok=True)
        plt.imsave(f"./tmp/mesh_slice/test{i}.png", lv_voxel_cleaned[i, :, :])
    raise ValueError
