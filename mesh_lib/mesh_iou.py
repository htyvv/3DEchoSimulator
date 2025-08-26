import pyvista as pv
import numpy as np
import torch

import glob
import os

import numpy as np
import pandas as pd
from natsort import natsorted
from pyntcloud import PyntCloud

from Epix2vox_reconstruction.data_extraction.slicing.config import cfg
from Epix2vox_reconstruction.data_extraction.slicing.plane_extract import prepare_meshes
import Epix2vox_reconstruction.data_extraction.slicing.maths_utils as maths_utils


def convert_vtk2np_voxels(in_mesh, vol_resolution):
    df = pd.DataFrame(data=in_mesh.points, columns=["x", "y", "z"])
    cloud = PyntCloud(df)

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=vol_resolution, n_y=vol_resolution, n_z=vol_resolution)
    voxelgrid = cloud.structures[voxelgrid_id]
    binary_voxel_array = voxelgrid.get_feature_vector(mode="binary")
    # binary_voxel_array = closing(Binary_voxel_array, cube(2))
    return binary_voxel_array


def calc_iou(mesh_1, mesh_2):
    intersection = torch.sum(mesh_1.mul(mesh_2)).float()
    union = torch.sum(torch.ge(mesh_1.add(mesh_2), 1)).float()

    return intersection / union


def subsample_mesh(cfg, in_mesh):
    """Subsamples the mesh to a lower resolution given a subsampling factor. The subsampled meshes are used as a method
    to calculate centre of masses more quickly than the very high resolution full meshes.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire full resolution cardiac mesh.

    Returns:
        pyvista.core.pointset.UnstructuredGrid: Lower resolution subsampled mesh.
    """
    return in_mesh.extract_cells(range(0, in_mesh.n_cells, cfg.PARAMETERS.SUBSAMPLE_FACTOR))


base_dir = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned"

mesh_1_dir = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/Final_models_01+full_heart_mesh_001.vtk"

stat_result_dir = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/z_stats"
mesh_1 = pv.read(mesh_1_dir)

mesh_1 = maths_utils.translate_mesh_to_origin(mesh_1)
mesh_1 = subsample_mesh(cfg, mesh_1)
mesh_1 = pv.wrap(mesh_1).threshold((1, 5), invert=False, scalars="elemTag", preference="cell")
voxel_1 = convert_vtk2np_voxels(mesh_1, 64)


all_iou = []
for mesh_2_dir in glob.glob(os.path.join(base_dir, "*.vtk"))[1:]:
    mesh_2 = pv.read(mesh_2_dir)
    mesh_2 = maths_utils.translate_mesh_to_origin(mesh_2)
    mesh_2 = subsample_mesh(cfg, mesh_2)
    mesh_2 = pv.wrap(mesh_2).threshold((1, 5), invert=False, scalars="elemTag", preference="cell")
    voxel_2 = convert_vtk2np_voxels(mesh_2, 64)

    iou = calc_iou(torch.from_numpy(voxel_1), torch.from_numpy(voxel_2))
    print("IOU:", iou)
    print("Mesh 1:", mesh_1_dir)
    print("Mesh 2:", mesh_2_dir)
    print("=====================================")

    all_iou.append(iou)
    # raise ValueError
##
np.save(os.path.join(stat_result_dir, "iou_lv_rv_la_ra_aorta.npy"), np.array(all_iou))
##
import matplotlib.pyplot as plt

iou_all = np.load("/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/z_stats/iou_all.npy")
iou_lv = np.load("/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/z_stats/iou_lv.npy")
iou_lv_rv = np.load("/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/z_stats/iou_lv_rv.npy")
iou_lv_rv_la_ra_aorta = np.load("/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/z_stats/iou_lv_rv_la_ra_aorta.npy")

iou_all_score = 0.6772
iou_lv_rv_score = 0.7022
iou_lv_score = 0.7745

plt.boxplot(
    [iou_all, iou_lv_rv_la_ra_aorta, iou_lv_rv, iou_lv],
    vert=True,
    patch_artist=True,
    labels=["All", "LV+RV+LA+RA+Aorta", "LV+RV", "LV"],
)
plt.axhline(y=iou_all_score, color="r", linestyle="--", label="All")
plt.axhline(y=iou_lv_rv_score, color="g", linestyle="--", label="LV+RV")
plt.axhline(y=iou_lv_score, color="b", linestyle="--", label="LV")
plt.legend()

plt.ylabel("IOU")

plt.ylim(0.3, 0.9)

plt.title("IOU of meshes")
plt.show()
