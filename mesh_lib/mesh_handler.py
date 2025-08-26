from pyntcloud import PyntCloud
import pandas as pd
import numpy as np

import os
import sys
# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from mesh_lib.slicing import *
from mesh_lib.inhouse_style_mask import process_view as inhouse_style_process
from mesh_lib.inhouse_style_mask import fill_mask
import mesh_lib.image_operations as image_operations
from mesh_lib.synthetic_mask import *
from mesh_lib.math_utils import *
from mesh_lib.utils import *


# def convert_vtk2np_voxels(in_mesh, vol_resolution_x, vol_resolution_y, vol_resolution_z):
#     df = pd.DataFrame(data=in_mesh.points, columns=["x", "y", "z"])
#     cloud = PyntCloud(df)
#
#     voxelgrid_id = cloud.add_structure(
#         "voxelgrid", n_x=vol_resolution_x, n_y=vol_resolution_y, n_z=vol_resolution_z, regular_bounding_box=False
#     )
#     voxelgrid = cloud.structures[voxelgrid_id]
#     binary_voxel_array = voxelgrid.get_feature_vector(mode="binary")
#
#     return binary_voxel_array


def compute_voxel(mesh=None, target_shape=None, fill_inside=False, target_chamber=None, blur_threshold=0.5):
    df = pd.DataFrame(data=mesh.points, columns=["y", "x", "z"])

    x_resol = int(abs(mesh.bounds[1] - mesh.bounds[0]))
    y_resol = int(abs(mesh.bounds[3] - mesh.bounds[2]))
    z_resol = int(abs(mesh.bounds[5] - mesh.bounds[4]))

    size_x = None
    size_y = None
    size_z = None

    if target_shape is not None:
        x_resol = target_shape - 2
        y_resol = x_resol
        z_resol = x_resol
        size_x = x_resol
        size_y = y_resol
        size_z = z_resol
        # add [0, 0, 0] row to the df
        df = pd.concat([pd.DataFrame([[0, 0, 0]], columns=["y", "x", "z"]), df], ignore_index=True)
        # add [x_resol, y_resol, z_resol] row to the df
        df = pd.concat([df, pd.DataFrame([[x_resol, y_resol, z_resol]], columns=["y", "x", "z"])], ignore_index=True)

    cloud = PyntCloud(df)

    voxelgrid_id = cloud.add_structure(
        "voxelgrid",
        n_x=x_resol,
        n_y=y_resol,
        n_z=z_resol,
        # size_x=size_x,
        # size_y=size_y,
        # size_z=size_z,
        regular_bounding_box=False,
    )
    voxelgrid = cloud.structures[voxelgrid_id]
    out_voxel_array = voxelgrid.get_feature_vector(mode="binary")
    spacing_info = voxelgrid.shape
    # for i in range(10, 85, 5):
    #
    #     plt.imshow(out_voxel_array[:, :, i])
    #     plt.show()

    # pad the voxel array
    out_voxel_array = np.pad(out_voxel_array, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=0)
    # plt.imshow(out_voxel_array[:, :, 30])
    # plt.show()
    for i in range(0, out_voxel_array.shape[0], 1):
        blur = cv2.GaussianBlur(out_voxel_array[:, :, i], (0, 0), sigmaX=2, sigmaY=2)
        out_voxel_array[:, :, i] = (blur > blur_threshold).astype(np.uint8)

    # for i in range(10, 85, 3):

    #     plt.imshow(out_voxel_array[:, :, i])
    #     plt.show()
    #
    # plt.imshow(out_voxel_array[:, :, 30])
    # plt.show()

    if target_chamber == "rv":
        # find apex
        apex_candidates = np.where(out_voxel_array == 1)
        apex = np.argmin(apex_candidates[0])
        apex = [apex_candidates[0][apex], apex_candidates[1][apex], apex_candidates[2][apex]]
        # print(apex)
        out_voxel_array[:, :, 0 : apex[2]] = 0

    if fill_inside:
        for i in range(0, out_voxel_array.shape[0], 1):
            h, w = out_voxel_array[0, :, :].shape[0], out_voxel_array[0, :, :].shape[1]
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            holes = cv2.floodFill(out_voxel_array[i, :, :].copy().astype(np.uint8), mask, (0, 0), 1)[1]

            holes = abs(1 - holes)

            out_voxel_array[i, :, :] = holes

        lv_voxel_cleaned = np.zeros(out_voxel_array.shape, dtype=np.uint8)
        for i, slice in enumerate(out_voxel_array):
            # _, contours, hierarchy = cv2.findContours(
            contours, hierarchy = cv2.findContours(
                slice.astype(np.uint8) * 100, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            max_cnt = None
            if len(contours) != 0:
                # find the max length contour
                max_cnt = contours[np.argmax([len(cnt) for cnt in contours])]

            if max_cnt is not None:
                cv2.drawContours(lv_voxel_cleaned[i, :, :], [max_cnt], 0, (1), -1)

    else:
        lv_voxel_cleaned = out_voxel_array
    return {"voxel": lv_voxel_cleaned.astype(np.uint8), "spacing": spacing_info}


def pad_echo_inhouse_style_masks(data_dict):
    """
    Pads all echo_inhouse_style_masks in the dataset to have the same dimensions, based on the largest width and height found.
    """

    def find_max_h_w(data_dict):
        max_height = 0
        max_width = 0
        for view, view_data in data_dict.items():
            if view == "voxel_in_image_space":  # Skip the voxel_in_image_space entry
                continue
            mask = view_data["echo_inhouse_style_mask"]
            if mask.shape[0] > max_height:
                max_height = mask.shape[0]
            if mask.shape[1] > max_width:
                max_width = mask.shape[1]

        return max_height, max_width

    max_height, max_width = find_max_h_w(data_dict)

    for view, view_data in data_dict.items():
        if view == "voxel_in_image_space":  # Skip the voxel_in_image_space entry
            continue
        mask = view_data["echo_inhouse_style_mask"]
        pad_height = max_height - mask.shape[0]
        pad_width = max_width - mask.shape[1]
        pad_size = [pad_height // 2, pad_height - pad_height // 2, pad_width // 2, pad_width - pad_width // 2]
        padded_mask = np.pad(
            mask, [[pad_size[0], pad_size[1]], [pad_size[2], pad_size[3]]], mode="constant", constant_values=0,
        )

        view_data["padded_echo_inhouse_style_mask"] = padded_mask
        view_data["space_info"]["padded echo inhouse style mask bounds"] = padding_bounds(
            view_data["space_info"]["echo inhouse style mask bounds"], pad_size
        )

    return data_dict


class Mesh:
    def __init__(self, cfg, mesh, scale=1, origin=True):
        self.cfg = cfg
        self.scale = scale
        # self.mesh_origin, self.subsampled_mesh = self.load_mesh(mesh_dir)

        print(f"Loading {mesh}")
        if type(mesh) == str:
            self.mesh = pv.get_reader(mesh).read()
        else:
            self.mesh = mesh
        if origin:
            self.mesh = maths_utils.translate_mesh_to_origin(self.mesh)
        # scale down the mesh
        self.mesh.points[:] *= self.scale

        self.subsampled_mesh = subsample_mesh(cfg, self.mesh)
        print(f"Loaded {mesh}")

        self.reset_plotter()

    def load_mesh(self, mesh_dir):
        origin_centred_mesh, low_res_mesh = prepare_meshes(self.cfg, mesh_dir)
        return origin_centred_mesh, low_res_mesh

    def generate_echo_slices_per_view(self, view_list):
        """
        Generates spacing-consistent echo slices for each view in the provided view list for a patient unit.
        """
        echo_slices = {}
        for view in view_list:
            # Get a single standard cardiac slice
            cardiac_slice, space_info = self.get_cardiac_slice(view, plot_flag=False, background=False)

            # Convert the cardiac slice to a numpy array
            projected_slice_arr, space_info = self.get_numpy_slice_arr(cardiac_slice, space_info)

            # Process the numpy slice array to create an inhouse style mask
            inhouse_style_mask = self.inhouse_style_process(projected_slice_arr, view)

            # Apply US processing to the inhouse style mask to generate the echo inhouse style mask
            echo_inhouse_style_mask, space_info = self.us_process(inhouse_style_mask, view, space_info)

            # Store the results in the echo_slices dictionary
            echo_slices[view] = {
                "projected_slice_arr": projected_slice_arr,
                "inhouse_style_mask": inhouse_style_mask,
                "echo_inhouse_style_mask": echo_inhouse_style_mask,
                "space_info": space_info,
            }

        # Pad the echo inhouse style masks to have the same dimensions
        pad_echo_inhouse_style_masks(echo_slices)

        return echo_slices

    def plot_cardiac_mesh(self, style=None):
        if not hasattr(self, "plotter"):
            self.reset_plotter()

        if style == "wireframe":
            color = "white"
            smooth_shading = True
            opacity = 0.15
            self.plotter.add_mesh(
                self.mesh, style=style, color=color, smooth_shading=smooth_shading, opacity=opacity
            )  # Cardiac mesh
        else:
            self.plotter.add_mesh(self.mesh, style=style)

    def get_key_cardiac_points(
        self, selected_view=["PLAX", "PSAX basal", "PSAX mid", "PSAX apex", "A4CH", "A2CH", "A3CH"]
    ):
        self.cardiac_out_points = get_cardiac_images(self.cfg, self.mesh, self.subsampled_mesh, selected_view)
        return self.cardiac_out_points

    def get_cardiac_slice(self, view, plot_flag=False, background=False):
        if view not in self.cfg.DATA_OUT.SELECTED_VIEWS:
            raise ValueError(f"View {view} not in {self.cfg.DATA_OUT.SELECTED_VIEWS}")

        if not hasattr(self, "cardiac_out_points"):
            self.get_key_cardiac_points()

        space_info = {}

        view_data, land_marks = self.cardiac_out_points[view]
        normal, plane_origin = view_data
        if len(normal) == 4:
            normal = normal[:3]

        pv_cardiac_slice = self.mesh.slice(normal=normal, origin=plane_origin)

        # cardiac_slice = slice_with_plane(self.mesh, origin=plane_origin, normal=normal)
        # pv_cardiac_slice = pv.wrap(cardiac_slice)
        slice_plane_center = np.mean(pv_cardiac_slice.points, axis=0)
        if plot_flag:
            if not hasattr(self, "plotter"):
                self.reset_plotter()

            self.plotter.add_mesh(
                pv_cardiac_slice,
                scalars=self.cfg.LABELS.LABEL_NAME,
                smooth_shading=True,
                show_scalar_bar=False,
                # Extracted slice
            )

            if background:
                bounds = list(pv_cardiac_slice.bounds)
                bounds[0] = -90
                bounds[1] = 90
                bounds[2] = -90
                bounds[3] = 90
                true_plane = pv.Plane(
                    center=slice_plane_center - 0.1,
                    # center=[0, 0, 0],
                    direction=normal,
                    i_size=(bounds[1] - bounds[0]),
                    j_size=bounds[3] - bounds[2],
                )
                self.plotter.add_mesh(
                    true_plane, color=self.cfg.DATA_OUT.SAVE_BCKGD_CLR, smooth_shading=True
                )  # Blank plane

        space_info["sliced plane normal"] = normal
        space_info["sliced plane center"] = slice_plane_center
        space_info["sliced plane landmarks"] = land_marks

        return pv_cardiac_slice, space_info  # normal, slice_plane_center, land_marks

    def get_multiple_cardiac_slices(self, view, space_info):
        if view not in self.cfg.DATA_OUT.SELECTED_VIEWS:
            raise ValueError(f"View {view} not in {self.cfg.DATA_OUT.SELECTED_VIEWS}")

        if not hasattr(self, "cardiac_out_points"):
            self.get_key_cardiac_points()

        view_data, land_marks = self.cardiac_out_points[view]
        normal, plane_origin = view_data
        if len(normal) == 4:
            normal = normal[:3]

        v = np.cross((1, 0, 0), normal)
        ANGLE_STEP = 3
        slice_config = {
            "A2CH": {"angle": [[-7, 12, ANGLE_STEP], [3, 12, ANGLE_STEP]], "axis": [v, (0, 0, 1)]},
            "A4CH": {"angle": [[-6, 0, ANGLE_STEP], [-12, 0, ANGLE_STEP]], "axis": [v, (1, 0, 0)]},
            "PSAX mid": {
                "angle": [[-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP]],
                "axis": [v, (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            },
            "PSAX apex": {
                "angle": [[-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP]],
                "axis": [v, (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            },
            "PSAX basal": {
                "angle": [[-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP], [-7, 7, ANGLE_STEP]],
                "axis": [v, (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            },
        }

        for aug_info in zip(slice_config[view]["angle"], slice_config[view]["axis"]):
            aug_angle_rng, aug_axis = aug_info
            for aug_angle in range(*aug_angle_rng):

                normal, plane_origin = rotate_plane(normal, plane_origin, aug_angle, aug_axis)
                pv_cardiac_slice = self.mesh.slice(normal=normal, origin=plane_origin)
                slice_plane_center = np.mean(pv_cardiac_slice.points, axis=0)

                space_info["sliced plane normal"] = normal
                space_info["sliced plane center"] = slice_plane_center
                space_info["sliced plane landmarks"] = land_marks

                yield pv_cardiac_slice, space_info

    def get_numpy_slice_arr(self, cardiac_slice, space_info):
        """Convert cardiac slice to numpy array.
        :param view: view name
        :param cardiac_slice: vtkmodules.vtkCommonDataModel.vtkPolyData
        :param normal: normal vector of the plane
        :return: numpy array
        """
        if "sliced plane normal" not in space_info:
            raise ValueError("sliced plane normal is required to project the slice into xy plane")
        normal = space_info["sliced plane normal"]
        transformed_slice = pv.wrap(maths_utils.rotate_to_xy_plane(cardiac_slice, normal))
        transformed_slice_bounds = transformed_slice.bounds
        projected_origin = np.mean(transformed_slice.points, axis=0)
        slice_arr = get_plane_numpy(self.cfg, transformed_slice)

        space_info["projected plane center"] = projected_origin
        space_info["projected plane bounds"] = transformed_slice_bounds
        return slice_arr, space_info  # projected_origin, transformed_slice_bounds

    def revert_projected_array(self, projected_slice_arr, projected_slice_bounds, space_info):
        if "projected plane center" not in space_info:
            raise ValueError("projected plane center is required to revert the projected slice")

        if "sliced plane normal" not in space_info:
            raise ValueError("sliced plane normal is required to revert the projected slice")

        if space_info["flip"]:
            projected_slice_arr = np.flip(projected_slice_arr, axis=0)

        projected_center = space_info["projected plane center"]
        normal = space_info["sliced plane normal"]

        projected_slice_arr = projected_slice_arr.T
        slice_grid = get_grid_from_2d_arr(projected_slice_arr, self.cfg.LABELS.LABEL_NAME, projected_slice_bounds)

        # translate the plane to the original projected center
        slice_grid = translate_plane_to_point(slice_grid, projected_center)
        slice_grid = pv.wrap(revert_plane_to_original_normal(slice_grid, normal))

        return slice_grid

    def save_slice_screenshot(self, transformed_slice, save_path):
        save_plane_img(self.cfg, transformed_slice, save_path)

    def inhouse_style_process(self, slice_arr, view):
        slice_arr = fill_mask(slice_arr, view)

        inhouse_stye_mask = inhouse_style_process[view](slice_arr)
        return inhouse_stye_mask

    def us_process(self, slice_arr, view, space_info):
        if "projected plane bounds" not in space_info:
            raise ValueError("projected plane bounds is required to pad the projected plane bounds")

        projected_slice_bounds = space_info["projected plane bounds"]
        slice_arr, pad_size = mask_augmentation(slice_arr, view)
        padded_projected_slice_bounds = padding_bounds(projected_slice_bounds[:4], pad_size)

        if view == "A4CH":
            slice_arr = cv2.flip(slice_arr, 0)
        slice_arr = apply_us_cone(slice_arr)
        # if view == "A4CH":
        #     slice_arr = cv2.flip(slice_arr, 0)

        space_info["echo inhouse style mask bounds"] = padded_projected_slice_bounds
        space_info["padding size"] = pad_size

        space_info["flip"] = view == "A4CH"

        return slice_arr, space_info

    def translate_mesh_to_a2c_plane(self):
        if not hasattr(self, "cardiac_out_points"):
            self.get_key_cardiac_points(["A2CH"])
        elif "A2CH" not in self.cardiac_out_points:
            self.get_key_cardiac_points(["A2CH"])

        a2ch_normal = self.cardiac_out_points["A2CH"][0][0]

        self.mesh = pv.UnstructuredGrid(maths_utils.rotate_to_xy_plane(self.mesh, a2ch_normal))
        self.subsampled_mesh = subsample_mesh(self.cfg, self.mesh)

        self.get_key_cardiac_points(["A2CH"])

        apex_pts_xy = np.array(self.cardiac_out_points["A2CH"][1][1][:2])
        mv_pts_xy = np.array(self.cardiac_out_points["A2CH"][1][2][:2])

        # calculate angle between x axis and line from apex to mitral valve
        apex_mv_vec = mv_pts_xy - apex_pts_xy
        apex_mv_vec = apex_mv_vec / np.linalg.norm(apex_mv_vec)
        x_axis = np.array([1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        angle = np.arccos(np.dot(apex_mv_vec, x_axis))
        angle = np.rad2deg(angle)

        self.mesh = self.mesh.rotate_z(90 - angle)

        self.mesh = maths_utils.translate_mesh_to_origin(self.mesh)
        self.subsampled_mesh = subsample_mesh(self.cfg, self.mesh)

    def transform_mesh_to_slice_space(self, mesh, bounds, space_info, slice_arr, plot_flag=False):
        if "sliced plane normal" not in space_info:
            raise ValueError("sliced plane normal is required to transform the mesh to slice space")

        normal = space_info["sliced plane normal"]
        translated_mesh = pv.wrap(
            # maths_utils.rotate_to_xy_plane(self.lv_wall_mesh + self.mv_mesh + self.av_mesh, normal)
            maths_utils.rotate_to_xy_plane(mesh, normal)
        )

        # translate the translated mesh to the center of the slice array
        translated_mesh = translated_mesh.translate(
            -np.array(
                [
                    # space_info["padded projected plane bounds"][0],
                    # space_info["padded projected plane bounds"][2],
                    bounds[0],
                    bounds[2],
                    # space_info["projected plane bounds"][4],
                    translated_mesh.bounds[4],
                ]
            )
        )

        if plot_flag:
            plotter = pv.Plotter()
            plotter.add_mesh(translated_mesh, color="white", opacity=0.5)

            # make structured grid for the slice array
            slice_grid = get_grid_from_2d_arr(
                slice_arr.T, self.cfg.LABELS.LABEL_NAME, [0, slice_arr.shape[1] + 1, 0, slice_arr.shape[0] + 1, 0, 1]
            )
            plotter.add_mesh(slice_grid, opacity=1)

            box_size = len(slice_arr)

            line = pv.Line((0, 0, 0), (0, 0, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, 0, 0), (0, box_size, 0))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, 0, 0), (box_size, 0, 0))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, box_size, 0), (0, box_size, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, box_size, 0), (box_size, box_size, 0))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((box_size, 0, 0), (box_size, 0, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((box_size, 0, 0), (box_size, box_size, 0))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, 0, box_size), (0, box_size, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, 0, box_size), (box_size, 0, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((0, box_size, box_size), (box_size, box_size, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((box_size, 0, box_size), (box_size, box_size, box_size))
            plotter.add_mesh(line, color="red", line_width=5)
            line = pv.Line((box_size, box_size, 0), (box_size, box_size, box_size))
            plotter.add_mesh(line, color="red", line_width=5)

            plotter.show_grid()

            plotter.show()

        return translated_mesh

    @property
    def lv_wall_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["lv_myocardium"], Tags["lv_myocardium"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def la_wall_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["la_myocardium"], Tags["la_myocardium"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def rv_wall_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["rv_myocardium"], Tags["rv_myocardium"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def aorta_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["aorta"], Tags["aorta"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def pulmonary_artery_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["pulmonary_artery"], Tags["pulmonary_artery"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def mv_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["mitral_valve"], Tags["mitral_valve"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def tv_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["tricuspid_valve"], Tags["tricuspid_valve"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def av_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["aortic_valve"], Tags["aortic_valve"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def pv_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["pulmonary_valve"], Tags["pulmonary_valve"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def appendage_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["appendage"], Tags["appendage"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def appendage_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["appendage_border"], Tags["appendage_border"]), invert=False, scalars="elemTag", preference="cell"
        )

    @property
    def left_superior_pulmonary_vein_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["left_superior_pulmonary_vein"], Tags["left_superior_pulmonary_vein"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def left_inferior_pulmonary_vein_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["left_inferior_pulmonary_vein"], Tags["left_inferior_pulmonary_vein"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def left_superior_pulmonary_vein_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["left_superior_pulmonary_vein_border"], Tags["left_superior_pulmonary_vein_border"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def left_inferior_pulmonary_vein_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["left_inferior_pulmonary_vein_border"], Tags["left_inferior_pulmonary_vein_border"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def right_superior_pulmonary_vein_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["right_superior_pulmonary_vein"], Tags["right_superior_pulmonary_vein"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def right_inferior_pulmonary_vein_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["right_inferior_pulmonary_vein"], Tags["right_inferior_pulmonary_vein"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def right_inferior_pulmonary_vein_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["right_inferior_pulmonary_vein_border"], Tags["right_inferior_pulmonary_vein_border"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def right_superior_pulmonary_vein_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["right_superior_pulmonary_vein_border"], Tags["right_superior_pulmonary_vein_border"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def superior_vena_cava_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["superior_vena_cava"], Tags["superior_vena_cava"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def inferior_vena_cava_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["inferior_vena_cava"], Tags["inferior_vena_cava"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    def superior_vena_cava_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["superior_vena_cava_border"], Tags["superior_vena_cava_border"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def inferior_vena_cava_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["inferior_vena_cava"], Tags["inferior_vena_cava"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def inferior_vena_cava_border_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["inferior_vena_cava_border"], Tags["inferior_vena_cava_border"]),
            invert=False,
            scalars="elemTag",
            preference="cell",
        )

    @property
    def pericardium_mesh(self):
        return pv.wrap(self.mesh).threshold(
            (Tags["pericardium"], Tags["pericardium"]), invert=False, scalars="elemTag", preference="cell",
        )

    @property
    def lv_voxel(self):
        return compute_voxel(self.lv_wall_mesh + self.mv_mesh + self.av_mesh, fill_inside=True)

    @property
    def lv_wall_voxel(self):
        return compute_voxel(self.lv_wall_mesh)

    @property
    def rv_voxel(self):
        return compute_voxel(self.rv_wall_mesh + self.tv_mesh + self.pv_mesh)

    @property
    def la_voxel(self):
        raise NotImplementedError

    @property
    def ra_voxel(self):
        raise NotImplementedError

    @property
    def lv_volume(self):
        return self.lv_voxel["voxel"].sum() * np.prod(self.lv_voxel["spacing"]) * (0.1 / self.scale) ** 3
        # return (self.lv_voxel["voxel"].sum() * np.prod(self.lv_voxel["spacing"]) * 1e-3)

    @property
    def lv_mass(self):
        raise NotImplementedError

    def reset_plotter(self):
        self.plotter = pv.Plotter()
        # self.plotter.show_grid()

    def show(self):
        self.plotter.show()
        self.reset_plotter()

    def plot_point_direction(self, point, normal):
        self.plotter.add_mesh(pv.Arrow(point, normal, tip_length=10000, tip_radius=10,), color="red")

    # def __repr__(self):
    #     return f"Mesh: {self.mesh}"


def get_lv_lv_wall_la_voxel(mesh, target_shape):
    lv_wall_voxel_in_image_space = compute_voxel(mesh.lv_wall_mesh, target_shape=target_shape[0], fill_inside=False)
    mv_voxel_in_image_space = compute_voxel(
        mesh.mv_mesh, target_shape=target_shape[0], fill_inside=False, blur_threshold=0.25
    )
    lv_voxel_in_image_space = compute_voxel(
        mesh.lv_wall_mesh + mesh.mv_mesh + mesh.av_mesh, target_shape=target_shape[0], fill_inside=True,
    )
    la_mesh = (
        mesh.la_wall_mesh
        + mesh.mv_mesh
        + mesh.appendage_mesh
        + mesh.appendage_border_mesh
        + mesh.left_superior_pulmonary_vein_mesh
        + mesh.left_inferior_pulmonary_vein_mesh
        + mesh.left_superior_pulmonary_vein_border_mesh
        + mesh.right_inferior_pulmonary_vein_mesh
        + mesh.right_inferior_pulmonary_vein_border_mesh
        + mesh.right_superior_pulmonary_vein_mesh
        + mesh.right_inferior_pulmonary_vein_border_mesh
        + mesh.right_superior_pulmonary_vein_border_mesh
    )
    la_voxel_in_image_space = compute_voxel(
        la_mesh, target_shape=target_shape[0], fill_inside=True, blur_threshold=0.25,
    )
    la_wall_voxel_in_image_space = compute_voxel(
        la_mesh, target_shape=target_shape[0], fill_inside=False, blur_threshold=0.25,
    )

    lv_voxel_in_image_space["voxel"] += (mv_voxel_in_image_space["voxel"]).astype(np.uint8)
    lv_voxel_in_image_space["voxel"] = (lv_voxel_in_image_space["voxel"] > 0) * 1

    la_voxel_in_image_space["voxel"] += (la_wall_voxel_in_image_space["voxel"]).astype(np.uint8)
    la_voxel_in_image_space["voxel"] = (la_voxel_in_image_space["voxel"] > 0) * 1

    lv_lv_wall_la_voxel_in_image_space = np.zeros(lv_voxel_in_image_space["voxel"].shape, dtype=np.uint8)

    lv_lv_wall_la_voxel_in_image_space[lv_wall_voxel_in_image_space["voxel"] == 1] = 2
    lv_lv_wall_la_voxel_in_image_space[la_voxel_in_image_space["voxel"] == 1] = 3
    lv_lv_wall_la_voxel_in_image_space[lv_voxel_in_image_space["voxel"] == 1] = 1

    return {
        "voxel": lv_lv_wall_la_voxel_in_image_space,
        "spacing": lv_voxel_in_image_space["spacing"],
    }


def get_la_voxel(mesh, target_shape):
    la_mesh = (
        mesh.la_wall_mesh
        + mesh.mv_mesh
        + mesh.appendage_mesh
        + mesh.appendage_border_mesh
        + mesh.left_superior_pulmonary_vein_mesh
        + mesh.left_inferior_pulmonary_vein_mesh
        + mesh.left_superior_pulmonary_vein_border_mesh
        + mesh.right_inferior_pulmonary_vein_mesh
        + mesh.right_inferior_pulmonary_vein_border_mesh
        + mesh.right_superior_pulmonary_vein_mesh
        + mesh.right_inferior_pulmonary_vein_border_mesh
        + mesh.right_superior_pulmonary_vein_border_mesh
    )
    la_voxel_in_image_space = compute_voxel(
        la_mesh, target_shape=target_shape[0], fill_inside=True, blur_threshold=0.25,
    )
    la_wall_voxel_in_image_space = compute_voxel(
        la_mesh, target_shape=target_shape[0], fill_inside=False, blur_threshold=0.25,
    )
    mv_voxel = compute_voxel(mesh.mv_mesh, target_shape=target_shape[0], fill_inside=False, blur_threshold=0.25,)

    la_voxel_in_image_space["voxel"] += (la_wall_voxel_in_image_space["voxel"]).astype(np.uint8)
    la_voxel_in_image_space["voxel"] -= (mv_voxel["voxel"]).astype(np.uint8)
    la_voxel_in_image_space["voxel"] = (la_voxel_in_image_space["voxel"] > 0) * 1

    # la_voxel_in_image_space = np.zeros(la_voxel_in_image_space["voxel"].shape, dtype=np.uint8)
    #
    # la_voxel_in_image_space[la_voxel_in_image_space["voxel"] == 1] = 1

    return la_voxel_in_image_space


if __name__ == "__main__":
    import glob
    import os
    from natsort import natsorted
    import math
    from tqdm import tqdm

    from mesh.config import cfg

    all_data_paths = []
    for i in range(40):
        final_model_dir = os.path.join(
            "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/", r"Final_models_{}/".format(i + 1)
        )
        all_data_paths.extend(natsorted(glob.glob(final_model_dir + "*.vtk")))

    # for path in all_data_paths:
    # path = "/Users/jaeikjeon/Workspace/CODE/2023/echo4d/template_ssm/Final_models_01/Full_Heart_Mesh_5.vtk"
    # path = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned_a2c_plane_projected/Final_models_18+full_heart_mesh_016.vtk"
    # path = "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/Final_models_18+full_heart_mesh_016.vtk"
    # path = "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/Final_models_01+full_heart_mesh_001.vtk"
    # path = "/Users/jaeikjeon/Workspace/CODE/2023/echo4d/tmp/Final_models_01+full_heart_mesh_001.vtk"
    # path = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/Final_models_02+full_heart_mesh_009.vtk"
    path = "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned/Final_models_25+full_heart_mesh_020.vtk"

    # paths = glob.glob("/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned_a2c_plane_projected/*.vtk")
    # paths = glob.glob("/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned")
    # paths = glob.glob("/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/cleaned_a2c_plane_projected/*.vtk")
    view_list = ["A2CH", "A4CH", "PSAX mid", "PSAX basal", "PSAX apex"]
    voxel_space_view = "A2CH"

    version = "2023_12_23"
    base_save_path = f"/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/Synthetic_Echo/mask_from_mesh/{version}"

    mesh = Mesh(cfg, path, scale=1)
    fname = os.path.basename(path).split(".")[0]

    mesh.show()

    ##
    """
    Echo slices
    """
    # generate echo slices
    echo_slices = mesh.generate_echo_slices_per_view(view_list)

    for view in echo_slices:
        plt.imshow(echo_slices[view]["padded_echo_inhouse_style_mask"])
        plt.show()
    ##
    """
    Voxel Computations
    """
    # transform the mesh to the space defined by the echo slice corresponding to the [voxel_space_view].
    transformed_mesh = mesh.transform_mesh_to_slice_space(
        # mesh.lv_wall_mesh + mesh.mv_mesh + mesh.av_mesh,
        # mesh.lv_wall_mesh,
        mesh.mesh,
        echo_slices[voxel_space_view]["space_info"]["padded echo inhouse style mask bounds"],
        echo_slices[voxel_space_view]["space_info"],
        echo_slices[voxel_space_view]["padded_echo_inhouse_style_mask"],
        plot_flag=True,
    )
    transformed_mesh = Mesh(cfg, transformed_mesh, scale=1, origin=False)
    target_shape = echo_slices[voxel_space_view]["padded_echo_inhouse_style_mask"].shape
    lv_lv_wall_la_voxel = get_lv_lv_wall_la_voxel(transformed_mesh, target_shape)
    # plt.imshow(lv_lv_wall_la_voxel["voxel"][:, :, 40])
    # plt.show()
    # plt.imshow(echo_slices[voxel_space_view]["padded_echo_inhouse_style_mask"])
    # plt.show()

    ##

    ##
    # mv_voxel_in_image_space = compute_voxel(transformed_mesh.mv_mesh, target_shape=target_shape[0], fill_inside=False)
    # rv_wall_voxel_in_image_space = compute_voxel(
    #     transformed_mesh.rv_wall_mesh, target_shape=target_shape[0], fill_inside=False
    # )
    # rv_voxel_in_image_space = compute_voxel(
    #     transformed_mesh.lv_wall_mesh
    #     + transformed_mesh.rv_wall_mesh
    #     + transformed_mesh.tv_mesh
    #     + transformed_mesh.pv_mesh,
    #     target_shape=target_shape[0],
    #     fill_inside=True,
    #     target_chamber="rv",
    # )

    # plt.imshow(lv_voxel_in_image_space["voxel"][:, :, 40])
    # plt.show()
    # plt.imshow(lv_wall_voxel_in_image_space["voxel"][:, :, 40])
    # plt.show()
    # plt.imshow(mv_voxel_in_image_space["voxel"][:, :, 40])
    # plt.show()
    # plt.imshow(la_voxel_in_image_space["voxel"][:, :, 40])
    # plt.show()
    # plt.imshow(rv_wall_voxel_in_image_space["voxel"][:, 100, :])
    # plt.show()
    # plt.imshow(rv_voxel_in_image_space["voxel"][:, 100, :])
    # plt.show()
    # plt.imshow(echo_slices[voxel_space_view]["padded_echo_inhouse_style_mask"])
    # plt.show()

    """
    plot mesh
    """
    for view in echo_slices:
        echo_inhouse_style_mask = echo_slices[view]["echo_inhouse_style_mask"]
        # revert things back to the original space
        reverted_us_slice_arr_grid = mesh.revert_projected_array(
            echo_inhouse_style_mask,
            echo_slices[view]["space_info"]["echo inhouse style mask bounds"],
            echo_slices[view]["space_info"],
        )

        # mesh.plotter.add_mesh(reverted_us_slice_arr_grid, opacity=1, show_edges=True)

    # # mesh_handler.plot_cardiac_mesh()
    # mesh.plot_cardiac_mesh(style="wireframe")
    # mesh.show()

    ##
    data = {}
    data[fname] = {}
    voxel_space_view = "A2CH"
    for view in view_list:
        # single standard cardiac slice
        cardiac_slice, space_info = mesh.get_cardiac_slice(view, plot_flag=False, background=False)

        # multiple cardiac slices with rotations
        # multiple_slices = mesh_handler.get_multiple_cardiac_slices(view, space_info)

        # for idx, (cardiac_slice, space_info) in enumerate(multiple_slices):

        projected_slice_arr, space_info = mesh.get_numpy_slice_arr(cardiac_slice, space_info)

        inhouse_style_mask = mesh.inhouse_style_process(projected_slice_arr, view)

        echo_inhouse_style_mask, space_info = mesh.us_process(inhouse_style_mask, view, space_info)

        # revert things back to the original space
        reverted_slice_arr_grid = mesh.revert_projected_array(
            inhouse_style_mask, space_info["projected plane bounds"], space_info
        )
        reverted_us_slice_arr_grid = mesh.revert_projected_array(
            echo_inhouse_style_mask, space_info["echo inhouse style mask bounds"], space_info
        )

        # or can transform the mesh to the sliced array space
        transformed_mesh = mesh.transform_mesh_to_slice_space(
            mesh.lv_wall_mesh + mesh.mv_mesh + mesh.av_mesh,
            space_info["echo inhouse style mask bounds"],
            space_info,
            echo_inhouse_style_mask,
            plot_flag=False,
        )

        if view == voxel_space_view:
            # voxel_in_image_space = compute_voxel(transformed_mesh, pretext_img=echo_inhouse_style_mask)
            target_shape = echo_inhouse_style_mask.shape
            voxel_in_image_space = compute_voxel(transformed_mesh, target_shape=target_shape[0])
        data[fname][view] = {
            "projected_slice_arr": projected_slice_arr,
            "inhouse_style_mask": inhouse_style_mask,
            "echo_inhouse_style_mask": echo_inhouse_style_mask,
            "space_info": space_info,
        }

    data[fname][f"voxel_in_image_space"] = [voxel_in_image_space, voxel_space_view]
    ##
    def pad_echo_inhouse_style_masks(data_dict):
        """
        Pads all echo_inhouse_style_masks in the dataset to have the same dimensions, based on the largest width and height found.
        """

        def find_max_h_w(data_dict):
            max_height = 0
            max_width = 0
            for view, view_data in data_dict.items():
                if view == "voxel_in_image_space":  # Skip the voxel_in_image_space entry
                    continue
                mask = view_data["echo_inhouse_style_mask"]
                if mask.shape[0] > max_height:
                    max_height = mask.shape[0]
                if mask.shape[1] > max_width:
                    max_width = mask.shape[1]

            return max_height, max_width

        max_height, max_width = find_max_h_w(data_dict)

        for view, view_data in data_dict.items():
            if view == "voxel_in_image_space":  # Skip the voxel_in_image_space entry
                continue
            mask = view_data["echo_inhouse_style_mask"]
            pad_height = max_height - mask.shape[0]
            pad_width = max_width - mask.shape[1]
            pad_size = [pad_height // 2, pad_height - pad_height // 2, pad_width // 2, pad_width - pad_width // 2]
            padded_mask = np.pad(
                mask, [[pad_size[0], pad_size[1]], [pad_size[2], pad_size[3]]], mode="constant", constant_values=0,
            )

            view_data["padded_echo_inhouse_style_mask"] = padded_mask
            view_data["space_info"]["padded echo inhouse style mask bounds"] = padding_bounds(
                view_data["space_info"]["echo inhouse style mask bounds"], pad_size
            )

        return data_dict

        pad_echo_inhouse_style_masks(data[fname])

        # reverted_us_slice_arr_grid = mesh.revert_projected_array(
        #     padded_mask, view_data["space_info"]["padded projected plane bounds"], view_data["space_info"]
        # )
        # plt.imshow(padded_mask)
        # plt.show()

        # print(padded_mask.shape)
        # mesh.plotter.add_mesh(reverted_us_slice_arr_grid, opacity=1, show_edges=True)

    # # mesh_handler.plot_cardiac_mesh()
    mesh.plot_cardiac_mesh(style="wireframe")
    mesh.show()

    ##
    # for i in ramgelv_voxel:
    plt.imshow(lv_voxel["voxel"][:, :, 10])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 15])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 20])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 25])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 30])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 35])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 40])
    plt.show()
    plt.imshow(lv_voxel["voxel"][:, :, 45])
    plt.show()
    # resized_lv_voxel = zoom(lv_voxel["voxel"], (1 / 0.8, 1 / 0.8, 1 / 0.8))
    # # resized_lv_voxel = np.array(
    # #     [cv2.resize(lv_voxel["voxel"][i, :, :], (256, 256)) for i in range(lv_voxel["voxel"].shape[0])]
    # # )
    # # plt.imshow(resized_lv_voxel["voxel"][:, :, resized_lv_voxel["voxel"].shape[2] // 2])
    # plt.imshow(resized_lv_voxel[:, :, resized_lv_voxel.shape[2] // 2])
    # plt.show()
    print(mesh.mesh.bounds)
    print(mesh.lv_volume)

    ##
    # base_save_path = f"/mnt/NAS02_data1/jaeik/ECHO/DATA/Synthetic_Echo/mask_from_mesh/{version}"
    # voxel_shape = []
    # lv_volumes = []
    # for path in tqdm(paths):
    #     mesh_handler = MeshHandler(cfg, path)
    #     lv_voxel = mesh_handler.lv_voxel
    #     voxel_shape.append(lv_voxel["voxel"].shape)
    #     lv_volume = mesh_handler.lv_volume
    #     lv_volumes.append(lv_volume)
    # ##
    # plt.hist(lv_volumes, bins=100)
    # plt.xlabel("LV volume (mL)")
    # plt.ylabel("Frequency")
    # plt.title("LV volume distribution")
    #
    # plt.show()
    ##

    # base_save_path = f"/mnt/NAS02_data1/jaeik/ECHO/DATA/Synthetic_Echo/mask_from_mesh/{version}"
    # voxel_shape = []
    # lv_volumes = []
    # for path in tqdm(paths):
    #     mesh_handler = MeshHandler(cfg, path)
    #     lv_voxel = mesh_handler.lv_voxel
    #     voxel_shape.append(lv_voxel["voxel"].shape)
    #     lv_volume = mesh_handler.lv_volume
    #     lv_volumes.append(lv_volume)
    # ##
    # plt.hist(lv_volumes, bins=100)
    # plt.xlabel("LV volume (mL)")
    # plt.ylabel("Frequency")
    # plt.title("LV volume distribution")
    #
    # plt.show()

    ##

    #
    # # raise ValueError
    # ##
    # # for i in range(lv_voxel.shape[0]):
    # #     os.makedirs(f"./tmp/mesh_slice", exist_ok=True)
    # #     plt.imsave(f"./tmp/mesh_slice/test{i}.png", lv_voxel[i, :, :])
    # # mesh_handler.get_key_cardiac_points()
    # # mesh_handler.translate_mesh_to_a2c_plane()

    ##
    # calculate voxel
    mesh2 = Mesh(cfg, transformed_mesh, origin=False)
    points = mesh2.mesh.points
    xyzmin = points.min(0)
    xyzmax = points.max(0)

    x_resol = int(abs(mesh2.mesh.bounds[1] - mesh2.mesh.bounds[0]))
    y_resol = int(abs(mesh2.mesh.bounds[3] - mesh2.mesh.bounds[2]))
    z_resol = int(abs(mesh2.mesh.bounds[5] - mesh2.mesh.bounds[4]))

    sizes = np.asarray([x_resol, y_resol, z_resol])
    for n, size in enumerate(sizes):
        margin = (((points.ptp(0)[n] // size) + 1) * size) - points.ptp(0)[n]
        xyzmin[n] -= margin / 2
        xyzmax[n] += margin / 2
        x_y_z[n] = ((xyzmax[n] - xyzmin[n]) / size).astype(int)

    ##
    df = pd.DataFrame(data=points, columns=["x", "y", "z"])
    cloud = PyntCloud(df)

    x_resol = int(abs(mesh2.mesh.bounds[1] - mesh2.mesh.bounds[0]))
    y_resol = int(abs(mesh2.mesh.bounds[3] - mesh2.mesh.bounds[2]))
    z_resol = int(abs(mesh2.mesh.bounds[5] - mesh2.mesh.bounds[4]))

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=x_resol, n_y=y_resol, n_z=z_resol, regular_bounding_box=False)
    voxelgrid = cloud.structures[voxelgrid_id]
    out_voxel_array = voxelgrid.get_feature_vector(mode="binary")
    spacing_info = voxelgrid.shape

    ##
    for view in view_list:
        # single standard cardiac slice
        cardiac_slice, space_info = mesh.get_cardiac_slice(view, plot_flag=False, background=False)

        # multiple cardiac slices with rotations
        # multiple_slices = mesh_handler.get_multiple_cardiac_slices(view, space_info)

        # for idx, (cardiac_slice, space_info) in enumerate(multiple_slices):

        projected_slice_arr, space_info = mesh.get_numpy_slice_arr(cardiac_slice, space_info)

        inhouse_style_mask = mesh.inhouse_style_process(projected_slice_arr, view)

        echo_inhouse_style_mask, space_info = mesh.us_process(inhouse_style_mask, view, space_info)
        plt.imshow(echo_inhouse_style_mask)
        plt.show()

        # revert things back to the original space
        reverted_slice_arr_grid = mesh.revert_projected_array(
            inhouse_style_mask, space_info["projected plane bounds"], space_info
        )
        reverted_us_slice_arr_grid = mesh.revert_projected_array(
            echo_inhouse_style_mask, space_info["echo inhouse style mask bounds"], space_info
        )

        # # or can transform the mesh to the sliced array space
        # transformed_mesh = mesh.transform_mesh_to_slice_space(space_info, echo_inhouse_style_mask, plot_flag=True)

        mesh_file_name = (path.split("/")[-2] + "+" + path.split("/")[-1]).split(".")[0]
        save_path = os.path.join(base_save_path, mesh_file_name, view)

        original_slice_save_path = os.path.join(save_path, "0_original_slice")
        inhouse_style_slice_save_path = os.path.join(save_path, "1_inhouse_style_slice")
        us_slice_save_path = os.path.join(save_path, "2_us_slice")
        os.makedirs(original_slice_save_path, exist_ok=True)
        os.makedirs(inhouse_style_slice_save_path, exist_ok=True)
        os.makedirs(us_slice_save_path, exist_ok=True)

        # raise ValueError

        # # save original slice
        # plt.imsave(
        #     os.path.join(original_slice_save_path, f"{idx}.png"), projected_slice_arr,
        # )
        #
        # # save inhouse style slice
        # plt.imsave(
        #     os.path.join(inhouse_style_slice_save_path, f"{idx}.png"), inhouse_style_mask,
        # )
        #
        # # save us slice
        # plt.imsave(
        #     os.path.join(us_slice_save_path, f"{idx}.png"), echo_inhouse_style_mask,
        # )

        mesh.plotter.add_mesh(reverted_slice_arr_grid, opacity=1, show_edges=True)
        mesh.plotter.add_mesh(reverted_us_slice_arr_grid, opacity=1, show_edges=True)

    # # mesh_handler.plot_cardiac_mesh()
    mesh.plot_cardiac_mesh(style="wireframe")
    mesh.show()

    ##
    # mesh_handler = MeshHandler(cfg, path)
    # mesh_handler.get_key_cardiac_points()
    def rotate_to_xy_plane(mesh, normalized_v):
        """Calculate the rotation matrix to get an arbitrary plane to the xy plane, then apply it to that plane."""
        rot_mat = calc_rot_mat_to_xy_plane(normalized_v)
        return transform_vtk_rotation_matrix(mesh, np.ravel(rot_mat))

    def calc_rot_mat_to_xy_plane(normalized_v, check_determinant=False):
        """Find the 3D rotation matrix which will transform an arbitrary plane to be in the xy plane i.e. no z component."""

        def expand_rot_mat(rot_mat):
            full_rot_mat = np.zeros((4, 4))
            full_rot_mat[:3, :3] = rot_mat
            full_rot_mat[3, 3] = 1.0
            return full_rot_mat

        a, b, c = normalized_v

        square = a ** 2 + b ** 2 + c ** 2

        cos_theta = c / np.sqrt(square)
        sin_theta = np.sqrt((a ** 2 + b ** 2) / square)

        u1 = b / np.sqrt(a ** 2 + b ** 2)
        u2 = -a / np.sqrt(a ** 2 + b ** 2)

        r_mat = np.array(
            [
                [cos_theta + u1 ** 2 * (1 - cos_theta), u1 * u2 * (1 - cos_theta), u2 * sin_theta],
                [u1 * u2 * (1 - cos_theta), cos_theta + u2 ** 2 * (1 - cos_theta), -u1 * sin_theta],
                [-u2 * sin_theta, u1 * sin_theta, cos_theta],
            ]
        )

        if check_determinant:
            print("det of R is: {0:.5f}".format(np.linalg.det(r_mat)))

        return expand_rot_mat(r_mat)

    def rotate_to_xz_plane(mesh, normalized_v):
        """Calculate the rotation matrix to get an arbitrary plane to the xz plane, then apply it to that plane."""
        rot_mat = calc_rot_mat_to_xz_plane(normalized_v)
        return transform_vtk_rotation_matrix(mesh, np.ravel(rot_mat))

    def calc_rot_mat_to_xz_plane(normalized_v, check_determinant=False):
        """Find the 3D rotation matrix which will transform an arbitrary plane to be in the xz plane i.e. no y component."""

        def expand_rot_mat(rot_mat):
            full_rot_mat = np.zeros((4, 4))
            full_rot_mat[:3, :3] = rot_mat
            full_rot_mat[3, 3] = 1.0
            return full_rot_mat

        a, b, c = normalized_v

        square = a ** 2 + b ** 2 + c ** 2

        cos_theta = b / np.sqrt(square)
        sin_theta = np.sqrt((a ** 2 + c ** 2) / square)

        u1 = c / np.sqrt(a ** 2 + c ** 2)
        u2 = -a / np.sqrt(a ** 2 + c ** 2)

        r_mat = np.array(
            [
                [cos_theta + u1 ** 2 * (1 - cos_theta), u1 * u2 * (1 - cos_theta), u2 * sin_theta],
                [u1 * u2 * (1 - cos_theta), cos_theta + u2 ** 2 * (1 - cos_theta), -u1 * sin_theta],
                [-u2 * sin_theta, u1 * sin_theta, cos_theta],
            ]
        )

        if check_determinant:
            print("det of R is: {0:.5f}".format(np.linalg.det(r_mat)))

        return expand_rot_mat(r_mat)

    a4ch_normal = mesh_handler.cardiac_out_points["A2CH"][0][0]
    apex_pts, mv_pts = mesh_handler.cardiac_out_points["A2CH"][1][1:]

    mesh = mesh_handler.mesh_origin.copy()

    mesh = rotate_to_xy_plane(mesh, a4ch_normal)
    mesh = pv.UnstructuredGrid(mesh)
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, opacity=1, show_edges=True)
    #
    # plotter.show_grid()
    # plotter.show()

    ##
    new_mesh_handler = MeshHandler(cfg, mesh)
    projected_key_pts = new_mesh_handler.get_key_cardiac_points(["A2CH"])
    print(projected_key_pts)
    apex_pts_xy = np.array(new_mesh_handler.cardiac_out_points["A2CH"][1][1][:2])
    mv_pts_xy = np.array(new_mesh_handler.cardiac_out_points["A2CH"][1][2][:2])

    # calculate angle between x axis and line from apex to mitral valve
    apex_mv_vec = mv_pts_xy - apex_pts_xy
    apex_mv_vec = apex_mv_vec / np.linalg.norm(apex_mv_vec)
    x_axis = np.array([1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    angle = np.arccos(np.dot(apex_mv_vec, x_axis))
    angle = np.rad2deg(angle)
    print("angle between x axis and line from apex to mitral valve: {0:.2f}".format(angle))

    rotated_mesh = mesh.rotate_z(90 - angle)
    print(rotated_mesh.center)
    new_mesh_handler = MeshHandler(cfg, rotated_mesh)
    rotated_projected_key_pts = new_mesh_handler.get_key_cardiac_points()
    print(rotated_projected_key_pts)

    apex_ptps_xy = np.array(new_mesh_handler.cardiac_out_points["A2CH"][1][1])

    # shift mesh so that apex is at origin
    print(rotated_mesh.center)
    shisfted_rotated_mesh = rotated_mesh.translate(
        np.array([-apex_ptps_xy[0], -apex_ptps_xy[1], -apex_ptps_xy[2]])
    ).copy()
    print(shisfted_rotated_mesh.center)
    new_new_mesh_handler = MeshHandler(cfg, shisfted_rotated_mesh)
    shifted_rotated_projected_key_pts = new_new_mesh_handler.get_key_cardiac_points()
    print(shifted_rotated_projected_key_pts)

    # rotate mesh to align y axis with line from apex to mitral valve
    ##
    # rotated_mesh = mesh.rotate_y(angle)
    rotated_mesh = pv.UnstructuredGrid(rotated_mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(rotated_mesh, opacity=1, show_edges=True)

    plotter.show_grid()
    plotter.show()
    ##

    new_mesh_handler = MeshHandler(cfg, rotated_mesh)
    new_mesh_handler.get_key_cardiac_points()

    ##
    loaded_mesh = mesh_handler.mesh_origin.threshold((1, 1), invert=False, scalars="elemTag", preference="cell")
    plotter = pv.Plotter()
    plotter.add_mesh(loaded_mesh, opacity=1, show_edges=True)
    plotter.show_grid()
    plotter.show()

    ##
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

    def image_padding(img, extra_size):
        img = np.pad(
            img, ((extra_size[0], extra_size[1]), (extra_size[2], extra_size[3])), "constant", constant_values=0,
        )
        return img

    def adjust_bounds_with_padding(original_bounds, padding):
        """
        Adjusts the bounds of a grid for a padded image, keeping the center consistent.

        Parameters:
        original_bounds (list): The original bounds as [x0, x1, y0, y1].
        padding (list): Padding added to each side as [pad_x0, pad_x1, pad_y0, pad_y1].

        Returns:
        list: New bounds as [new_x0, new_x1, new_y0, new_y1].
        """
        x0, x1, y0, y1 = original_bounds
        pad_y0, pad_y1, pad_x0, pad_x1 = padding

        # Calculate new bounds
        new_x0 = x0 - pad_x0
        new_x1 = x1 + pad_x1
        new_y0 = y0 - pad_y0
        new_y1 = y1 + pad_y1

        return [new_x0, new_x1, new_y0, new_y1]

    mask = inhouse_stye.copy()
    image_hw_max = np.max(mask.shape)
    if view != "lv_plax":
        extra_pad_size = np.array(image_hw_max / 8 + np.random.random(4) * image_hw_max / 6, dtype=int)
    else:
        extra_pad_size = np.array(np.random.random(4) * image_hw_max / 8, dtype=int)

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

    mask = image_padding(mask, pad_size)

    new_bounds = adjust_bounds_with_padding(projected_slice_bounds[:4], pad_size)

    # def get_grid_from_2d_arr(arr, label_name, bounds):
    #     xrng = np.arange(bounds[0], bounds[1], 1, dtype=np.float32,)[1:]
    #     yrng = np.arange(bounds[2], bounds[3], 1, dtype=np.float32,)[1:]
    #     zrng = np.arange(0, 1, 1, dtype=np.float32,)
    #     rgb_arr = arr.reshape(arr.shape[0] * arr.shape[1], order="F")
    #
    #     x, y, z = np.meshgrid(xrng, yrng, zrng, indexing="ij")
    #     resized_grid = pv.StructuredGrid(x, y, z)
    #     resized_grid.point_data[label_name] = rgb_arr
    #
    #     return resized_grid
    #
    # slice_grid = get_grid_from_2d_arr(mask.T, cfg.LABELS.LABEL_NAME, new_bounds)

    la_mask = mask == 1
    lv_mask = mask == 2
    lv_wall_mask = mask == 3

    inhouse_style_mask = np.zeros_like(mask)
    inhouse_style_mask[la_mask] = 1
    inhouse_style_mask[lv_mask] = 2
    inhouse_style_mask[lv_wall_mask] = 3
    padded_mask = inhouse_style_mask.copy()
    cone = generate_US_cones(1, inhouse_style_mask.shape[np.argmax(inhouse_style_mask.shape)])
    cone = next(cone)

    out_label_img = padded_mask * cone[0]
    # out_label_img *= cone  # clips labels that are outside the cone
    out_label_img[out_label_img > 0] += 1  # make space for new label to match already generated datasets
    out_label_img[(out_label_img == 0) & (cone[0] == 1.0)] = 1
    reverted_slice_arr_grid = mesh_handler.revert_projected_array(out_label_img, new_bounds, projected_center, normal)

    mesh_handler.plotter.add_mesh(reverted_slice_arr_grid, opacity=1, show_edges=True)
    mesh_handler.plot_cardiac_mesh()
    mesh_handler.show()

    # pad = extra_pad_size - mask.shape
    # if pad % 2 == 1:
    #     pad_x0 = 1 + extra_pad_size[0]
    #     pad_x1 = extra_pad_size[1]
    #     pad_y0 = 0 + extra_pad_size[2]
    #     pad_y1 = 0 + extra_pad_size[3]
    mask = image_padding(mask, extra_pad_size)
    new_bounds = adjust_bounds_with_padding(projected_slice_bounds[:4], extra_pad_size)
    new_bounds.extend(projected_slice_bounds[4:])

    reverted_slice_arr_grid = mesh_handler.revert_projected_array(mask, new_bounds, projected_center, normal)

    # pad_size = get_pad_size(mask.shape, np.max(mask.shape), axis=np.argmin(mask.shape), extra_pad_size=extra_pad_size)
    image = pad_to_size(mask, np.max(mask.shape), extra_pad_size, axis=np.argmin(mask.shape))
    # image = pad_to_size(image, np.max(image.shape), [0, 0, 0, 0], axis=np.argmin(image.shape))
    print(image.shape)
    # plt.imshow(image)
    # plt.show()
    la_mask = image == 1
    lv_mask = image == 2
    lv_wall_mask = image == 3

    inhouse_style_mask = np.zeros_like(image)
    inhouse_style_mask[la_mask] = 1
    inhouse_style_mask[lv_mask] = 2
    inhouse_style_mask[lv_wall_mask] = 3
    padded_mask = inhouse_style_mask.copy()
    cone = generate_US_cones(1, inhouse_style_mask.shape[np.argmax(inhouse_style_mask.shape)])
    cone = next(cone)

    out_label_img = padded_mask * cone[0]
    # out_label_img *= cone  # clips labels that are outside the cone
    out_label_img[out_label_img > 0] += 1  # make space for new label to match already generated datasets
    out_label_img[(out_label_img == 0) & (cone[0] == 1.0)] = 1
    ##
    # reverted_slice_arr_grid = mesh_handler.revert_projected_array(
    #     inhouse_stye, projected_slice_bounds, projected_center, normal
    # )
    mesh_handler.plotter.add_mesh(reverted_slice_arr_grid, opacity=1, show_edges=True)
    mesh_handler.plot_cardiac_mesh()
    mesh_handler.show()

    # plt.imshow(out_label_img)
    # plt.show()
    ##

    a2ch_slice_arr = a2ch_slice_arr.T

    # find_plane_bounds_in_mesh(a2ch_cardiac_slice.points, a2ch_cardiac_slice.faces, a2ch_normal)

    # flip the slzie in the y axis
    # a2ch_slice_arr = cv2.flip(a2ch_slice_arr, 1)
    # # rotate the slice to 90 degrees
    # a2ch_slice_arr = cv2.rotate(a2ch_slice_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # xrng = np.arange(
    #     -math.floor(a2ch_slice_arr.shape[0] / 2), math.ceil(a2ch_slice_arr.shape[0] / 2), 1, dtype=np.float32,
    # )
    # yrng = np.arange(
    #     -math.floor(a2ch_slice_arr.shape[1] / 2), math.ceil(a2ch_slice_arr.shape[1] / 2), 1, dtype=np.float32,
    # )
    transformed_a2c_bounds = a2c_transformed_slice.bounds
    xrng = np.arange(transformed_a2c_bounds[0], transformed_a2c_bounds[1], 1, dtype=np.float32,)[1:]
    yrng = np.arange(transformed_a2c_bounds[2], transformed_a2c_bounds[3], 1, dtype=np.float32,)[1:]
    zrng = np.arange(0, 1, 1, dtype=np.float32,)
    rgb_arr = a2ch_slice_arr.reshape(a2ch_slice_arr.shape[0] * a2ch_slice_arr.shape[1], order="F")

    x, y, z = np.meshgrid(xrng, yrng, zrng, indexing="ij")
    resized_grid = pv.StructuredGrid(x, y, z)
    resized_grid.point_data["elemTag"] = rgb_arr

    # resized_grid = resized_grid.translate(projected_origin)
    # translation_vector = projected_origin - np.mean(resized_grid.points, axis=0)
    # resized_grid = resized_grid.translate(translation_vector)
    translation_vector = projected_origin - np.mean(resized_grid.points, axis=0)
    resized_grid = resized_grid.translate([0, 0, translation_vector[2] + 0])

    # resized_grid = pv.wrap(rotate_plane_to_target_normal(resized_grid, a2ch_normal))
    resized_grid = pv.wrap(revert_plane_to_original_normal(resized_grid, a2ch_normal))
    # Calculate the current center of the plane
    current_center = np.mean(resized_grid.points, axis=0)

    # Calculate the translation vector

    # shift ot the a2ch plane origin

    # resized_grid = pv.wrap(rotate_plane_to_target_normal(resized_grid, [0, 0, 1], a2ch_normal))
    # mesh_handler.plot_point_direction(np.mean(a2ch_cardiac_slice.points, axis=0), a2ch_normal * 50)
    # mesh_handler.plot_point_direction(np.mean(a2ch_cardiac_slice.points, axis=0), orthogonal_vec * 50)
    # mesh_handler.plotter.add_mesh(resized_grid, opacity=0.5, show_edges=True)
    # resized_grid = resized_grid.translate(projected_origin)
    # translation_vector = projected_origin - np.mean(resized_grid.points, axis=0)
    # resized_grid = resized_grid.translate([0, 0, translation_vector[2] + 0])

    # mesh_handler.plotter.add_mesh(a2c_transformed_slice, opacity=0.5, show_edges=True)
    mesh_handler.plotter.add_mesh(resized_grid, opacity=1, show_edges=True)
    mesh_handler.show()

    raise ValueError
    ##
    # flip the slzie in the y axis
    # a2ch_slice_arr = cv2.flip(a2ch_slice_arr, 1)
    # # rotate the slice to 90 degrees
    # a2ch_slice_arr = cv2.rotate(a2ch_slice_arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    a2ch_slice_arr, projected_origin = mesh_handler.get_numpy_slice_arr(a2ch_cardiac_slice, a2ch_normal)
    transformed_a2c_bounds = a2c_transformed_slice.bounds

    a2ch_slice_arr = a2ch_slice_arr.T
    # resized_grid = pv.StructuredGrid(x, y, z)
    # xrng = np.arange(
    #     -math.ceil(a2ch_slice_arr.shape[0] / 2), math.floor(a2ch_slice_arr.shape[0] / 2), 1, dtype=np.float32,
    # )
    # yrng = np.arange(
    #     -math.ceil(a2ch_slice_arr.shape[1] / 2), math.floor(a2ch_slice_arr.shape[1] / 2), 1, dtype=np.float32,
    # )
    zrng = np.arange(0, 1, 1, dtype=np.float32,)

    # xrng = np.arange(int(transformed_a2c_bounds[0]), np.ceil(transformed_a2c_bounds[1]), 1, dtype=np.float32,)
    # yrng = np.arange(int(transformed_a2c_bounds[2]), np.ceil(transformed_a2c_bounds[3]), 1, dtype=np.float32,)
    xrng = np.arange(transformed_a2c_bounds[0], transformed_a2c_bounds[1], 1, dtype=np.float32,)[1:]
    yrng = np.arange(transformed_a2c_bounds[2], transformed_a2c_bounds[3], 1, dtype=np.float32,)[1:]
    rgb_arr = a2ch_slice_arr.reshape(a2ch_slice_arr.shape[0] * a2ch_slice_arr.shape[1], order="F")

    x, y, z = np.meshgrid(xrng, yrng, zrng, indexing="ij")
    resized_grid = pv.StructuredGrid(x, y, z)
    resized_grid.point_data["elemTag"] = rgb_arr
    resized_grid.point_data["elemTag"] = rgb_arr

    # resized_grid = resized_grid.translate(projected_origin)
    translation_vector = projected_origin - np.mean(resized_grid.points, axis=0)
    resized_grid = resized_grid.translate([0, 0, translation_vector[2] + 0])

    mesh_handler.plotter.add_mesh(a2c_transformed_slice, opacity=0.5, show_edges=True)
    mesh_handler.plotter.add_mesh(resized_grid, opacity=1, show_edges=True)
    mesh_handler.show()

    raise ValueError
    ##

    plt.imshow(a2ch_slice_arr)
    plt.show()

    a4ch_cardiac_slice, a4ch_normal, a4ch_plane_origin, a4ch_land_marks = mesh_handler.get_cardiac_slice(
        "A4CH", plot_flag=True
    )

    a3ch_cardiac_slice, a3ch_normal, a3ch_plane_origin, a3ch_land_marks = mesh_handler.get_cardiac_slice(
        "A3CH", plot_flag=True
    )

    (
        psax_av_level_cardiac_slice,
        psax_av_level_normal,
        psax_av_level_plane_origin,
        psax_av_level_land_marks,
    ) = mesh_handler.get_cardiac_slice("PSAX mid", plot_flag=True)

    (
        psax_basal_level_cardiac_slice,
        psax_basal_level_normal,
        psax_basal_level_plane_origin,
        psax_basal_level_land_marks,
    ) = mesh_handler.get_cardiac_slice("PSAX basal", plot_flag=True)

    (
        psax_apex_cardiac_slice,
        psax_apex_normal,
        psax_apex_plane_origin,
        psax_apex_land_marks,
    ) = mesh_handler.get_cardiac_slice("PSAX apex", plot_flag=True)

    # mesh_handler.show()

    # a2c_slice_arr = mesh_handler.get_numpy_slice_arr(cardiac_slice, normal)
    # plotter = pv.Plotter()
    # plotter.add_mesh(a2ch_slice)
    # plotter.show()
    raise ValueError

    ##
    if __name__ == "__main__":
        import glob
        import os

        from natsort import natsorted
        from Epix2vox_reconstruction.data_extraction.slicing.config import cfg

        # all_data_paths = natsorted(glob.glob(cfg.DATA_IN.DATA_FOLDER + "*.vtk"))
        # start = time.perf_counter()
        # case_names = [file.split(os.sep)[-1].split(".")[0] for file in all_data_paths]

        # case_names = [
        #     # "/Users/jaeikjeon/Workspace/CODE/2023/echo4d/template_ssm_resultsheart_seg/heart_render/heart/full_heart_mesh_001/full_heart_mesh_001.vtk"
        #     # "/mnt/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/Final_models_01/Full_Heart_Mesh_1.vtk"
        #     "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/Final_models_01/Full_Heart_Mesh_1.vtk"
        # ]
        all_data_paths = []
        for i in range(40):
            final_model_dir = os.path.join(
                "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/", r"Final_models_{}/".format(i + 1)
            )
            all_data_paths.extend(natsorted(glob.glob(final_model_dir + "*.vtk")))
        os.listdir("/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/heart_mesh_ssm/Final_models_01")
        # save_paths = [
        #     os.path.join(
        #         "/Users/jaeikjeon/Workspace/CODE/2023/echo4d/template_ssm_resultsheart_seg",
        #         "heart_seg",
        #         "heart_render",
        #         "heart",
        #         "Full_Heart_Mesh_1",
        #     )
        #     for case_name in case_names
        # ]
        save_paths = ["/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/Synthetic_Echo/mask_from_mesh"]
        base_save_path = "/Volumes/NAS02/NAS02_data1/jaeik/ECHO/DATA/Synthetic_Echo/mask_from_mesh_2023_12_14"
        for path in all_data_paths:
            # if not os.path.exists(path):
            #     os.makedirs(path)

            # with Pool(1) as pool:
            # pool.starmap(
            save_path = os.path.join(base_save_path, path.split("/")[-2] + "+" + path.split("/")[-1].split(".")[0])
            os.makedirs(save_path, exist_ok=True)
            print(save_path)
            run_slice_extraction(cfg, path, save_path, True)
            # )

        # finish = time.perf_counter()
    ##
    # load npz file
    npz_file = np.load(
        "/Users/jaeikjeon/Workspace/CODE/2023/echo4d/template_ssm_resultsheart_seg/heart_render/heart/full_heart_mesh_001/full_heart_mesh_001.npz"
    )
