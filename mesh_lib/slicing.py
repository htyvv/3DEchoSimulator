import os

import numpy as np
import pyvista as pv
import vtkmodules.all as vtk
from scipy.spatial import distance
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import cv2
import mesh_lib.math_utils as maths_utils
import imutils
import pyvista as pv

# pv.start_xvfb()

from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkCommonTransforms import vtkTransform


def get_grid_from_2d_arr(arr, label_name, bounds):
    xrng = np.arange(bounds[0], bounds[1], 1, dtype=np.float32,)[1:]
    yrng = np.arange(bounds[2], bounds[3], 1, dtype=np.float32,)[1:]
    zrng = np.arange(0, 1, 1, dtype=np.float32,)
    rgb_arr = arr.reshape(arr.shape[0] * arr.shape[1], order="F")

    x, y, z = np.meshgrid(xrng, yrng, zrng, indexing="ij")
    resized_grid = pv.StructuredGrid(x, y, z)
    resized_grid.point_data[label_name] = rgb_arr

    return resized_grid


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


def transform_vtk_rotation_matrix(surface, matrix):
    """Apply rotation matrix to a vtk mesh."""
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surface)
    transformFilter.Update()
    return transformFilter.GetOutput()


def revert_plane_to_original_normal(mesh, original_normal):
    rot_mat = calc_rot_mat_to_xy_plane(original_normal)
    rot_mat = maths_utils.invert_rot_mat_to_xy_plane(rot_mat)
    return transform_vtk_rotation_matrix(mesh, np.ravel(rot_mat))


def translate_plane_to_point(plane, point):
    translation_vector = point - np.mean(plane.points, axis=0)
    plane = plane.translate([0, 0, translation_vector[2] + 0])
    return plane


def save_plane_img(cfg, transformed_slice, save_loc):
    """Saves the extracted plane image.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        transformed_slice (pyvista.core.pointset.UnstructuredGrid): Extracted slice that is transformed to xy plane.
        save_loc (str): Location of where to save image to.

    Returns:

    """

    """
        (TRUE)
        resampler = vtk.vtkResampleToImage()
        resampler.AddInputDataObject(transformed_slice)
        bounds = np.array(list(transformed_slice.bounds))

        resampler.SetSamplingBounds(*bounds[:5], 1.0)
        x_size = int(bounds[1] - bounds[0])
        y_size = int(bounds[3] - bounds[2])
        print(x_size, y_size)
        resampler.SetSamplingDimensions(x_size, y_size, 1)
        resampler.Update()

        img_as_array = vtk_to_numpy(resampler.GetOutput().GetPointData().GetArray(cfg.LABELS.LABEL_NAME))
        print(img_as_array.shape)
        # img_as_array = img_as_array.reshape((int(np.sqrt(img_as_array.shape[0])), int(np.sqrt(img_as_array.shape[0]))))
        img_as_array = img_as_array.reshape((y_size, x_size))

        =========================================
        resampler.SetSamplingBounds(*bounds[:5], 1.01)
        resampler.SetSamplingDimensions(256, 256, 1)
        resampler.Update()

        img_as_array = vtk_to_numpy(resampler.GetOutput().GetPointData().GetArray(cfg.LABELS.LABEL_NAME))
        img_as_array = img_as_array.reshape((int(np.sqrt(img_as_array.shape[0])), int(np.sqrt(img_as_array.shape[0]))))
        print(np.unique(img_as_array))

        =========================================

        resampler.SetSamplingBounds(*bounds[:5], 1.01)
        x_size = int(bounds[1] - bounds[0])
        y_size = int(bounds[3] - bounds[2])
        print(x_size, y_size)
        resampler.SetSamplingDimensions(x_size, y_size, 1)
        resampler.Update()

        img_as_array = vtk_to_numpy(resampler.GetOutput().GetPointData().GetArray(cfg.LABELS.LABEL_NAME))
        print(img_as_array.shape)
        # img_as_array = img_as_array.reshape((int(np.sqrt(img_as_array.shape[0])), int(np.sqrt(img_as_array.shape[0]))))
        img_as_array = img_as_array.reshape((x_size, y_size))
    """
    # plt.imshow(img_as_array)
    # plt.show()
    transformed_slice = pv.wrap(transformed_slice)

    transformed_slice.plot(
        cpos="xy",
        off_screen=True,
        show_axes=False,
        window_size=cfg.DATA_OUT.SAVE_IMG_RESOLUTION,
        background=cfg.DATA_OUT.SAVE_BCKGD_CLR,
        anti_aliasing=False,
        screenshot=save_loc,
        scalars=cfg.LABELS.LABEL_NAME,
        show_scalar_bar=False,
    )
    return


def save_plane_numpy(cfg, transformed_slice, save_loc, view_name):
    resampler = vtk.vtkResampleToImage()
    resampler.AddInputDataObject(transformed_slice)
    bounds = np.array(list(transformed_slice.bounds))

    resampler.SetSamplingBounds(*bounds[:5], 1.0)
    x_size = int(bounds[1] - bounds[0])
    y_size = int(bounds[3] - bounds[2])
    resampler.SetSamplingDimensions(x_size, y_size, 1)
    resampler.Update()

    img_as_array = vtk_to_numpy(resampler.GetOutput().GetPointData().GetArray(cfg.LABELS.LABEL_NAME))
    img_as_array = img_as_array.reshape((y_size, x_size))

    (h, w) = img_as_array.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    if view_name == "a4c":
        M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
        img_as_array = cv2.warpAffine(img_as_array, M, (w, h))
    elif view_name == "a3c":
        img_as_array = imutils.rotate_bound(img_as_array, 90)
        # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
        # img_as_array = cv2.warpAffine(img_as_array, M, (h, w))
    elif view_name == "psax_aortic":
        img_as_array = imutils.rotate_bound(img_as_array, 135)
        # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
        # img_as_array = cv2.warpAffine(img_as_array, M, (h, w))
    np.save(save_loc + ".npy", img_as_array)

    plt.imsave(save_loc + ".png", img_as_array)

    return


def get_plane_numpy(cfg, transformed_slice):
    resampler = vtk.vtkResampleToImage()
    # Explicitly tell the resampler to process the 'elemTag' cell data array.
    # This is crucial for ensuring the scalars are interpolated to the output image.
    resampler.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, cfg.LABELS.LABEL_NAME
    )
    resampler.AddInputDataObject(transformed_slice)
    bounds = np.array(list(transformed_slice.bounds))

    resampler.SetSamplingBounds(*bounds[:5], 1.0)
    x_size = int(bounds[1] - bounds[0])
    y_size = int(bounds[3] - bounds[2])
    # Ensure dimensions are at least 1 to avoid VTK errors
    if x_size < 1: x_size = 1
    if y_size < 1: y_size = 1
    resampler.SetSamplingDimensions(x_size, y_size, 1)
    resampler.Update()

    output_image = resampler.GetOutput()
    if not output_image:
        return np.zeros((y_size, x_size), dtype=np.uint8)

    img_vtk_array = output_image.GetPointData().GetArray(cfg.LABELS.LABEL_NAME)
    if not img_vtk_array:
        # This can happen if the input slice has no cells with the specified array.
        return np.zeros((y_size, x_size), dtype=np.uint8)

    img_as_array = vtk_to_numpy(img_vtk_array)
    img_as_array = img_as_array.reshape((y_size, x_size))

    return img_as_array.astype(np.uint8)


def align_slice(mesh, a, b, c, preferred_direction=None):
    """ align a slice to the given plane given 3 landmarks. If preferred_direction is given then this function
    will flip the norm to choose the one that was closest to the preferred_direction """

    def _choose_closest(x, ys):
        "choose the y that is closest to x (by norm) "
        min_dist = None
        chosen_y = None
        assert len(ys) > 1, "must give at least one y to choose closest"
        for y in ys:
            dist = np.linalg.norm(x - y)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                chosen_y = y
        return chosen_y

    def calculate_plane_normal(a, b, c):
        """
        :param a: 3D point
        :param b: 3D point
        :param c: 3D point
        :return: Vector normal to a plane which crosses the abc points.
        """
        x = np.cross(b - a, b - c)
        return x / np.linalg.norm(x)

    def calculate_rotation(reference_vector, target_vector):
        """
        Calculates the rotation matrix which rotates the object to align the target vector direction to reference
        vector direction. Assumes that both vectors are anchored at the beginning of the coordinate system
        :param reference_vector: Vector with referential direction. The rotation matrix will align the target_vector's
        direction to this one.
        :param target_vector:  Vector pointing to a  structure corresponding to the referential vector.
        :return: 3x3 rotation matrix (rot), where [rot @ target_vector = reference_vector] in terms of direction.
        """

        unit_reference_vector = reference_vector / np.linalg.norm(reference_vector)
        unit_target_vector = target_vector / np.linalg.norm(target_vector)
        c = unit_target_vector @ unit_reference_vector
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)
        else:
            v = np.cross(unit_target_vector, unit_reference_vector)
            vx = np.array(([0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]))
            vx2 = vx @ vx
            return np.eye(3) + vx + vx2 / (1 + c)

    def translate(mesh, rotation_matrix, translation_vector):
        translate = vtk.vtkTransform()
        translation_matrix = np.eye(4)
        translation_matrix[:-1, :-1] = rotation_matrix
        translation_matrix[:-1, -1] = translation_vector

        translate.SetMatrix(translation_matrix.ravel())
        transformer = vtk.vtkTransformFilter()
        # transformer.SetInputConnection(mesh.GetOutputPort())
        transformer.SetInputData(mesh)
        transformer.SetTransform(translate)
        transformer.Update()
        mesh = transformer
        # center_of_heart = self.get_center(self.mesh)
        # Invalidate landmarks after translate
        # self._landmarks = dict()
        return mesh

    def rotate(mesh, alpha=0, beta=0, gamma=0, rotation_matrix=None):
        rotate = vtk.vtkTransform()
        if rotation_matrix is not None:
            translation_matrix = np.eye(4)
            translation_matrix[:-1, :-1] = rotation_matrix
            rotate.SetMatrix(translation_matrix.ravel())
        else:
            rotate.Identity()
            rotate.RotateX(alpha)
            rotate.RotateY(beta)
            rotate.RotateZ(gamma)
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(mesh.GetOutputPort())
        # transformer.SetInputData(mesh)
        transformer.SetTransform(rotate)
        transformer.Update()
        mesh = transformer

        return mesh
        # self.center_of_heart = self.get_center(self.mesh)

    center = np.mean((a, b, c), axis=0)
    mesh = translate(mesh, rotation_matrix=np.eye(3), translation_vector=-center)

    a2, b2, c2 = [x - center for x in [a, b, c]]
    _normal = calculate_plane_normal(a2, b2, c2)
    if preferred_direction is not None:
        reverse_normal = _normal * -1
        _normal = _choose_closest(preferred_direction, [_normal, reverse_normal])
    rot1 = calculate_rotation(np.array([0, 0, 1]), _normal)
    a3, b3, c3 = [rot1 @ x for x in [a2, b2, c2]]
    rot2 = calculate_rotation(np.array([0, 1, 0]), b3 / np.linalg.norm(b3))
    rot3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    rot = rot3 @ rot2 @ rot1
    mesh = rotate(mesh, rotation_matrix=rot)
    # Invalidate landmarks after align
    # self._landmarks = dict()
    return mesh


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


def prepare_meshes(cfg, mesh_path):
    """Prepares the full cardiac mesh and subsampled mesh for further processing. This entails 1) Loading mesh file
    2) Removing the Aorta 3) Translating mesh to origin and subsampling.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        mesh_path (str): Location of binary mesh file to load from.

    Returns:
        Tuple[pyvista.core.pointset.UnstructuredGrid, pyvista.core.pointset.UnstructuredGrid]: Translated full
        resolution mesh and subsampled mesh.
    """
    print("Loading mesh")
    case_mesh = pv.get_reader(mesh_path).read()

    # case_mesh = pv.wrap(case_mesh).threshold(
    #     (cfg.LABELS.AORTA - 1, cfg.LABELS.AORTA + 1), invert=True, scalars=cfg.LABELS.LABEL_NAME, preference="cell"
    # )
    print("Translating mesh to origin and subsampling")
    origin_centred_mesh = maths_utils.translate_mesh_to_origin(case_mesh)
    low_res_mesh = subsample_mesh(cfg, origin_centred_mesh)
    return origin_centred_mesh, low_res_mesh


def slice_extraction(in_mesh, origin, normal, land_marks):
    # create a plane to cut (xz normal=(1,0,0);XY =(0,0,1),YZ =(0,1,0)
    in_mesh = align_slice(in_mesh, land_marks[0], land_marks[1], land_marks[2])
    # in_mesh.plot()

    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    # create cutter
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputConnection(in_mesh.GetOutputPort())
    # cutter.SetInputData(in_mesh)
    cutter.Update()
    return cutter


# def slice_with_plane(in_mesh, plane):
#     cutter = vtk.vtkCutter()
#     cutter.SetCutFunction(plane)
#     cutter.SetInputData(in_mesh)
#     cutter.Update()
#     return cutter.GetOutput()


# def slice_with_plane(in_mesh, origin, normal):
#     """ Slices a surface mesh with a given origin and normal that defines a slicing plane.
#
#     Args:
#         in_mesh (obj): Mesh for slicing.
#         origin (list): Origin of plane to perform slicing.
#         normal (list): Normal of plane to perform slicing.
#
#     Returns:
#         cutter.GetOutput(): Extracted slice object
#
#     """
#     ##
#     # Now you can use this plane in PyVista, for example, to slice a mesh
#     # Example: slicing a PyVista mesh
#
#     import pyvista as pv
#
#     plotter = pv.Plotter()
#     axis = (1, 0, 0)  # Rotation axis (here, y-axis)
#
#     # calculate vector perpendicular to normal
#     v = np.cross(axis, normal)
#     # for axis in [v]:
#     for axis in [(0, 0, 1), (1, 0, 0), (0, 1, 0), v]:
#         for angle in range(-24, 24, 6):
#             # Define the rotation - angle and axis
#
#             plane = vtk.vtkPlane()
#             plane.SetOrigin(origin)
#             plane.SetNormal(normal)
#
#             """
#             add augmentation to the plane
#             (1) rotate the plane
#             (2) translate the plane
#             """
#             # Create a vtkTransform
#             transform = vtkTransform()
#
#             # Apply the rotation to the transform
#             transform.RotateWXYZ(angle, axis)
#
#             # Transform the plane's normal and origin
#             transformed_normal = transform.TransformNormal(plane.GetNormal())
#             transformed_origin = transform.TransformPoint(plane.GetOrigin())
#
#             # mesh = pv.Sphere()
#             sliced_mesh = in_mesh.slice(normal=transformed_normal, origin=transformed_origin)
#
#             plotter.add_mesh(sliced_mesh)
#     color = "white"
#     smooth_shading = True
#     opacity = 0.15
#
#     plotter.add_mesh(in_mesh, color=color, smooth_shading=smooth_shading, opacity=opacity)
#     plotter.show_grid()
#     plotter.show()
#     # sliced_mesh.plot()
#     ##
#     cutter = vtk.vtkCutter()
#     cutter.SetCutFunction(plane)
#     cutter.SetInputData(in_mesh)
#     cutter.Update()
#     return cutter.GetOutput()


def slice_with_plane(in_mesh, origin, normal):
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(in_mesh)
    cutter.Update()
    return cutter.GetOutput()


def find_lv_apex(cfg, subsampled_mesh):
    """Function to find an intial rough point for the left ventricular apex.

    Args:
        cfg (easydict.EasyDict): configuration file.
        subsampled_mesh (pyvista.core.pointset.UnstructuredGrid): Low resolution mesh to find initial LV apex from.

    Returns:
        Tuple[pyvista.core.pyvista_ndarray.pyvista_ndarray, numpy.ndarray]: Initial LV apex coordinates and valid points
        found in the left ventricle for performing the more computationally expensive ray trace LV apex search
    """
    mitral_valve_centroid = calc_label_com(cfg, subsampled_mesh, cfg.LABELS.MITRAL_VALVE)
    lv_points = pv.wrap(cell_threshold(cfg, subsampled_mesh, start=cfg.LABELS.LV, end=cfg.LABELS.LV)).points
    """
    np.atleast_2d(mitral_valve_centroid): shape (1, 3)
    lv_points: shape (n points, 3)
    norm_dist: shape (n points, 1)
    """
    norm_dist = distance.cdist(np.atleast_2d(mitral_valve_centroid), lv_points, "euclidean").T
    max_indice = np.argmax(norm_dist)

    apex_coord = lv_points[max_indice, :]
    lv_length = np.linalg.norm([apex_coord, mitral_valve_centroid])
    thresholded_points = []
    for ii in range(len(norm_dist)):
        if norm_dist[ii] > lv_length * cfg.PARAMETERS.THRESHOLD_PERCENTAGE:
            thresholded_points += [lv_points[ii, :]]

    return apex_coord, np.squeeze(thresholded_points)


def find_lv_apex_raytrace(cfg, in_mesh, subsampled_mesh):
    """More accurate method for finding the left ventricular apex using a ray tracing method. An initial fast but
    inaccurate LV apex location is found and from this point the left ventricle is thresholded by a defined percentage
    to avoid wasted computation time, as this is an expensive task. Thresholding also adds robustness as it reduces
    the possibility of finding thin points near base of the left ventricle.

    Args:
        cfg (easydict.EasyDict): configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire cardiac mesh to find LV apex from.
        subsampled_mesh (pyvista.core.pointset.UnstructuredGrid): Low resolution mesh to find initial LV apex from.

    Returns:
        lv_apex_coords (numpy.ndarray): Accurate coordinates of the left ventricular apex.
    """
    lv_mesh = pv.wrap(in_mesh).threshold(
        (cfg.LABELS.LV, cfg.LABELS.LV), invert=False, scalars=cfg.LABELS.LABEL_NAME, preference="cell"
    )

    mitral_valve_centroid = calc_label_com(cfg, subsampled_mesh, cfg.LABELS.MITRAL_VALVE)

    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(in_mesh.extract_surface())
    obb_tree.BuildLocator()
    points_intersection = vtk.vtkPoints()

    __, threshold_points = find_lv_apex(cfg, subsampled_mesh)  # find fast but inaccurate lv location as starting point

    points_of_intersection = []
    for ii in range(len(threshold_points)):
        code = obb_tree.IntersectWithLine(
            mitral_valve_centroid, threshold_points[ii], points_intersection, None
        )  # looks like it isn't called but it is important and must be.

        points_vtk_intersection_data = points_intersection.GetData()
        num_points_intersection = points_vtk_intersection_data.GetNumberOfTuples()

        if num_points_intersection == 2:  # this means the ray has gone through both endo and epicardium
            for idx in range(num_points_intersection):
                _tup0 = points_vtk_intersection_data.GetTuple3(0)
                _tup1 = points_vtk_intersection_data.GetTuple3(1)
                points_of_intersection.append([_tup0, _tup1])

    points_of_intersection = np.squeeze(points_of_intersection)
    thinnest_point = np.argmax(
        np.linalg.norm([points_of_intersection[:, 0, :] - points_of_intersection[:, 1, :]], axis=2)
    )
    lv_apex_coords = points_of_intersection[thinnest_point, 1, :]
    return lv_apex_coords


def calc_label_com(cfg, in_mesh, label):
    """Calculates the centre of mass of a specific label within a mesh containing multiple labels.
    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire cardiac mesh.
        label (int or list): Value of the label within the mesh.

    Returns:
        numpy.ndarray: Centre of mass of selected label.
    """
    if isinstance(label, int):
        return pv.wrap(cell_threshold(cfg, in_mesh, start=label, end=label)).center_of_mass()
    else:
        return [
            pv.wrap(cell_threshold(cfg, in_mesh, start=label_val, end=label_val)).center_of_mass()
            for label_val in label
        ]


def plot_mesh_w_slice(cfg, cardiac_slice, normal, mesh_origin, bounds):
    """ Function to visualize how each standard view actually slices through the heart in 3D.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        transformed_slice (pyvista.core.pointset.UnstructuredGrid): cardiac slice for visualization.
        transformed_mesh (pyvista.core.pointset.UnstructuredGrid): Full mesh for visualization.
    """
    # pv_cardiac_slice = pv.wrap(cardiac_slice)

    true_plane = pv.Plane(
        center=np.mean(cardiac_slice.points - 0.1, axis=0),
        direction=normal,
        i_size=(bounds[1] - bounds[0]),
        j_size=bounds[3] - bounds[2],
    )

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_origin, style="wireframe", color="white", smooth_shading=True, opacity=0.15)  # Cardiac mesh
    plotter.add_mesh(
        cardiac_slice, scalars=cfg.LABELS.LABEL_NAME, smooth_shading=True, show_scalar_bar=False,  # Extracted slice
    )
    plotter.add_mesh(true_plane, color=cfg.DATA_OUT.SAVE_BCKGD_CLR, smooth_shading=True)  # Blank plane

    plotter.show_grid()
    plotter.show()
    return


def cell_threshold(cfg, in_mesh, start, end):
    """Perform thresholding on individual cell values of a mesh.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire cardiac mesh.
        start (int): Starting value to perform thresholding from.
        end (int): Ending value to perform thresholding to.

    Returns:
        vtkmodules.vtkCommonDataModel.vtkPolyData: Cell thresholded mesh.
    """
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(in_mesh)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, cfg.LABELS.LABEL_NAME)
    threshold.SetLowerThreshold(start)
    threshold.SetUpperThreshold(end)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    # threshold.ThresholdBetween(start, end)
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()


def vtk_cell_field_to_numpy(vtk_variable, array_name):
    return vtk_to_numpy(vtk_variable.GetCellData().GetArray(array_name))


def get_2ch(cfg, subsampled_mesh, lv_apex):
    """Calculates Apical 2 Chamber view coordinates for slicing."""
    rv_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.RV)
    mv_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.MITRAL_VALVE)

    __, out_pnts = maths_utils.pnt2line(tuple(rv_points), tuple(lv_apex), tuple(mv_points))

    out_vec = np.squeeze(rv_points - np.array(out_pnts))

    return [out_vec, np.squeeze(out_pnts)], [tuple(out_pnts), tuple(lv_apex), tuple(mv_points)]


def get_rv_inflow(cfg, subsampled_mesh, right_atrium_mesh_pnts):
    """Calculates Right ventricle inflow view coordinates for slicing."""
    rv_pv_pnts = calc_label_com(cfg, subsampled_mesh, label=[cfg.LABELS.RV, cfg.LABELS.PULMONARY_VALVE])
    rv_inflow_pnts = np.vstack((rv_pv_pnts, right_atrium_mesh_pnts))
    return maths_utils.plane_equation_calc(rv_inflow_pnts), np.mean(rv_inflow_pnts, axis=0)


def get_lv_plax(cfg, subsampled_mesh, aortic_valve_mesh_pnts):
    """Calculates left ventricle parasternal long axis view coordinates for slicing."""
    lv_plax_pnts = np.vstack(
        (calc_label_com(cfg, subsampled_mesh, label=[cfg.LABELS.LV, cfg.LABELS.MITRAL_VALVE]), aortic_valve_mesh_pnts)
    )
    return maths_utils.plane_equation_calc(lv_plax_pnts), np.mean(lv_plax_pnts, axis=0)


def get_3ch(cfg, subsampled_mesh, lv_apex, aortic_valve_mesh_pnts):
    """Calculates left ventricle parasternal long axis view coordinates for slicing."""
    mv_com = calc_label_com(cfg, subsampled_mesh, label=[cfg.LABELS.MITRAL_VALVE])
    lv_plax_pnts = np.vstack((mv_com, lv_apex, aortic_valve_mesh_pnts))
    return (
        # [maths_utils.plane_equation_calc(lv_plax_pnts), np.mean(lv_plax_pnts, axis=0)],
        [-maths_utils.plane_equation_calc(lv_plax_pnts), np.mean(lv_plax_pnts, axis=0)],
        [tuple(mv_com[0]), tuple(lv_apex), tuple(aortic_valve_mesh_pnts)],
    )


def get_psax_aortic(left_atrium_mesh_pnts, right_atrium_mesh_pnts, aortic_valve_mesh_pnts):
    """Calculates aortic valve level parasternal short axis view coordinates for slicing."""
    psax_aortic_pnts = np.vstack((left_atrium_mesh_pnts, right_atrium_mesh_pnts, aortic_valve_mesh_pnts))
    return maths_utils.plane_equation_calc(psax_aortic_pnts), np.mean(psax_aortic_pnts, axis=0)


def get_psax_mv(vert_vec, heart_com, lv_apex):
    """Calculates mitral valve level parasternal short axis view coordinates for slicing."""
    psax_mv_normal = maths_utils.find_plane_from_normal(vert_vec, (1 - 0.1) * heart_com + 0.1 * lv_apex)
    return tuple([psax_mv_normal, np.mean(np.squeeze(maths_utils.find_points_on_plane(psax_mv_normal)), axis=0)])


def get_psax_pm(vert_vec, heart_com, lv_apex):
    """Calculates papillary muscle level parasternal short axis view coordinates for slicing."""
    psax_pm_normal = maths_utils.find_plane_from_normal(vert_vec, (1 - 0.4) * heart_com + 0.4 * lv_apex)
    return tuple([psax_pm_normal, np.mean(np.squeeze(maths_utils.find_points_on_plane(psax_pm_normal)), axis=0)])


def get_psax_lower(vert_vec, heart_com, lv_apex):
    """Calculates lower level parasternal short axis view coordinates for slicing."""
    psax_lower_normal = maths_utils.find_plane_from_normal(vert_vec, (1 - 0.7) * heart_com + 0.7 * lv_apex)
    return tuple([psax_lower_normal, np.mean(np.squeeze(maths_utils.find_points_on_plane(psax_lower_normal)), axis=0)])


def get_a4c(left_atrium_mesh_pnts, right_atrium_mesh_pnts, lv_apex):
    """Calculates Apical 4 Chamber view coordinates for slicing."""
    a4c_pnts = np.vstack((left_atrium_mesh_pnts, right_atrium_mesh_pnts, lv_apex))
    return maths_utils.plane_equation_calc(a4c_pnts), np.mean(a4c_pnts, axis=0)


def get_a5c(left_atrium_mesh_pnts, aortic_valve_mesh_pnts, lv_apex):
    """Calculates Apical 5 Chamber view coordinates for slicing."""
    a5c_pnts = np.vstack((left_atrium_mesh_pnts, aortic_valve_mesh_pnts, lv_apex))
    # return -maths_utils.plane_equation_calc(a5c_pnts), np.mean(a5c_pnts, axis=0)
    return maths_utils.plane_equation_calc(a5c_pnts), np.mean(a5c_pnts, axis=0)


def get_plane(origin, normal):
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    return plane


def rotate_plane(normal, plane, angle, axis):
    # Create a vtkTransform
    transform = vtkTransform()

    # Apply the rotation to the transform
    transform.RotateWXYZ(angle, axis)

    # Transform the plane's normal and origin
    normal = transform.TransformNormal(normal)
    plane = transform.TransformPoint(plane)

    return normal, plane


def get_cardiac_images(
    cfg, in_mesh, subsampled_mesh, selected_view=["PLAX", "PSAX basal", "PSAX mid", "PSAX apex", "A4CH", "A2CH", "A3CH"]
):
    """Function to calculate commonly used landmarks for slicing and calling functions for all the selected views.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire full resolution cardiac mesh.
        subsampled_mesh (pyvista.core.pointset.UnstructuredGrid):  Low resolution mesh.

    Returns:
        returned_points4imgs (list): coordinate information for all the selected views.
    """
    heart_com = np.array(in_mesh.center)
    lv_apex = find_lv_apex_raytrace(cfg, in_mesh, subsampled_mesh)
    vert_vec = (lv_apex - heart_com) / np.linalg.norm(lv_apex - heart_com)

    # plotter = pv.Plotter()
    # plotter.add_mesh(in_mesh)
    # for i in [-0.01, 0.25, 0.75, 0.85, 0.99]:
    #     plotter.add_mesh(
    #         pv.PolyData(((1 - i) * heart_com + i * (lv_apex))),
    #         color="purple",
    #         point_size=300,
    #         render_points_as_spheres=True,
    #     )
    # # plotter.add_mesh(subsampled_mesh, color="red")
    # plotter.add_mesh(pv.PolyData(lv_apex), color="green", point_size=100, render_points_as_spheres=True)
    # plotter.add_mesh(pv.PolyData(heart_com), color="blue", point_size=100, render_points_as_spheres=True)
    # # plotter.add_mesh(pv.PolyData(heart_com + vert_vec), color="yellow", point_size=100, render_points_as_spheres=True)
    # plotter.add_mesh(pv.PolyData(vert_vec), color="red", point_size=100, render_points_as_spheres=True)
    # plotter.add_mesh(pv.PolyData(lv_apex - heart_com), color="red", point_size=100, render_points_as_spheres=True)
    #
    # # for i in range(-200, 200):
    # #     i = i / 100 + 0.01
    #
    # plotter.add_mesh(pv.PolyData([0, 0, 0]), color="black", point_size=300, render_points_as_spheres=True)
    # # show axis
    # plotter.show_axes()
    # plotter.show()

    l_atrium_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.LA)
    r_atrium_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.RA)
    aortic_valve_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.AORTIC_VALVE)
    mitral_valve_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.MITRAL_VALVE)
    tricuspid_valve_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.TRICUSPID_VALVE)

    returned_points4imgs = {}
    # if "rv_inflow" in cfg.DATA_OUT.SELECTED_VIEWS:
    #     rv_inflow = get_rv_inflow(cfg, subsampled_mesh, r_atrium_points)
    #     returned_points4imgs["rv_inflow"] = rv_inflow
    # returned_points4imgs += [rv_inflow]
    # if "PLAX" in cfg.DATA_OUT.SELECTED_VIEWS:
    if "PLAX" in selected_view:
        # lv_plax = get_lv_plax(cfg, subsampled_mesh, aortic_valve_points)
        lv_plax = get_a5c(l_atrium_points, aortic_valve_points, lv_apex)
        # returned_points4imgs += [lv_plax]
        returned_points4imgs["PLAX"] = [lv_plax, [l_atrium_points, aortic_valve_points, lv_apex]]
    # if "psax_aortic" in cfg.DATA_OUT.SELECTED_VIEWS:
    #     psax_aortic = get_psax_aortic(l_atrium_points, r_atrium_points, aortic_valve_points)
    #     # returned_points4imgs += [psax_aortic]
    #     returned_points4imgs["psax_aortic"] = [psax_aortic, [l_atrium_points, r_atrium_points, aortic_valve_points]]
    # if "PSAX basal" in cfg.DATA_OUT.SELECTED_VIEWS:
    if "PSAX basal" in selected_view:
        psax_mv = get_psax_mv(vert_vec, heart_com, lv_apex)
        returned_points4imgs["PSAX basal"] = [psax_mv, [vert_vec, heart_com, lv_apex]]
    if "PSAX mid" in selected_view:  # cfg.DATA_OUT.SELECTED_VIEWS:
        psax_pm = get_psax_pm(vert_vec, heart_com, lv_apex)
        returned_points4imgs["PSAX mid"] = [psax_pm, [vert_vec, heart_com, lv_apex]]
    if "PSAX apex" in selected_view:  # cfg.DATA_OUT.SELECTED_VIEWS:
        psax_lower = get_psax_lower(vert_vec, heart_com, lv_apex)
        # returned_points4imgs += [psax_lower]
        returned_points4imgs["PSAX apex"] = [psax_lower, [vert_vec, heart_com, lv_apex]]
    if "A4CH" in selected_view:  # cfg.DATA_OUT.SELECTED_VIEWS:
        # a4c = get_a4c(l_atrium_points, r_atrium_points, lv_apex)
        a4c = get_a4c(mitral_valve_points, tricuspid_valve_points, lv_apex)
        # returned_points4imgs += [a4c]
        returned_points4imgs["A4CH"] = [a4c, [l_atrium_points, r_atrium_points, lv_apex]]
    # if "a5c" in cfg.DATA_OUT.SELECTED_VIEWS:
    #     a5c = get_a5c(l_atrium_points, aortic_valve_points, lv_apex)
    #     # returned_points4imgs += [a5c]
    #     returned_points4imgs["a5c"] = a5c
    if "A2CH" in selected_view:  # cfg.DATA_OUT.SELECTED_VIEWS:
        (a2c, a2c_land_marks) = get_2ch(cfg, subsampled_mesh, lv_apex)
        # returned_points4imgs += [a2c]
        returned_points4imgs["A2CH"] = [a2c, a2c_land_marks]
    if "A3CH" in selected_view:  # cfg.DATA_OUT.SELECTED_VIEWS:
        a3c, a3c_land_marks = get_3ch(cfg, subsampled_mesh, lv_apex, aortic_valve_points)
        # returned_points4imgs += [a3c]
        returned_points4imgs["A3CH"] = [a3c, a3c_land_marks]
    return returned_points4imgs
