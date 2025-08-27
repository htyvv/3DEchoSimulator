import numpy as np
import vtk


def find_points_on_plane(plane_vars):
    """Finds coordinates of points which lay on a plane"""
    plane_vars = np.array([float(var) for var in plane_vars])
    u1 = -plane_vars[3] / plane_vars[2]
    u2 = -plane_vars[3] / plane_vars[1]
    u3 = -plane_vars[3] / plane_vars[0]

    return [0, 0, u1], [0, u2, 0], [u3, 0, 0]


def find_plane_from_normal(vector, point):
    """Given a vector and point coordinate find the plane equation."""
    return -vector[0], -vector[1], -vector[2], np.sum(np.multiply(vector, point))


def normalized(a, axis=-1, order=2):
    """Normalize a vector"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))


def calc_rot_mat_to_xy_plane(normalized_v, check_determinant=False):
    """Find the 3D rotation matrix which will transform an arbitrary plane to be in the xy plane i.e. no z component."""

    def expand_rot_mat(rot_mat):
        full_rot_mat = np.zeros((4, 4))
        full_rot_mat[:3, :3] = rot_mat
        full_rot_mat[3, 3] = 1.0
        return full_rot_mat

    a, b, c = normalized_v

    # Handle case where the vector is already aligned with z-axis
    if np.allclose([a, b], [0, 0]):
        if c > 0:
            return np.identity(4)
        else: # Pointing along -z, rotate 180 degrees around x-axis
            rot_mat = np.array([
                [1., 0., 0.],
                [0.,-1., 0.],
                [0., 0.,-1.]
            ])
            return expand_rot_mat(rot_mat)

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


def invert_rot_mat_to_xy_plane(rot_mat_4x4):
    """
    Inverts the rotation matrix which was used to transform an arbitrary plane to be in the xy plane.

    Parameters:
    rot_mat_4x4 (np.array): The 4x4 rotation matrix returned by calc_rot_mat_to_xy_plane.

    Returns:
    np.array: The inverted 4x4 rotation matrix.
    """

    def invert_expand_rot_mat(rot_mat):
        inv_full_rot_mat = np.zeros((4, 4))
        inv_full_rot_mat[:3, :3] = rot_mat.T  # Transpose of the 3x3 rotation matrix
        inv_full_rot_mat[3, 3] = 1.0
        return inv_full_rot_mat

    # Extract the 3x3 rotation matrix
    r_mat = rot_mat_4x4[:3, :3]

    # Invert the rotation matrix
    inv_r_mat = invert_expand_rot_mat(r_mat)

    return inv_r_mat


def calc_rot_mat_to_normal(original_normal, target_normal, check_determinant=False):
    """Find the 3D rotation matrix which will transform an arbitrary plane to be in the target_normal"""

    def expand_rot_mat(rot_mat):
        full_rot_mat = np.zeros((4, 4))
        full_rot_mat[:3, :3] = rot_mat
        full_rot_mat[3, 3] = 1.0
        return full_rot_mat

    a, b, c = original_normal
    a_t, b_t, c_t = target_normal

    square = a ** 2 + b ** 2 + c ** 2
    square_t = a_t ** 2 + b_t ** 2 + c_t ** 2

    cos_theta = (a * a_t + b * b_t + c * c_t) / np.sqrt(square * square_t)
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    u1 = (b * c_t - c * b_t) / np.sqrt((b * c_t - c * b_t) ** 2 + (c * a_t - a * c_t) ** 2 + (a * b_t - b * a_t) ** 2)
    u2 = (c * a_t - a * c_t) / np.sqrt((b * c_t - c * b_t) ** 2 + (c * a_t - a * c_t) ** 2 + (a * b_t - b * a_t) ** 2)
    u3 = (a * b_t - b * a_t) / np.sqrt((b * c_t - c * b_t) ** 2 + (c * a_t - a * c_t) ** 2 + (a * b_t - b * a_t) ** 2)

    r_mat = np.array(
        [
            [
                cos_theta + u1 ** 2 * (1 - cos_theta),
                u1 * u2 * (1 - cos_theta),
                u1 * u3 * (1 - cos_theta),
                u2 * sin_theta,
            ],
            [
                u1 * u2 * (1 - cos_theta),
                cos_theta + u2 ** 2 * (1 - cos_theta),
                u2 * u3 * (1 - cos_theta),
                -u1 * sin_theta,
            ],
            [
                u1 * u3 * (1 - cos_theta),
                u2 * u3 * (1 - cos_theta),
                cos_theta + u3 ** 2 * (1 - cos_theta),
                -u3 * sin_theta,
            ],
            [-u2 * sin_theta, u1 * sin_theta, u3 * sin_theta, cos_theta],
        ]
    )

    if check_determinant:
        print("det of R is: {0:.5f}".format(np.linalg.det(r_mat)))

    return expand_rot_mat(r_mat)


def plane_equation_calc(in_points):
    """Calculation of the equation of a plane."""
    x = np.cross(in_points[1] - in_points[0], in_points[1] - in_points[2])
    a, b, c = x / np.linalg.norm(x)
    return normalized(np.array([a, b, c]), axis=-1, order=1)


def rotate_to_xy_plane(mesh, normalized_v):
    """Calculate the rotation matrix to get an arbitrary plane to the xy plane, then apply it to that plane."""
    rot_mat = calc_rot_mat_to_xy_plane(normalized_v)
    return transform_vtk_rotation_matrix(mesh, np.ravel(rot_mat))


def rotate_plane_to_target_normal(mesh, orginal_normal, target_normal):
    """Calculate the rotation matrix to get an arbitrary plane to the xy plane, then apply it to that plane."""
    rot_mat = calc_rot_mat_to_normal(orginal_normal, target_normal)
    return transform_vtk_rotation_matrix(mesh, np.ravel(rot_mat))


def transform_vtk_rotation_matrix(surface, matrix):
    """Apply rotation matrix to a vtk mesh."""
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surface)
    transformFilter.Update()
    return transformFilter.GetOutput()


def pnt2line(pnt, start, end):
    """Find the location of the closest projection of a point to a vector (line)."""

    def add(v, w):
        x, y, z = v
        X, Y, Z = w
        return x + X, y + Y, z + Z

    def scale(v, sc):
        x, y, z = v
        return x * sc, y * sc, z * sc

    def unit(v):
        x, y, z = v
        mag = np.linalg.norm(v)
        return x / mag, y / mag, z / mag

    def vector(b, e):
        x, y, z = b
        X, Y, Z = e
        return X - x, Y - y, Z - z

    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / np.linalg.norm(line_vec))
    t = np.dot(line_unitvec, pnt_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = scale(line_vec, t)
    dist = np.linalg.norm(vector(nearest, pnt_vec))
    nearest = add(nearest, start)
    return dist, nearest


def translate_mesh_to_origin(mesh):
    center_of_mass = mesh.center_of_mass()

    transform_matrix = np.array(
        [[1, 0, 0, -center_of_mass[0]], [0, 1, 0, -center_of_mass[1]], [0, 0, 1, -center_of_mass[2]], [0, 0, 0, 1]]
    )
    return mesh.transform(transform_matrix)
