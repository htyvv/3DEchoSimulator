import sys
import vtk
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QColorDialog, QHBoxLayout,
    QVBoxLayout, QGridLayout, QLabel, QSlider, QComboBox, QPushButton, QGroupBox,
    QListWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtk.util.numpy_support import vtk_to_numpy

# --- mesh_lib imports ---
try:
    from mesh_lib.mesh_handler import Mesh
    from mesh_lib.config import cfg
    from mesh_lib.slicing import get_plane_numpy
    from mesh_lib.math_utils import rotate_to_xy_plane
    from mesh_lib.synthetic_mask import mask_augmentation, apply_us_cone
    from mesh_lib.inhouse_style_mask import process_view, fill_mask
    from mesh_lib.utils import Tags
except ImportError as e:
    print(f"Error: Failed to import from mesh_lib. Make sure it's in the python path. Details: {e}")
    sys.exit(1)


class MainWindowV2(QMainWindow):
    probe_updated = Signal(list, list)

    KEY_CONFIG = {
        'PLANT_PROBE': 'p', 'TILT_UP': 'w', 'TILT_DOWN': 's',
        'ROCK_LEFT': 'a', 'ROCK_RIGHT': 'd', 'ROTATE_CW': 'e',
        'ROTATE_CCW': 'q', 'SLIDE_UP': 'Up', 'SLIDE_DOWN': 'Down',
        'SLIDE_LEFT': 'Left', 'SLIDE_RIGHT': 'Right',
    }
    SENSITIVITY = {
        'SLIDE': 1.0, 'TILT_ROCK': 2.0, 'ROTATE': 5.0,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EchoCardiacSimulator v2 (MeshLib)")
        self.setGeometry(100, 100, 1440, 900)
        
        # VTK and App State
        self.vtk_actor = None
        self.probe_actor = None
        self.wedge_actor = None
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.probe_transform = vtk.vtkTransform()
        self.is_planting_mode = False
        self.final_data = None
        
        # New mesh_lib handler
        self.mesh_handler = None
        self.current_standard_view = None

        # New 2D View components
        self.cutting_plane = vtk.vtkPlane()
        self.cutter = vtk.vtkCutter()
        self.cutter.SetCutFunction(self.cutting_plane)
        self.triangulator = vtk.vtkContourTriangulator()
        self.triangulator.SetInputConnection(self.cutter.GetOutputPort())
        self.cut_mapper = vtk.vtkPolyDataMapper()
        self.cut_mapper.SetInputConnection(self.triangulator.GetOutputPort())
        self.cut_actor = vtk.vtkActor()

        # UI Setup
        self._setup_ui()
        
        # VTK View Setup
        self._setup_3d_view()
        self._setup_2d_view()
        self._connect_signals()
        self._update_planting_mode_visuals()

    def _setup_ui(self):
        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Menu Bar ---
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open VTK File")
        open_action.triggered.connect(self._open_file_dialog)
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # --- Viewer Panel (Left) ---
        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        self.vtkWidget_3d = QVTKRenderWindowInteractor()
        self.vtkWidget_2d = QVTKRenderWindowInteractor()
        viewer_layout.addWidget(QLabel("3D View"))
        viewer_layout.addWidget(self.vtkWidget_3d, 2)
        viewer_layout.addWidget(QLabel("2D Slice View"))
        viewer_layout.addWidget(self.vtkWidget_2d, 1)

        # --- Options Panel (Right) ---
        options_panel = QWidget()
        options_layout = QVBoxLayout(options_panel)
        options_panel.setFixedWidth(350)

        # File Info
        self.file_info_label = QLabel("No file loaded.")
        self.file_info_label.setWordWrap(True)
        options_layout.addWidget(self.file_info_label)

        # Standard Views
        standard_views_group = QGroupBox("Standard Views")
        standard_views_layout = QGridLayout(standard_views_group)
        views = ["A4CH", "A2CH", "A3CH", "PLAX"]
        for i, view in enumerate(views):
            btn = QPushButton(view)
            btn.clicked.connect(lambda checked=False, v=view: self._set_standard_view(v))
            standard_views_layout.addWidget(btn, i // 2, i % 2)
        options_layout.addWidget(standard_views_group)

        # Available Labels
        labels_group = QGroupBox("Available Labels in File")
        labels_layout = QVBoxLayout(labels_group)
        self.labels_list_widget = QListWidget()
        labels_layout.addWidget(self.labels_list_widget)
        options_layout.addWidget(labels_group)

        # Heart Model Options
        heart_group = QGroupBox("Heart Model Options")
        heart_layout = QGridLayout(heart_group)
        self.heart_color_button = QPushButton("Color")
        self.heart_opacity_slider = QSlider(Qt.Horizontal)
        self.heart_opacity_slider.setRange(0, 100); self.heart_opacity_slider.setValue(100)
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Surface", "Wireframe", "Points"])
        heart_layout.addWidget(QLabel("Color:"), 0, 0); heart_layout.addWidget(self.heart_color_button, 0, 1)
        heart_layout.addWidget(QLabel("Opacity:"), 1, 0); heart_layout.addWidget(self.heart_opacity_slider, 1, 1)
        heart_layout.addWidget(QLabel("Render Mode:"), 2, 0); heart_layout.addWidget(self.render_mode_combo, 2, 1)
        options_layout.addWidget(heart_group)

        # Probe Options
        self.probe_group = QGroupBox("Probe Info")
        probe_layout = QGridLayout(self.probe_group)
        self.probe_info_label = QLabel("Position: [N/A]")
        self.probe_info_label.setWordWrap(True)
        self.fan_radius_slider = QSlider(Qt.Horizontal)
        self.fan_radius_slider.setRange(20, 150); self.fan_radius_slider.setValue(80)
        self.fan_color_button = QPushButton("Color")
        self.fan_opacity_slider = QSlider(Qt.Horizontal)
        self.fan_opacity_slider.setRange(0, 100); self.fan_opacity_slider.setValue(50)
        probe_layout.addWidget(self.probe_info_label, 0, 0, 1, 2)
        probe_layout.addWidget(QLabel("Fan Radius:"), 1, 0); probe_layout.addWidget(self.fan_radius_slider, 1, 1)
        probe_layout.addWidget(QLabel("Fan Color:"), 2, 0); probe_layout.addWidget(self.fan_color_button, 2, 1)
        probe_layout.addWidget(QLabel("Fan Opacity:"), 3, 0); probe_layout.addWidget(self.fan_opacity_slider, 3, 1)
        options_layout.addWidget(self.probe_group)

        # Camera Options
        camera_group = QGroupBox("Camera Options")
        camera_layout = QGridLayout(camera_group)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.save_screenshot_button = QPushButton("Save Screenshot")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 200); self.sensitivity_slider.setValue(85)
        camera_layout.addWidget(self.reset_camera_button, 0, 0); camera_layout.addWidget(self.save_screenshot_button, 0, 1)
        camera_layout.addWidget(QLabel("3D Sensitivity:"), 1, 0); camera_layout.addWidget(self.sensitivity_slider, 1, 1)
        options_layout.addWidget(camera_group)
        
        options_layout.addStretch(1)
        
        self.main_layout.addWidget(viewer_panel, 3)
        self.main_layout.addWidget(options_panel, 1)

    def _setup_3d_view(self):
        self.renderer_3d = vtk.vtkRenderer()
        self.vtkWidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.vtkWidget_3d.GetRenderWindow().SetMultiSamples(8)
        self.interactor_3d = self.vtkWidget_3d.GetRenderWindow().GetInteractor()
        self.renderer_3d.SetBackground(0.1, 0.2, 0.4)
        self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.camera_style.SetDefaultRenderer(self.renderer_3d)
        self.camera_style.SetKeyPressActivation(False)
        self.interactor_3d.SetInteractorStyle(self.camera_style)
        self.interactor_3d.AddObserver("KeyPressEvent", self._key_press_handler, 1.0)
        self.interactor_3d.AddObserver("LeftButtonPressEvent", self._left_button_press_handler, 1.0)
        self.interactor_3d.AddObserver("MouseMoveEvent", self._mouse_move_handler, 1.0)
        if hasattr(self.camera_style, 'SetMotionFactor'): self.camera_style.SetMotionFactor(0.85)
        self.interactor_3d.Initialize()
        self.vtkWidget_3d.Start()
        
    def _setup_2d_view(self):
        self.renderer_2d = vtk.vtkRenderer()
        self.vtkWidget_2d.GetRenderWindow().AddRenderer(self.renderer_2d)
        self.vtkWidget_2d.GetRenderWindow().SetMultiSamples(8)
        self.renderer_2d.SetBackground(0.0, 0.0, 0.0)
        self.vtkWidget_2d.GetRenderWindow().GetInteractor().Disable()

        # Simplified 2D slice pipeline
        self.cut_actor.SetMapper(self.cut_mapper)
        self.cut_actor.GetProperty().SetColor(1.0, 1.0, 1.0) # Set slice to white
        self.cut_actor.GetProperty().SetLineWidth(2)
        self.renderer_2d.AddActor(self.cut_actor)

        # The mapper for the slice should use the same lookup table as the 3D model
        # to show colored slices in the future if needed.
        # For now, we use a single color.
        # self.cut_mapper.SetScalarVisibility(True)
        # self.cut_mapper.SetScalarModeToUseCellData()
        # self.cut_mapper.SelectColorArray(cfg.LABELS.LABEL_NAME)

    def _connect_signals(self):
        self.heart_color_button.clicked.connect(self._set_actor_color)
        self.heart_opacity_slider.valueChanged.connect(self._set_actor_opacity)
        self.render_mode_combo.currentIndexChanged.connect(self._set_render_mode)
        self.reset_camera_button.clicked.connect(self._reset_camera)
        self.save_screenshot_button.clicked.connect(self._save_screenshot)
        self.sensitivity_slider.valueChanged.connect(self._set_camera_sensitivity)
        self.fan_radius_slider.valueChanged.connect(self._update_probe_shape)
        self.fan_color_button.clicked.connect(self._set_probe_color)
        self.fan_opacity_slider.valueChanged.connect(self._set_probe_opacity)
        self.probe_updated.connect(self._handle_probe_update_gui)
        self.probe_updated.connect(self._update_2d_slice)

    def _mouse_move_handler(self, interactor, event):
        if self.is_planting_mode:
            self.current_standard_view = None
            mouse_pos = interactor.GetEventPosition()
            self.picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer_3d)
            if self.picker.GetCellId() >= 0 and self.picker.GetActor() == self.vtk_actor:
                pick_pos = self.picker.GetPickPosition()
                pick_normal = self.picker.GetPickNormal()
                camera = self.renderer_3d.GetActiveCamera()
                
                inward_normal = [-pick_normal[0], -pick_normal[1], -pick_normal[2]]
                probe_y = inward_normal
                vtk.vtkMath.Normalize(probe_y)

                cam_up = list(camera.GetViewUp())
                probe_x = [0,0,0]
                vtk.vtkMath.Cross(cam_up, probe_y, probe_x)
                vtk.vtkMath.Normalize(probe_x)

                probe_z = [0,0,0]
                vtk.vtkMath.Cross(probe_x, probe_y, probe_z)
                vtk.vtkMath.Normalize(probe_z)
                
                matrix = vtk.vtkMatrix4x4()
                matrix.Identity()
                for i in range(3):
                    matrix.SetElement(i, 0, probe_x[i])
                    matrix.SetElement(i, 1, probe_y[i])
                    matrix.SetElement(i, 2, probe_z[i])
                    matrix.SetElement(i, 3, pick_pos[i])

                self.probe_transform.SetMatrix(matrix)
                self._update_probe_view()

    def _left_button_press_handler(self, interactor, event):
        if self.is_planting_mode:
            self.is_planting_mode = False
            self._update_planting_mode_visuals()

    def _key_press_handler(self, interactor, event):
        if not self.probe_actor: return
        key = interactor.GetKeySym()

        if key == self.KEY_CONFIG['PLANT_PROBE']:
            self.is_planting_mode = not self.is_planting_mode
            self._update_planting_mode_visuals()
            return
        
        self.current_standard_view = None
        delta_t = vtk.vtkTransform()
        
        if key == self.KEY_CONFIG['TILT_UP']: delta_t.RotateZ(-self.SENSITIVITY['TILT_ROCK'])
        elif key == self.KEY_CONFIG['TILT_DOWN']: delta_t.RotateZ(self.SENSITIVITY['TILT_ROCK'])
        elif key == self.KEY_CONFIG['ROCK_LEFT']: delta_t.RotateX(-self.SENSITIVITY['TILT_ROCK'])
        elif key == self.KEY_CONFIG['ROCK_RIGHT']: delta_t.RotateX(self.SENSITIVITY['TILT_ROCK'])
        elif key == self.KEY_CONFIG['ROTATE_CCW']: delta_t.RotateY(-self.SENSITIVITY['ROTATE'])
        elif key == self.KEY_CONFIG['ROTATE_CW']: delta_t.RotateY(self.SENSITIVITY['ROTATE'])
        elif key == self.KEY_CONFIG['SLIDE_UP']: delta_t.Translate(self.SENSITIVITY['SLIDE'], 0, 0)
        elif key == self.KEY_CONFIG['SLIDE_DOWN']: delta_t.Translate(-self.SENSITIVITY['SLIDE'], 0, 0)
        elif key == self.KEY_CONFIG['SLIDE_LEFT']: delta_t.Translate(0, 0, -self.SENSITIVITY['SLIDE'])
        elif key == self.KEY_CONFIG['SLIDE_RIGHT']: delta_t.Translate(0, 0, self.SENSITIVITY['SLIDE'])
        
        self.probe_transform.Concatenate(delta_t)
        self._update_probe_view()

    def _update_planting_mode_visuals(self):
        if self.is_planting_mode:
            self.probe_group.setTitle("Probe Info (Planting ON)")
            self.probe_group.setStyleSheet("QGroupBox { border: 2px solid #FFD700; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        else:
            self.probe_group.setTitle("Probe Info")
            self.probe_group.setStyleSheet("")
        self.vtkWidget_3d.GetRenderWindow().Render()

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open VTK File", "", "VTK Files (*.vtk)")
        if file_path: self._render_vtk_file(file_path)

    def _render_vtk_file(self, file_path):
        if self.vtk_actor: self.renderer_3d.RemoveActor(self.vtk_actor)
        if self.probe_actor: self.renderer_3d.RemoveActor(self.probe_actor)
        reader = vtk.vtkDataSetReader(); reader.SetFileName(file_path); reader.Update()
        self.final_data = self.get_poly_data_from_reader(reader)
        if self.final_data is None: return

        try:
            self.mesh_handler = Mesh(cfg, pv.wrap(self.final_data), origin=False)
            print("Calculating all standard view points...")
            self.mesh_handler.get_key_cardiac_points()
            print("Done.")
        except Exception as e:
            print(f"Error initializing Mesh handler or calculating landmarks: {e}")
            self.mesh_handler = None

        self.file_info_label.setText(f"Loaded: {file_path.split('/')[-1]}")
        self._populate_labels_list()
        
        self.cutter.SetInputData(self.final_data)
        
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.final_data)
        
        # Create a dynamic and colorful LUT for the 3D view
        lut3d = vtk.vtkLookupTable()
        max_label = max(Tags.values()) if Tags else 0
        lut3d.SetNumberOfTableValues(max_label + 1)
        lut3d.SetTableRange(0, max_label)

        # Use vtkColorSeries to generate distinct colors
        color_series = vtk.vtkColorSeries()
        color_series.SetColorScheme(vtk.vtkColorSeries.BREWER_QUALITATIVE_PAIRED)

        # Explicitly set colors for important structures to ensure visibility
        explicit_colors = {
            Tags["lv_myocardium"]: (217, 2, 42),   # Red
            Tags["rv_myocardium"]: (2, 115, 183),  # Blue
            Tags["la_myocardium"]: (242, 102, 46), # Orange
            Tags["ra_myocardium"]: (247, 150, 53), # Yellow-Orange
            Tags["aorta"]: (155, 17, 30),          # Dark Red
        }

        for name, label_val in Tags.items():
            if label_val == 0: # background
                lut3d.SetTableValue(label_val, 0, 0, 0, 0) # Transparent
            elif label_val in explicit_colors:
                r, g, b = explicit_colors[label_val]
                lut3d.SetTableValue(label_val, r/255.0, g/255.0, b/255.0, 1.0)
            else:
                color = color_series.GetColor(label_val % color_series.GetNumberOfColors())
                lut3d.SetTableValue(label_val, color.GetRed()/255.0, color.GetGreen()/255.0, color.GetBlue()/255.0, 1.0)

        lut3d.Build()
        mapper.SetLookupTable(lut3d)
        mapper.SetScalarRange(0, max_label)
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray(cfg.LABELS.LABEL_NAME)

        self.vtk_actor = vtk.vtkActor(); self.vtk_actor.SetMapper(mapper)
        self.renderer_3d.AddActor(self.vtk_actor)
        
        initial_radius = self.fan_radius_slider.value()
        self.probe_actor = self._create_probe_actor(initial_radius)
        self.renderer_3d.AddActor(self.probe_actor)
        
        self.probe_transform.Identity()
        self.probe_transform.Translate(list(self.vtk_actor.GetCenter()))
        self._update_probe_view()

        self.renderer_3d.ResetCamera(); self.renderer_3d.ResetCameraClippingRange()
        self.vtkWidget_3d.GetRenderWindow().Render()
    
    def _populate_labels_list(self):
        self.labels_list_widget.clear()
        if not self.final_data:
            return

        cell_data = self.final_data.GetCellData()
        label_array_vtk = cell_data.GetArray(cfg.LABELS.LABEL_NAME)

        if not label_array_vtk:
            self.labels_list_widget.addItem("No 'elemTag' array found.")
            return

        label_array_np = vtk_to_numpy(label_array_vtk)
        unique_labels = np.unique(label_array_np)

        value_to_name = {v: k for k, v in Tags.items()}

        if len(unique_labels) == 0:
            self.labels_list_widget.addItem("'elemTag' array is empty.")
            return

        for label_val in sorted(unique_labels):
            name = value_to_name.get(label_val, "Unknown Label")
            self.labels_list_widget.addItem(f"{name} ({int(label_val)})")

    def get_poly_data_from_reader(self, reader):
        output = reader.GetOutput()
        if not output or output.GetNumberOfPoints() == 0:
            print("Error: Reader output is empty.")
            return None
        
        data_type = output.GetClassName()
        if data_type in ['vtkImageData', 'vtkStructuredPoints']:
            contour=vtk.vtkContourFilter(); contour.SetInputData(output); contour.SetValue(0, 128); contour.Update()
            return contour.GetOutput()
        elif data_type == 'vtkUnstructuredGrid':
            if output.GetNumberOfCells() == 0:
                return output
            if output.GetCell(0).GetCellType() in [vtk.VTK_VOXEL, vtk.VTK_TETRA, vtk.VTK_HEXAHEDRON]:
                surface=vtk.vtkDataSetSurfaceFilter(); surface.SetInputData(output); surface.Update()
                return surface.GetOutput()
            return output
        elif data_type == 'vtkPolyData':
            return output
        return None

    def _create_probe_actor(self, radius):
        cylinder = vtk.vtkCylinderSource(); cylinder.SetRadius(5); cylinder.SetHeight(14); cylinder.SetResolution(20); cylinder.SetCenter(0, -7, 0)
        cyl_mapper = vtk.vtkPolyDataMapper(); cyl_mapper.SetInputConnection(cylinder.GetOutputPort())
        cyl_actor = vtk.vtkActor(); cyl_actor.SetMapper(cyl_mapper); cyl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        points = vtk.vtkPoints(); fan_len = radius; fan_width = radius
        points.InsertNextPoint(0, 0, 0); points.InsertNextPoint(0, fan_len, -fan_width / 2); points.InsertNextPoint(0, fan_len,  fan_width / 2)
        triangle = vtk.vtkTriangle(); triangle.GetPointIds().SetId(0, 0); triangle.GetPointIds().SetId(1, 1); triangle.GetPointIds().SetId(2, 2)
        triangles = vtk.vtkCellArray(); triangles.InsertNextCell(triangle)
        polydata = vtk.vtkPolyData(); polydata.SetPoints(points); polydata.SetPolys(triangles)
        extrude = vtk.vtkLinearExtrusionFilter(); extrude.SetInputData(polydata); extrude.SetExtrusionTypeToVectorExtrusion()
        vec = [2, 0, 0]; extrude.SetVector(*vec)
        wedge_mapper = vtk.vtkPolyDataMapper(); wedge_mapper.SetInputConnection(extrude.GetOutputPort())
        self.wedge_actor = vtk.vtkActor(); self.wedge_actor.SetMapper(wedge_mapper); self.wedge_actor.GetProperty().SetColor(0.2, 1.0, 0.2); self.wedge_actor.GetProperty().SetOpacity(0.5)
        self.wedge_actor.SetPosition(-vec[0]/2.0, 0, 0)
        assembly = vtk.vtkAssembly(); assembly.AddPart(cyl_actor); assembly.AddPart(self.wedge_actor)
        sphere = vtk.vtkSphereSource(); sphere.SetRadius(2.0)
        sm = vtk.vtkPolyDataMapper(); sm.SetInputConnection(sphere.GetOutputPort())
        pivot = vtk.vtkActor(); pivot.SetMapper(sm); pivot.GetProperty().SetColor(1,1,0)
        assembly.AddPart(pivot)
        return assembly

    def _update_probe_shape(self, value):
        if not self.probe_actor or not self.renderer_3d: return
        self.renderer_3d.RemoveActor(self.probe_actor)
        self.probe_actor = self._create_probe_actor(value)
        self.renderer_3d.AddActor(self.probe_actor)
        self._update_probe_view()

    def _update_probe_view(self):
        if not self.probe_actor: return
        final_matrix = self.probe_transform.GetMatrix()
        self.probe_actor.SetUserMatrix(final_matrix)
        pos = [final_matrix.GetElement(i, 3) for i in range(3)]
        cutting_normal = [final_matrix.GetElement(i, 0) for i in range(3)]
        self.probe_updated.emit(pos, cutting_normal)

    def _handle_probe_update_gui(self, pos, cutting_normal):
        pos_str = f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        self.probe_info_label.setText(pos_str)
        self.vtkWidget_3d.GetRenderWindow().Render()

    def _set_standard_view(self, view_name):
        if not self.mesh_handler:
            print("Please load a mesh first.")
            return

        print(f"Setting standard view to: {view_name}")
        self.current_standard_view = view_name

        try:
            view_data, _ = self.mesh_handler.cardiac_out_points[view_name]
            normal, origin = view_data
        except (KeyError, AttributeError) as e:
            print(f"Could not get data for view {view_name}: {e}. Check if landmarks were found.")
            self.current_standard_view = None
            return

        probe_x = np.array(normal)
        if np.linalg.norm(probe_x) == 0: return
        probe_x = probe_x / np.linalg.norm(probe_x)

        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(probe_x, up)) > 0.99:
            up = np.array([0.0, 0.0, 1.0])

        probe_z = np.cross(probe_x, up)
        probe_z = probe_z / np.linalg.norm(probe_z)
        probe_y = np.cross(probe_z, probe_x)
        
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, probe_x[i])
            matrix.SetElement(i, 1, probe_y[i])
            matrix.SetElement(i, 2, probe_z[i])
            matrix.SetElement(i, 3, origin[i])
        
        self.probe_transform.SetMatrix(matrix)
        self._update_probe_view()

    def _update_2d_slice(self, pos, cutting_normal):
        if self.final_data is None or not self.probe_actor:
            return

        # Update the cutting plane based on the probe's current position and orientation
        self.cutting_plane.SetOrigin(pos)
        self.cutting_plane.SetNormal(cutting_normal)
        self.cutter.Update()
        self.triangulator.Update()

        # Set up the 2D camera to look "down the probe"
        cam2d = self.renderer_2d.GetActiveCamera()
        
        # The camera's focal point is the probe's position
        cam2d.SetFocalPoint(pos)
        # The camera's position is slightly "behind" the probe, looking along the normal
        cam_pos = [pos[i] - cutting_normal[i] * 20 for i in range(3)]
        cam2d.SetPosition(cam_pos)

        # The "up" vector for the camera determines the slice's orientation.
        # We use the probe's Y-axis (from its transform matrix) to ensure the pivot is at the top.
        probe_matrix = self.probe_transform.GetMatrix()
        view_up = [-probe_matrix.GetElement(i, 1) for i in range(3)]
        cam2d.SetViewUp(view_up)

        self.renderer_2d.ResetCamera()
        self.vtkWidget_2d.GetRenderWindow().Render()

    def _set_actor_color(self):
        if not self.vtk_actor: return
        color=QColorDialog.getColor()
        if color.isValid(): self.vtk_actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF()); self.vtkWidget_3d.GetRenderWindow().Render()
    def _set_probe_color(self):
        if not self.wedge_actor: return
        color=QColorDialog.getColor()
        if color.isValid(): self.wedge_actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF()); self.fan_color_button.setStyleSheet(f"background-color: {color.name()}"); self.vtkWidget_3d.GetRenderWindow().Render()
    def _set_actor_opacity(self, value):
        if not self.vtk_actor: return
        self.vtk_actor.GetProperty().SetOpacity(value / 100.0); self.vtkWidget_3d.GetRenderWindow().Render()
    def _set_probe_opacity(self, value):
        if not self.wedge_actor: return
        self.wedge_actor.GetProperty().SetOpacity(value / 100.0); self.vtkWidget_3d.GetRenderWindow().Render()
    def _set_render_mode(self, index):
        if not self.vtk_actor: return
        if index==0: self.vtk_actor.GetProperty().SetRepresentationToSurface()
        elif index==1: self.vtk_actor.GetProperty().SetRepresentationToWireframe()
        elif index==2: self.vtk_actor.GetProperty().SetRepresentationToPoints()
        self.vtkWidget_3d.GetRenderWindow().Render()
    
    def _reset_camera(self):
        if self.vtk_actor: self.renderer_3d.ResetCamera(); self.renderer_3d.ResetCameraClippingRange(); self.vtkWidget_3d.GetRenderWindow().Render()
    def _save_screenshot(self):
        file_path, _=QFileDialog.getSaveFileName(self, "Save Screenshot", "screenshot.png", "Images (*.png *.xpm *.jpg)")
        if file_path: self.grab().save(file_path)
    def _set_camera_sensitivity(self, value):
        if self.camera_style: self.camera_style.SetMotionFactor(value / 100.0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.window = MainWindowV2()
    app.window.show()
    sys.exit(app.exec())
