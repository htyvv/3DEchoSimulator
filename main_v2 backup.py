import sys
import vtk
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QColorDialog, QHBoxLayout,
    QVBoxLayout, QGridLayout, QLabel, QSlider, QComboBox, QPushButton, QGroupBox,
    QListWidget, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# --- mesh_lib imports ---
try:
    from mesh_lib.mesh_handler import Mesh
    from mesh_lib.config import cfg
    from mesh_lib.slicing import get_plane_numpy
    from mesh_lib.math_utils import rotate_to_xy_plane
    from mesh_lib.math_utils import calc_rot_mat_to_xy_plane
    from mesh_lib.inhouse_style_mask import process_view, fill_mask
    from mesh_lib.utils import Tags
except ImportError as e:
    print(f"Error: Failed to import from mesh_lib. Make sure it's in the python path. Details: {e}")
    sys.exit(1)

# scikit-image is used for fast polygon filling
try:
    from skimage.draw import polygon as skimage_polygon
except ImportError:
    print("Warning: scikit-image not found. Please install it (`pip install scikit-image`) for 2D slicing.")
    skimage_polygon = None


class MainWindowV2(QMainWindow):
    probe_updated = Signal()

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
        
        self.display_mesh = None
        
        # New mesh_lib handler
        self.mesh_handler = None
        self.current_standard_view = None
        self.obb_tree = None
        self.heart_center = [0, 0, 0]

        # --- 2D View Components ---
        # Image pipeline for rendering the slice with effects
        self.image_actor_2d = vtk.vtkImageActor()
        self.data_importer_2d = vtk.vtkImageImport()
        self.color_map_2d = vtk.vtkImageMapToColors()
        self.empty_image_data = vtk.vtkImageData() # Placeholder for empty slices

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

        # 2D View Options
        view_2d_group = QGroupBox("2D View Options")
        view_2d_layout = QVBoxLayout(view_2d_group)
        self.show_colors_checkbox = QCheckBox("Show Slice Colors")
        self.show_colors_checkbox.setEnabled(False) # Disabled as we are not using labels for now
        self.apply_cone_checkbox = QCheckBox("Apply Cone Effect")
        self.apply_cone_checkbox.setChecked(True)
        view_2d_layout.addWidget(self.show_colors_checkbox)
        view_2d_layout.addWidget(self.apply_cone_checkbox)
        options_layout.addWidget(view_2d_group)

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

        # Setup the image-based rendering pipeline for the 2D view
        self.lut_2d = self._create_2d_lut()
        self.color_map_2d.SetLookupTable(self.lut_2d)
        self.color_map_2d.SetOutputFormatToRGBA()
        self.color_map_2d.SetInputConnection(self.data_importer_2d.GetOutputPort())
        
        self.image_actor_2d.SetInputData(self.color_map_2d.GetOutput())
        self.renderer_2d.AddActor(self.image_actor_2d)

    def _create_2d_lut(self):
        """Creates a lookup table for the 2D slice view."""
        lut = vtk.vtkLookupTable()
        # We only need 3 values now: 0=background, 1=cone, 2=tissue
        lut.SetNumberOfTableValues(3)
        lut.SetTableRange(0, 2)

        # 0: Background (transparent)
        lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
        # 1: Cone speckle background
        lut.SetTableValue(1, 0.1, 0.1, 0.1, 1.0)
        # 2: Tissue color (white)
        lut.SetTableValue(2, 1.0, 1.0, 1.0, 1.0)

        lut.Build()
        return lut

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
        self.show_colors_checkbox.stateChanged.connect(self._update_probe_view)
        self.apply_cone_checkbox.stateChanged.connect(self._update_probe_view)

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

        # 프로브 심기(Planting) 기능 처리
        # 이 키가 눌리면 planting 모드를 토글하고, 이벤트 전파를 중단(AbortFlagOn)하여
        # VTK의 기본 'p'(pick) 기능이 실행되지 않도록 합니다.
        if key == self.KEY_CONFIG['PLANT_PROBE']:
            self.is_planting_mode = not self.is_planting_mode
            self._update_planting_mode_visuals()
            interactor.AbortFlagOn() # 올바른 메서드: 이벤트 전파 중단
            return

        delta_t = vtk.vtkTransform()
        key_handled = True # 키가 처리되었는지 여부를 추적하는 플래그

        # 프로브 조작 키 처리
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
        else:
            key_handled = False # 우리가 정의한 키가 아니면 플래그를 False로 설정

        # 우리가 정의한 키 중 하나가 눌렸다면,
        # 프로브를 업데이트하고 이벤트 전파를 중단합니다.
        # 'w' 키가 TILT_UP으로 사용되었으므로 여기서 처리되어
        # VTK의 기본 'wireframe' 기능이 실행되지 않습니다.
        if key_handled:
            self.current_standard_view = None
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
        print(f"Loading VTK file: {file_path}")
        if self.vtk_actor: self.renderer_3d.RemoveActor(self.vtk_actor)
        if self.probe_actor: self.renderer_3d.RemoveActor(self.probe_actor)
        reader = vtk.vtkDataSetReader(); reader.SetFileName(file_path); reader.Update()
        
        mesh_from_file = self.get_poly_data_from_reader(reader)
        if mesh_from_file is None: return

        self.display_mesh = mesh_from_file

        # --- Dynamic Fan Radius ---
        bounds = self.display_mesh.GetBounds()
        x_len = bounds[1] - bounds[0]
        y_len = bounds[3] - bounds[2]
        z_len = bounds[5] - bounds[4]
        diagonal_length = np.sqrt(x_len**2 + y_len**2 + z_len**2)
        
        # Set a reasonable max and default value based on mesh size
        new_max_radius = int(diagonal_length * 1.2)
        new_default_radius = int(diagonal_length * 0.8)
        self.fan_radius_slider.setRange(20, new_max_radius)
        self.fan_radius_slider.setValue(new_default_radius)
        # --- End Dynamic Fan Radius ---

        try:
            self.mesh_handler = Mesh(cfg, pv.wrap(self.display_mesh), origin=False)
            print("Calculating all standard view points...")
            self.mesh_handler.get_key_cardiac_points()
            print("Done.")
        except Exception as e:
            print(f"Error initializing Mesh handler or calculating landmarks: {e}")
            self.mesh_handler = None

        self.file_info_label.setText(f"Loaded: {file_path.split('/')[-1]}")
        self._populate_labels_list()
        
        # Create an OBB tree for efficient ray-casting for probe placement
        self.obb_tree = vtk.vtkOBBTree()
        self.obb_tree.SetDataSet(self.display_mesh) # Use display mesh for picking
        self.obb_tree.BuildLocator()
        self.heart_center = self.display_mesh.GetCenter()
        
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.display_mesh)
        
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
        
        # Create 2D LUT after 3D actor is created
        self.lut_2d = self._create_2d_lut()
        self.color_map_2d.SetLookupTable(self.lut_2d)
        
        self.renderer_3d.AddActor(self.vtk_actor)
        
        initial_radius = self.fan_radius_slider.value()
        self.probe_actor = self._create_probe_actor(initial_radius)
        self.renderer_3d.AddActor(self.probe_actor)
        
        self.probe_transform.Identity()
        self.probe_transform.Translate(list(self.vtk_actor.GetCenter()))
        self._update_probe_view()

        self.renderer_3d.ResetCamera(); self.renderer_3d.ResetCameraClippingRange()
        self.vtkWidget_3d.GetRenderWindow().Render()
        # Update 2D slice after loading
        self._update_2d_slice()
    
    def _populate_labels_list(self):
        self.labels_list_widget.clear()
        if not self.display_mesh:
            return

        cell_data = self.display_mesh.GetCellData()
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
        self.probe_updated.emit()

    def _handle_probe_update_gui(self):
        if not self.probe_actor: return
        final_matrix = self.probe_transform.GetMatrix()
        pos = [final_matrix.GetElement(i, 3) for i in range(3)]
        pos_str = f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        self.probe_info_label.setText(pos_str)
        self.vtkWidget_3d.GetRenderWindow().Render()

    def _set_standard_view(self, view_name):
        if not self.mesh_handler or not self.obb_tree:
            print("Please load a mesh first.")
            return

        print(f"Setting standard view to: {view_name}")
        self.current_standard_view = view_name

        try:
            # This is the anatomically correct plane definition from mesh_lib
            view_data, landmarks = self.mesh_handler.cardiac_out_points[view_name]
            normal, internal_origin = view_data
        except (KeyError, AttributeError) as e:
            print(f"Could not get data for view {view_name}: {e}. Check if landmarks were found.")
            self.current_standard_view = None
            return

        # --- Part 1: Find the probe's 3D position from the generated 2D slice contour ---
        # This ensures the 3D probe tip is always on the edge of the 2D slice it generates.
        
        # Create a temporary cutter to get the slice contour without affecting the main pipeline yet.
        temp_plane = vtk.vtkPlane()
        temp_plane.SetOrigin(internal_origin)
        temp_plane.SetNormal(normal)
        temp_cutter = vtk.vtkCutter()
        temp_cutter.SetCutFunction(temp_plane)
        temp_cutter.SetInputData(self.display_mesh)
        temp_cutter.Update()
        slice_contour_polydata = temp_cutter.GetOutput()
        
        surface_origin = None
        probe_y_dir = None

        if slice_contour_polydata.GetNumberOfPoints() == 0:
            print(f"Warning: No intersection found for {view_name}. Probe position may be incorrect.")
            surface_origin = internal_origin
            probe_y_dir = -np.array(normal) # Fallback direction
        else:
            contour_points = vtk_to_numpy(slice_contour_polydata.GetPoints().GetData())
            
            # --- New Logic: Find the point on the contour closest to the LV Apex ---
            
            # 1. Get the LV Apex coordinate from the landmarks for the current view
            lv_apex = None
            if view_name in ["A2CH", "A3CH"]:
                lv_apex = landmarks[1]
            elif view_name in ["A4CH", "PLAX"]:
                lv_apex = landmarks[2]
            
            if lv_apex is not None:
                # 2. Find the contour point with the minimum distance to the LV Apex
                distances_to_apex = np.linalg.norm(contour_points - lv_apex, axis=1)
                surface_origin = contour_points[np.argmin(distances_to_apex)]
                
                # 3. The probe's fan should point from the new pivot (surface_origin) towards the plane's center
                probe_y_dir = np.array(internal_origin) - surface_origin
            else: # For non-apical views like PSAX
                # A simple heuristic: place the probe at the top-most point of the slice contour.
                up_vector = np.array([0, 0, 1]) # Global Z-axis as "up"
                projections = np.dot(contour_points, up_vector)
                surface_origin = contour_points[np.argmax(projections)]
                # The probe fan points into the heart, opposite to the plane normal.
                probe_y_dir = -np.array(normal)

        # --- Part 2: Calculate and set the 3D probe's visual transform ---
        probe_x = np.array(normal)
        probe_x = probe_x / np.linalg.norm(probe_x)

        # Make probe_y orthogonal to probe_x (Gram-Schmidt)
        probe_y = probe_y_dir - np.dot(probe_y_dir, probe_x) * probe_x
        probe_y = probe_y / np.linalg.norm(probe_y)

        # probe_z completes the right-handed coordinate system
        probe_z = np.cross(probe_x, probe_y)
        
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, probe_x[i])
            matrix.SetElement(i, 1, probe_y[i])
            matrix.SetElement(i, 2, probe_z[i])
            matrix.SetElement(i, 3, surface_origin[i])
        
        # --- Part 3: Update the main transform and trigger a full update ---
        self.probe_transform.SetMatrix(matrix)
        self._update_probe_view()

    def _update_2d_slice(self):
        if self.display_mesh is None or not self.probe_actor:
            self.image_actor_2d.SetInputData(self.empty_image_data)
            cam2d = self.renderer_2d.GetActiveCamera()
            if cam2d: cam2d.SetParallelScale(128) # Default zoom
            self.vtkWidget_2d.GetRenderWindow().Render()
            return

        # --- Step 1: Define Grid in Probe's Local Space ---
        h, w = 192, 192 # Reduced resolution for performance
        radius = self.fan_radius_slider.value()

        # Always create a rectangular grid. The cone effect will be a post-processing mask.
        y_coords = np.linspace(0, radius, h)
        # The width of the slice should be proportional to the depth (radius)
        z_coords = np.linspace(-radius, radius, w)

        spacing_y = y_coords[1] - y_coords[0] if h > 1 else 1
        spacing_z = z_coords[1] - z_coords[0] if w > 1 else 1

        zz, yy = np.meshgrid(z_coords, y_coords)
        # Our slice plane is the YZ plane in the probe's local coordinates (X=0)
        grid_points_local = np.vstack([np.zeros(w*h), yy.ravel(), zz.ravel()]).T

        # --- Step 2: Transform Grid to World Space ---
        vtk_points_local = vtk.vtkPoints()
        vtk_points_local.SetData(numpy_to_vtk(grid_points_local, deep=True))
        polydata_local = vtk.vtkPolyData()
        polydata_local.SetPoints(vtk_points_local)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(self.probe_transform)
        transform_filter.SetInputData(polydata_local)
        transform_filter.Update()
        
        grid_in_world_space = transform_filter.GetOutput()

        # --- Step 3: Determine which points are inside the mesh ---
        select_enclosed = vtk.vtkSelectEnclosedPoints()
        select_enclosed.SetInputData(grid_in_world_space)
        select_enclosed.SetSurfaceData(self.display_mesh)
        select_enclosed.Update()

        # --- Step 4: Create Image from the inside/outside mask ---
        is_inside = vtk_to_numpy(select_enclosed.GetOutput().GetPointData().GetArray("SelectedPoints"))
        slice_arr = is_inside.reshape(h, w)

        # --- Step 5: Apply Cone & Colors ---
        # Now, we only care about inside (1) vs outside (0)
        # 0 -> background, 1 -> cone, 2 -> tissue
        final_arr = np.zeros_like(slice_arr, dtype=np.uint8)
        
        if self.apply_cone_checkbox.isChecked():
            # Create a fan-shaped mask for the cone effect
            angle_rad = np.pi * 2/3 # 120 degrees
            yy_mask, xx_mask = np.mgrid[:h, :w]
            origin_x, origin_y = w / 2.0, 0.0
            angle = np.arctan2(yy_mask - origin_y, xx_mask - origin_x)
            angle_mask = (angle > np.pi/2 - angle_rad/2) & (angle < np.pi/2 + angle_rad/2)
            
            # Tissue (value 2) is where the slice is inside the mesh AND inside the cone
            final_arr[(slice_arr == 1) & (angle_mask)] = 2
            # Speckle (value 1) is where the slice is outside the mesh BUT inside the cone
            final_arr[(slice_arr == 0) & (angle_mask)] = 1
        else:
            # No cone, just show the tissue
            final_arr[slice_arr == 1] = 2

        # --- Step 6: Render Image ---
        data_string = final_arr.astype(np.uint8).tobytes()
        self.data_importer_2d.CopyImportVoidPointer(data_string, len(data_string))
        self.data_importer_2d.SetDataScalarTypeToUnsignedChar()
        self.data_importer_2d.SetNumberOfScalarComponents(1)
        self.data_importer_2d.SetDataExtent(0, w - 1, 0, h - 1, 0, 0)
        self.data_importer_2d.SetWholeExtent(0, w - 1, 0, h - 1, 0, 0)
        # Set correct spacing to fix aspect ratio
        self.data_importer_2d.SetDataSpacing(spacing_z, spacing_y, 1.0)
        
        self.color_map_2d.Update()
        self.image_actor_2d.SetInputData(self.color_map_2d.GetOutput())

        # --- Set Camera for Top Pivot ---
        cam2d = self.renderer_2d.GetActiveCamera()
        cam2d.ParallelProjectionOn()
        # Center camera on the image
        cam2d.SetPosition(w / 2.0 * spacing_z, h / 2.0 * spacing_y, 1)
        cam2d.SetFocalPoint(w / 2.0 * spacing_z, h / 2.0 * spacing_y, 0)
        # Flip Y-axis to show fan pivot at the top
        cam2d.SetViewUp(0, -1, 0) 
        self.renderer_2d.ResetCamera()
        # Set zoom to fit the image height
        cam2d.SetParallelScale(h / 2.0 * spacing_y)

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
