import sys
import vtk
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QFileDialog, QColorDialog, QHBoxLayout
)
from PySide6.QtCore import Qt, Signal

from ui_setup import setup_menu_bar, setup_viewer_panel, setup_options_panel
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

class MainWindow(QMainWindow):
    # 정보창을 단순화했으므로 각도(angles) 정보는 다시 제외
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
        self.setWindowTitle("EchoCardiacSimulator")
        self.setGeometry(100, 100, 1600, 900)
        
        self.vtk_actor = None; self.probe_actor = None; self.wedge_actor = None
        self.camera_style = None; self.picker = vtk.vtkCellPicker(); self.picker.SetTolerance(0.005)
        self.probe_transform = vtk.vtkTransform()
        self.is_planting_mode = False

        self.cutting_plane = vtk.vtkPlane(); self.cutter = vtk.vtkCutter(); self.cut_actor = vtk.vtkActor()

        self.vtkWidget_3d = QVTKRenderWindowInteractor()
        self.vtkWidget_2d = QVTKRenderWindowInteractor()
        setup_menu_bar(self)
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        viewer_panel = setup_viewer_panel(self); options_panel = setup_options_panel(self)
        self.main_layout.addWidget(viewer_panel, 3); self.main_layout.addWidget(options_panel, 1)
        
        self._setup_3d_view()
        self._setup_2d_view()
        self._connect_signals()
        self._update_planting_mode_visuals()

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
        # [추가] 마우스 이동 이벤트를 감지하는 옵저버
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
        self.renderer_2d.GetActiveCamera().ParallelProjectionOn()
        self.cutter.SetCutFunction(self.cutting_plane)
        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputConnection(self.cutter.GetOutputPort())
        cut_mapper = vtk.vtkPolyDataMapper()
        cut_mapper.SetInputConnection(triangulator.GetOutputPort())
        self.cut_actor.SetMapper(cut_mapper)
        self.cut_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.renderer_2d.AddActor(self.cut_actor)

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

    # [추가] 마우스 이동 시 실시간으로 프로브를 위치시키는 핸들러
    def _mouse_move_handler(self, interactor, event):
        if self.is_planting_mode:
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

    # [수정] 마우스 클릭은 Plant 모드를 끄고 위치를 확정하는 역할만 수행
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

        self.file_info_label.setText(f"Loaded: {file_path.split('/')[-1]}")
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.final_data); mapper.ScalarVisibilityOff()
        self.vtk_actor = vtk.vtkActor(); self.vtk_actor.SetMapper(mapper)
        self.vtk_actor.GetProperty().SetColor(155/255.0, 17/255.0, 30/255.0)
        self.renderer_3d.AddActor(self.vtk_actor)
        
        self.cutter.SetInputData(self.final_data)
        
        initial_radius = self.fan_radius_slider.value()
        self.probe_actor = self._create_probe_actor(initial_radius)
        self.renderer_3d.AddActor(self.probe_actor)
        
        self.probe_transform.Identity()
        self.probe_transform.Translate(list(self.vtk_actor.GetCenter()))
        self._update_probe_view()

        self.renderer_3d.ResetCamera(); self.renderer_3d.ResetCameraClippingRange()
        self.vtkWidget_3d.GetRenderWindow().Render()
    
    def get_poly_data_from_reader(self, reader):
        data_type = reader.GetOutput().GetClassName()
        if data_type in ['vtkImageData', 'vtkStructuredPoints']:
            contour=vtk.vtkContourFilter(); contour.SetInputData(reader.GetOutput()); contour.SetValue(0, 128); contour.Update()
            return contour.GetOutput()
        elif data_type == 'vtkUnstructuredGrid':
            surface=vtk.vtkDataSetSurfaceFilter(); surface.SetInputData(reader.GetOutput()); surface.Update()
            return surface.GetOutput()
        elif data_type == 'vtkPolyData':
            return reader.GetOutput()
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
        # Hover 시에 계속 변하는 각도 값은 혼란을 주므로 위치만 표시
        pos_str = f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        self.probe_info_label.setText(pos_str)
        self.vtkWidget_3d.GetRenderWindow().Render()

    def _update_2d_slice(self, pos, cutting_normal):
        self.cutting_plane.SetOrigin(pos)
        self.cutting_plane.SetNormal(cutting_normal)
        cam2d = self.renderer_2d.GetActiveCamera()
        cam2d.SetFocalPoint(pos)
        cam2d.SetPosition(pos[0] + cutting_normal[0], pos[1] + cutting_normal[1], pos[2] + cutting_normal[2])
        if abs(cutting_normal[1]) > 0.99: cam2d.SetViewUp(0, 0, 1)
        else: cam2d.SetViewUp(0, 1, 0)
        self.renderer_2d.ResetCameraClippingRange()
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
    app.window = MainWindow()
    app.window.show()
    sys.exit(app.exec())