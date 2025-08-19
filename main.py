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
    probe_updated = Signal(list, list, list)

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
        self.probe_pos = [0,0,0]; self.probe_surface_normal = [0,1,0]
        self.probe_tilt = 0.0; self.probe_rock = 0.0; self.probe_rotate = 0.0
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

    def _setup_3d_view(self):
        self.renderer_3d = vtk.vtkRenderer()
        self.vtkWidget_3d.GetRenderWindow().AddRenderer(self.renderer_3d)
        self.vtkWidget_3d.GetRenderWindow().SetMultiSamples(8)
        self.interactor_3d = self.vtkWidget_3d.GetRenderWindow().GetInteractor()
        self.renderer_3d.SetBackground(0.1, 0.2, 0.4)
        self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.camera_style.SetKeyPressActivation(False)
        self.interactor_3d.SetInteractorStyle(self.camera_style)
        self.interactor_3d.AddObserver("KeyPressEvent", self._key_press_handler, 1.0)
        self.interactor_3d.AddObserver("LeftButtonPressEvent", self._left_button_press_handler, 1.0)
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
        # 라디오 버튼이 없어졌으므로 관련 시그널도 제거

    def _left_button_press_handler(self, interactor, event):
        if interactor.GetKeySym() == self.KEY_CONFIG['PLANT_PROBE']:
            mouse_pos = interactor.GetEventPosition()
            self.picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer_3d)
            if self.picker.GetCellId() >= 0 and self.picker.GetActor() == self.vtk_actor:
                self.probe_pos = list(self.picker.GetPickPosition())
                self.probe_surface_normal = list(self.picker.GetPickNormal())
                self.probe_tilt = 0.0; self.probe_rock = 0.0; self.probe_rotate = 0.0
                self._update_probe_transform()

    def _key_press_handler(self, interactor, event):
        if not self.probe_actor: return
        key = interactor.GetKeySym()
        if key == self.KEY_CONFIG['TILT_UP']: self.probe_tilt -= self.SENSITIVITY['TILT_ROCK']
        elif key == self.KEY_CONFIG['TILT_DOWN']: self.probe_tilt += self.SENSITIVITY['TILT_ROCK']
        elif key == self.KEY_CONFIG['ROCK_LEFT']: self.probe_rock -= self.SENSITIVITY['TILT_ROCK']
        elif key == self.KEY_CONFIG['ROCK_RIGHT']: self.probe_rock += self.SENSITIVITY['TILT_ROCK']
        elif key == self.KEY_CONFIG['ROTATE_CCW']: self.probe_rotate -= self.SENSITIVITY['ROTATE']
        elif key == self.KEY_CONFIG['ROTATE_CW']: self.probe_rotate += self.SENSITIVITY['ROTATE']
        elif key == self.KEY_CONFIG['SLIDE_UP']:
            transform_matrix = self.probe_actor.GetUserMatrix(); direction = [transform_matrix.GetElement(i, 1) for i in range(3)]; vtk.vtkMath.Normalize(direction)
            for i in range(3): self.probe_pos[i] += direction[i] * self.SENSITIVITY['SLIDE']
        elif key == self.KEY_CONFIG['SLIDE_DOWN']:
            transform_matrix = self.probe_actor.GetUserMatrix(); direction = [transform_matrix.GetElement(i, 1) for i in range(3)]; vtk.vtkMath.Normalize(direction)
            for i in range(3): self.probe_pos[i] -= direction[i] * self.SENSITIVITY['SLIDE']
        elif key == self.KEY_CONFIG['SLIDE_LEFT']:
            transform_matrix = self.probe_actor.GetUserMatrix(); direction = [transform_matrix.GetElement(i, 2) for i in range(3)]; vtk.vtkMath.Normalize(direction)
            for i in range(3): self.probe_pos[i] -= direction[i] * self.SENSITIVITY['SLIDE']
        elif key == self.KEY_CONFIG['SLIDE_RIGHT']:
            transform_matrix = self.probe_actor.GetUserMatrix(); direction = [transform_matrix.GetElement(i, 2) for i in range(3)]; vtk.vtkMath.Normalize(direction)
            for i in range(3): self.probe_pos[i] += direction[i] * self.SENSITIVITY['SLIDE']
        self._update_probe_transform()

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
        # stl_path = r"C:\Users\Taeyeong\Desktop\Scan_guide\3DEchoSimulator\asset\collection-of-ultrasound-probes-1.snapshot.4\Linear Transducer by Qadamian.STL"
        # self.probe_actor = self._create_probe_actor_from_stl(stl_path, initial_radius)
        self.renderer_3d.AddActor(self.probe_actor)
        self.probe_pos = self.vtk_actor.GetCenter()
        self.probe_surface_normal = [0, 1, 0]
        self._update_probe_transform()
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
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(5)
        cylinder.SetHeight(14)
        cylinder.SetResolution(20)
        cylinder.SetCenter(0, -7, 0)   # 윗면이 y=0 → 부채꼴 팁과 접점 일치

        cyl_mapper = vtk.vtkPolyDataMapper()
        cyl_mapper.SetInputConnection(cylinder.GetOutputPort())
        cyl_actor = vtk.vtkActor()
        cyl_actor.SetMapper(cyl_mapper)
        cyl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)

        # 부채꼴(팬): 로컬 팁을 (0,0,0)에 두고 X방향으로 얇게 두께를 준 형태(Extrude)
        points = vtk.vtkPoints()
        tip_x, tip_y = 0, 0
        fan_len = radius
        fan_width = radius
        points.InsertNextPoint(tip_x, tip_y, 0)
        points.InsertNextPoint(tip_x, tip_y + fan_len, -fan_width / 2)
        points.InsertNextPoint(tip_x, tip_y + fan_len,  fan_width / 2)

        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, 0)
        triangle.GetPointIds().SetId(1, 1)
        triangle.GetPointIds().SetId(2, 2)

        triangles = vtk.vtkCellArray()
        triangles.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputData(polydata)
        extrude.SetExtrusionTypeToVectorExtrusion()
        vec = [2, 0, 0]  # X방향 얇은 두께
        extrude.SetVector(*vec)

        wedge_mapper = vtk.vtkPolyDataMapper()
        wedge_mapper.SetInputConnection(extrude.GetOutputPort())
        self.wedge_actor = vtk.vtkActor()
        self.wedge_actor.SetMapper(wedge_mapper)
        self.wedge_actor.GetProperty().SetColor(0.2, 1.0, 0.2)
        self.wedge_actor.GetProperty().SetOpacity(0.5)
        self.wedge_actor.SetPosition(-vec[0]/2.0, 0, 0)  # 두께 중심 정렬

        assembly = vtk.vtkAssembly()
        assembly.AddPart(cyl_actor)
        assembly.AddPart(self.wedge_actor)

        # 디버그용 접점 시각화 (선택)
        sphere = vtk.vtkSphereSource(); sphere.SetRadius(2.0)
        sm = vtk.vtkPolyDataMapper(); sm.SetInputConnection(sphere.GetOutputPort())
        pivot = vtk.vtkActor(); pivot.SetMapper(sm); pivot.GetProperty().SetColor(1,1,0)
        assembly.AddPart(pivot)

        return assembly
    
    def _create_probe_actor_from_stl(self, stl_path, radius):
        # STL 읽기
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        # Transform: Z축을 Y축으로, 팁을 원점으로
        transform = vtk.vtkTransform()
        # 팁을 (0,0,0)으로 이동
        bounds = reader.GetOutput().GetBounds()
        z_min = bounds[4]
        z_max = bounds[5]
        transform.Translate(0, 0, -z_max)
        # Z축을 Y축으로 회전
        transform.RotateX(-90)  
        # transform.RotateY(90)  

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(reader.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        # STL → Mapper → Actor
        stl_mapper = vtk.vtkPolyDataMapper()
        stl_mapper.SetInputConnection(reader.GetOutputPort())

        stl_actor = vtk.vtkActor()
        stl_actor.SetMapper(stl_mapper)
        stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # probe 색상
        # 필요하다면 크기 조절
        stl_actor.SetScale(0.4, 0.4, 0.4)

        # wedge(팬) 부분 그대로 재사용
        # wedge_actor = self._create_wedge_actor(radius)

        assembly = vtk.vtkAssembly()
        assembly.AddPart(stl_actor)
        # assembly.AddPart(wedge_actor)

        # self.wedge_actor = wedge_actor
        self.wedge_actor = None
        return assembly

    def _update_probe_shape(self, value):
        if not self.probe_actor or not self.renderer_3d: return
        self.renderer_3d.RemoveActor(self.probe_actor)
        self.probe_actor = self._create_probe_actor(value)
        self.renderer_3d.AddActor(self.probe_actor)
        self._update_probe_transform()

    def _update_probe_transform(self):
        if not self.probe_actor:
            return

        t = vtk.vtkTransform()
        t.PostMultiply()  # 호출 순서대로 적용(회전들을 먼저, 이동은 마지막에)

        # 1) 표면 법선에 맞추는 기본 정렬 (로컬 Y=[0,1,0]을 표면 내향(inward_normal)으로 보냄)
        start_vec = [0, 1, 0]
        inward_normal = [-self.probe_surface_normal[0],
                        -self.probe_surface_normal[1],
                        -self.probe_surface_normal[2]]
        if vtk.vtkMath.Norm(inward_normal) < 1e-8:
            inward_normal = [0, 1, 0]
        vtk.vtkMath.Normalize(inward_normal)

        axis = [0, 0, 0]
        vtk.vtkMath.Cross(start_vec, inward_normal, axis)
        dot = max(-1.0, min(1.0, vtk.vtkMath.Dot(start_vec, inward_normal)))
        if vtk.vtkMath.Norm(axis) < 1e-8:
            # 평행/반평행 처리
            angle_deg = 0.0 if dot > 0 else 180.0
            axis = [1, 0, 0]  # 아무 직교축
        else:
            angle_deg = vtk.vtkMath.DegreesFromRadians(math.acos(dot))

        t.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])

        # 2) 로컬 회전: 접점(로컬 원점)을 축으로 tilt(Z), rock(X), rotate(Y)
        t.RotateZ(self.probe_tilt)    # Tilt: 팬을 위/아래로 젖힘
        t.RotateX(self.probe_rock)    # Rock: 좌/우로 기울임
        t.RotateY(self.probe_rotate)  # Rotate: 실린더 장축(Y)을 중심으로 회전

        # 3) 마지막에 접점을 원하는 월드 위치로 이동
        t.Translate(self.probe_pos)

        self.probe_actor.SetUserMatrix(t.GetMatrix())

        final_matrix = self.probe_actor.GetUserMatrix()
        cutting_normal = [final_matrix.GetElement(i, 0) for i in range(3)]  # 로컬 X축을 절단면 노멀로 유지
        angles = [self.probe_tilt, self.probe_rock, self.probe_rotate]
        self.probe_updated.emit(self.probe_pos, angles, cutting_normal)

    def _handle_probe_update_gui(self, pos, angles, cutting_normal):
        pos_str=f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        tilt_str=f"Tilt: {angles[0]:.1f}"; rock_str=f"Rock: {angles[1]:.1f}"; rotate_str=f"Rotate: {angles[2]:.1f}"
        self.probe_info_label.setText(f"{pos_str}\n{tilt_str}\n{rock_str}\n{rotate_str}")
        self.vtkWidget_3d.GetRenderWindow().Render()

    def _update_2d_slice(self, pos, angles, cutting_normal):
        self.cutting_plane.SetOrigin(pos)
        self.cutting_plane.SetNormal(cutting_normal)
        cam2d=self.renderer_2d.GetActiveCamera()
        cam2d.SetFocalPoint(pos)
        cam2d.SetPosition(pos[0]+cutting_normal[0], pos[1]+cutting_normal[1], pos[2]+cutting_normal[2])
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