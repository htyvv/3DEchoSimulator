# ui_setup.py

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QPushButton, QSlider, QLabel, QGroupBox, QGridLayout, QFrame,
    QVBoxLayout, QComboBox, QWidget
)
from PySide6.QtGui import QAction


def setup_menu_bar(main_window):
    menu_bar = main_window.menuBar()
    file_menu = menu_bar.addMenu("&File")
    open_action = QAction("&Open...", main_window)
    open_action.triggered.connect(main_window._open_file_dialog)
    file_menu.addAction(open_action)
    exit_action = QAction("&Exit", main_window)
    exit_action.triggered.connect(main_window.close)
    file_menu.addAction(exit_action)

def setup_viewer_panel(main_window):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    three_d_view_frame = QFrame(); three_d_view_frame.setFrameShape(QFrame.StyledPanel)
    three_d_view_layout = QVBoxLayout(three_d_view_frame)
    three_d_view_layout.addWidget(main_window.vtkWidget_3d)
    
    two_d_view_frame = QFrame(); two_d_view_frame.setFrameShape(QFrame.StyledPanel)
    two_d_view_layout = QVBoxLayout(two_d_view_frame)
    two_d_view_layout.addWidget(main_window.vtkWidget_2d)
    
    layout.addWidget(three_d_view_frame, 7)
    layout.addWidget(two_d_view_frame, 3)
    return panel

def setup_options_panel(main_window):
    panel = QWidget()
    layout = QVBoxLayout(panel)
    
    file_group = QGroupBox("File Info")
    file_layout = QVBoxLayout()
    main_window.file_info_label = QLabel("No file loaded.")
    file_layout.addWidget(main_window.file_info_label)
    file_group.setLayout(file_layout)
    layout.addWidget(file_group)

    main_window.animation_group = QGroupBox("Animation Control")
    animation_layout = QGridLayout()
    main_window.play_pause_button = QPushButton("Play")
    main_window.timeline_slider = QSlider(Qt.Horizontal)
    main_window.frame_label = QLabel("Frame: 0 / 0")
    animation_layout.addWidget(main_window.play_pause_button, 0, 0)
    animation_layout.addWidget(main_window.frame_label, 0, 1)
    animation_layout.addWidget(main_window.timeline_slider, 1, 0, 1, 2)
    main_window.animation_group.setLayout(animation_layout)
    main_window.animation_group.setEnabled(False)
    layout.addWidget(main_window.animation_group)

    vis_group = QGroupBox("Visualization Control")
    vis_layout = QGridLayout()
    vis_layout.addWidget(QLabel("Heart Color:"), 0, 0)
    main_window.heart_color_button = QPushButton("Select")
    vis_layout.addWidget(main_window.heart_color_button, 0, 1)
    vis_layout.addWidget(QLabel("Heart Opacity:"), 1, 0)
    main_window.heart_opacity_slider = QSlider(Qt.Horizontal)
    main_window.heart_opacity_slider.setRange(0, 100); main_window.heart_opacity_slider.setValue(100)
    vis_layout.addWidget(main_window.heart_opacity_slider, 1, 1)
    vis_layout.addWidget(QLabel("Fan Radius:"), 2, 0)
    main_window.fan_radius_slider = QSlider(Qt.Horizontal)
    main_window.fan_radius_slider.setRange(40, 200); main_window.fan_radius_slider.setValue(80)
    vis_layout.addWidget(main_window.fan_radius_slider, 2, 1)
    vis_layout.addWidget(QLabel("Fan Plane Color:"), 3, 0)
    main_window.fan_color_button = QPushButton("Select")
    vis_layout.addWidget(main_window.fan_color_button, 3, 1)
    vis_layout.addWidget(QLabel("Fan Plane Opacity:"), 4, 0)
    main_window.fan_opacity_slider = QSlider(Qt.Horizontal)
    main_window.fan_opacity_slider.setRange(0, 100); main_window.fan_opacity_slider.setValue(50)
    vis_layout.addWidget(main_window.fan_opacity_slider, 4, 1)
    vis_layout.addWidget(QLabel("Render Mode:"), 5, 0)
    main_window.render_mode_combo = QComboBox()
    main_window.render_mode_combo.addItems(["Surface", "Wireframe", "Points"])
    vis_layout.addWidget(main_window.render_mode_combo, 5, 1)
    vis_group.setLayout(vis_layout)
    layout.addWidget(vis_group)
    
    cam_save_group = QGroupBox("Camera & Save")
    cam_save_layout = QGridLayout(cam_save_group)
    main_window.reset_camera_button = QPushButton("Reset Camera")
    main_window.save_screenshot_button = QPushButton("Save Screenshot")
    main_window.save_slice_button = QPushButton("Save 2D Slice")
    cam_save_layout.addWidget(main_window.reset_camera_button, 0, 0)
    cam_save_layout.addWidget(main_window.save_screenshot_button, 0, 1)
    cam_save_layout.addWidget(main_window.save_slice_button, 0, 2)
    cam_save_layout.addWidget(QLabel("Sensitivity:"), 1, 0)
    main_window.sensitivity_slider = QSlider(Qt.Horizontal)
    main_window.sensitivity_slider.setRange(10, 200); main_window.sensitivity_slider.setValue(85)
    cam_save_layout.addWidget(main_window.sensitivity_slider, 1, 1, 1, 2)
    layout.addWidget(cam_save_group)
    
    probe_group = QGroupBox("Probe Info")
    probe_layout = QVBoxLayout()
    main_window.probe_info_label = QLabel("Position: [N/A]\nTilt: 0.0\nRock: 0.0\nRotate: 0.0")
    probe_layout.addWidget(main_window.probe_info_label)
    probe_group.setLayout(probe_layout)
    layout.addWidget(probe_group)
    
    layout.addStretch(1)
    return panel