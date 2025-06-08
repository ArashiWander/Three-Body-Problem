import pygame
import math
import numpy as np
import pygame.gfxdraw
import logging
import argparse
import os
import time
from collections import deque

# 依赖项检查与导入
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: 未找到 Matplotlib。")
    MATPLOTLIB_AVAILABLE = False
    cm, plt, FigureCanvasAgg = None, None, None

try:
    import pygame_gui
    from pygame_gui.elements import UIButton, UILabel, UIPanel, UIWindow, UIDropDownMenu, UIHorizontalSlider, UITextBox
    PYGAME_GUI_AVAILABLE = True
except ImportError:
     print("错误: 未找到 pygame_gui。请安装: pip install pygame_gui")
     PYGAME_GUI_AVAILABLE = False
     exit()

# 内部模块导入
from .constants import *
from . import constants as C
from . import __version__
from .utils import mass_to_display, distance_to_display, time_to_display
from .presets import PRESETS, PRESET_SOFTENING_LENGTHS
from .rendering import Body, render_gravitational_field
from .physics_utils import (
    step_simulation, calculate_center_of_mass, 
    detect_and_handle_collisions, get_world_bounds_sim
)
from .analysis import calculate_orbital_elements, EnergyMonitor

# --- 主模拟函数 ---
def main(softening_length_override=None):
    """主模拟循环。"""
    global SHOW_TRAILS, SHOW_GRAV_FIELD, ADAPTIVE_STEPPING, SPEED_FACTOR

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"N体模拟 v{__version__} (研究版)")
    clock = pygame.time.Clock()
    ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT), "theme.json")

    # --- UI 元素创建 ---
    control_panel_rect = pygame.Rect((WIDTH - UI_SIDEBAR_WIDTH, 0), (UI_SIDEBAR_WIDTH, HEIGHT))
    control_panel = pygame_gui.elements.UIPanel(relative_rect=control_panel_rect, manager=ui_manager, object_id="#control_panel")
    
    y_pos = 10
    # ... (此处省略了大部分UI元素的创建代码以保持简洁, 它们与之前版本类似)
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text="N体模拟器", manager=ui_manager, container=control_panel, object_id='#title_label'); y_pos += 40
    
    # --- 新增研究级UI元素 ---
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (80, 30)), text="积分器:", manager=ui_manager, container=control_panel); y_pos += 30
    integrator_dropdown = UIDropDownMenu(options_list=['Symplectic', 'RK4'], starting_option='Symplectic', relative_rect=pygame.Rect((20, y_pos), (UI_SIDEBAR_WIDTH - 40, 30)), manager=ui_manager, container=control_panel); y_pos += 40
    
    gr_button = UIButton(relative_rect=pygame.Rect((20, y_pos), (UI_SIDEBAR_WIDTH - 40, 30)), text="广义相对论修正: 关", manager=ui_manager, container=control_panel); y_pos += 50

    # --- 选中天体的数据窗口 ---
    data_window_rect = pygame.Rect(10, 10, 280, 250)
    data_window = UIWindow(rect=data_window_rect, manager=ui_manager, window_display_title="轨道数据", visible=False)
    data_labels = {}
    label_texts = ["半长轴:", "偏心率:", "近日点:", "远日点:", "轨道周期:", "速度:"]
    for i, text in enumerate(label_texts):
        data_labels[text] = UILabel(relative_rect=pygame.Rect(10, i * 35, 260, 30), text=f"{text} N/A", manager=ui_manager, container=data_window)

    # --- 能量监控图 ---
    energy_plot_rect = pygame.Rect(0, HEIGHT - UI_BOTTOM_HEIGHT, WIDTH - UI_SIDEBAR_WIDTH, UI_BOTTOM_HEIGHT)
    energy_plot_surface = pygame.Surface(energy_plot_rect.size, pygame.SRCALPHA)
    
    # --- 状态变量 ---
    state = {
        "bodies": [], "current_preset_name": "太阳 & 地球", "time_step": TIME_STEP_BASE,
        "paused": False, "running": True, "use_boundaries": True, "selected_body": None,
        "dragging_body": False, "dragging_camera": False, "mouse_offset": np.zeros(2),
        "camera_drag_start_screen": np.zeros(2), "camera_drag_start_pan": np.zeros(2),
        "current_zoom": ZOOM_BASE, "current_pan": INITIAL_PAN_OFFSET.copy(),
        "target_zoom": ZOOM_BASE, "target_pan": INITIAL_PAN_OFFSET.copy(),
        "simulation_time": 0.0, "next_body_mass": DEFAULT_NEXT_BODY_MASS,
        "next_body_radius_pixels": DEFAULT_NEXT_BODY_RADIUS_PIXELS,
        "frame_times": deque(maxlen=60),
        "color_options": [EARTH_COLOR, MARS_COLOR, VENUS_COLOR, MERCURY_COLOR, GAS_COLOR, ICE_COLOR],
        "color_index": 0, "adding_body_state": 0, "add_body_start_screen": np.zeros(2),
        "gravity_multiplier": 1.0, "integrator_type": 'Symplectic', "use_gr_correction": False,
        "energy_monitor": EnergyMonitor(), "last_status_update": 0
    }

    def load_preset(preset_name):
        # ... (与之前版本相同, 此处省略)
        pass
    
    def process_input_events():
        # ... (与之前版本类似，但增加了对新UI元素的处理)
        nonlocal state
        if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == integrator_dropdown:
                state['integrator_type'] = event.text
        elif event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == gr_button:
                state['use_gr_correction'] = not state['use_gr_correction']
                gr_button.set_text(f"广义相对论修正: {'开' if state['use_gr_correction'] else '关'}")
        
        # 选择天体时显示/隐藏数据窗口
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # ...
            if clicked_on_body:
                state['selected_body'] = clicked_on_body
                data_window.show()
                data_window.set_display_title(f"轨道数据: {state['selected_body'].name}")
            else:
                state['selected_body'] = None
                data_window.hide()

    def update_physics():
        if not state['paused'] and state['bodies']:
            current_g = INITIAL_G * state['gravity_multiplier']
            step_simulation(
                state['bodies'], state['time_step'] * SPEED_FACTOR, current_g,
                state['integrator_type'], state['use_gr_correction']
            )
            state['simulation_time'] += state['time_step'] * SPEED_FACTOR
            # ... (碰撞检测)
    
    def update_analysis_and_ui():
        # 更新轨道数据窗口
        if state['selected_body'] and data_window.visible:
            central_body = max((b for b in state['bodies'] if b != state['selected_body']), key=lambda b: b.mass, default=None)
            if central_body:
                elements = calculate_orbital_elements(state['selected_body'], central_body)
                data_labels["半长轴:"].set_text(f"半长轴: {distance_to_display(elements['semi_major_axis'])}")
                data_labels["偏心率:"].set_text(f"偏心率: {elements['eccentricity']:.4f}")
                data_labels["近日点:"].set_text(f"近日点: {distance_to_display(elements['periapsis'])}")
                data_labels["远日点:"].set_text(f"远日点: {distance_to_display(elements['apoapsis'])}")
                data_labels["轨道周期:"].set_text(f"轨道周期: {time_to_display(elements['period'])}")
                data_labels["速度:"].set_text(f"速度: {elements['speed']:.2f} m/s")

        # 更新能量图
        state['energy_monitor'].update(state['bodies'], INITIAL_G * state['gravity_multiplier'])
        energy_plot_surface.fill((30, 30, 40, 200))
        state['energy_monitor'].draw(energy_plot_surface)

    def draw_scene():
        screen.fill(BLACK)
        # ... (绘制天体和轨迹)
        screen.blit(energy_plot_surface, energy_plot_rect.topleft)
        ui_manager.draw_ui(screen)
        pygame.display.flip()

    # --- 初始化和主循环 ---
    load_preset(state['current_preset_name'])
    state['energy_monitor'].set_initial_energy(state['bodies'], INITIAL_G)

    while state['running']:
        real_dt = clock.tick(60) / 1000.0
        
        process_input_events()
        
        state['current_zoom'] += (state['target_zoom'] - state['current_zoom']) * CAMERA_SMOOTHING
        state['current_pan'] += (state['target_pan'] - state['current_pan']) * CAMERA_SMOOTHING

        update_physics()
        update_analysis_and_ui()
        
        for body in state['bodies']:
            body.update_trail(state['current_zoom'], state['current_pan'])
        
        draw_scene()

    pygame.quit()

if __name__ == "__main__":
    # ... (入口代码)
    pass

