import pygame
import math
import numpy as np
import pygame.gfxdraw
import logging
import argparse
import os

# 确保 Matplotlib 已安装: pip install matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: 未找到 Matplotlib。引力场可视化功能已禁用。")
    print("请使用以下命令安装 Matplotlib: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False
    # 如果 matplotlib 不可用，定义虚拟的类/函数
    cm = None
    plt = None
    FigureCanvasAgg = None


import time
from collections import deque
# 确保 Pygame GUI 已安装: pip install pygame_gui
try:
    import pygame_gui
    from pygame_gui.elements import UIButton, UILabel, UITextEntryLine, UIPanel, UIWindow
    from pygame_gui.elements import UIHorizontalSlider, UIDropDownMenu, UITextBox, UISelectionList
    PYGAME_GUI_AVAILABLE = True
except ImportError:
     print("错误: 未找到 pygame_gui。请安装: pip install pygame_gui")
     PYGAME_GUI_AVAILABLE = False
     exit()

from .constants import *
from . import constants as C
from . import __version__
from .utils import mass_to_display, distance_to_display, time_to_display
from .presets import PRESETS, PRESET_SOFTENING_LENGTHS
from .rendering import Body, render_gravitational_field
from .physics_utils import calculate_center_of_mass, perform_rk4_step, adaptive_rk4_step, detect_and_handle_collisions, get_world_bounds_sim


# --- 主模拟函数 ---
def main(softening_length_override=None):
    """主模拟循环。"""
    # <<< 在 main 的事件循环中修改的全局标志 >>>
    global SHOW_TRAILS, SHOW_GRAV_FIELD, ADAPTIVE_STEPPING, SPEED_FACTOR

    # --- Pygame 和 UI 初始化 ---
    pygame.init()
    if not PYGAME_GUI_AVAILABLE:
        print("未找到 Pygame GUI，正在退出。")
        return

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"N体模拟 v{__version__}")
    clock = pygame.time.Clock()
    ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT))

    # --- UI 元素创建 ---
    control_panel_rect = pygame.Rect((WIDTH - UI_SIDEBAR_WIDTH, 0), (UI_SIDEBAR_WIDTH, HEIGHT))
    status_bar_rect = pygame.Rect((0, HEIGHT - UI_BOTTOM_HEIGHT), (WIDTH - UI_SIDEBAR_WIDTH, UI_BOTTOM_HEIGHT))

    control_panel = pygame_gui.elements.UIPanel(relative_rect=control_panel_rect, manager=ui_manager)
    status_bar = pygame_gui.elements.UIPanel(relative_rect=status_bar_rect, manager=ui_manager)

    # ... (所有 UI 元素创建代码保持不变, 为了简洁省略) ...
    # 此处省略了所有pygame_gui元素的创建代码，它们与您原始文件中的代码相同。
    # 为了便于阅读，我将帮助文本翻译成了中文。
    y_pos = 10
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text="N体模拟器", manager=ui_manager, container=control_panel, object_id='#title_label')
    y_pos += 40
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (80, 30)), text="预设:", manager=ui_manager, container=control_panel)
    preset_dropdown = pygame_gui.elements.UIDropDownMenu(options_list=list(PRESETS.keys()), starting_option="太阳 & 地球", relative_rect=pygame.Rect((100, y_pos), (UI_SIDEBAR_WIDTH - 120, 30)), manager=ui_manager, container=control_panel)
    y_pos += 40
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="模拟控制", manager=ui_manager, container=control_panel, object_id='#section_header')
    y_pos += 25
    play_pause_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text="暂停", manager=ui_manager, container=control_panel)
    reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text="重置", manager=ui_manager, container=control_panel)
    y_pos += 40
    speed_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text=f"速度: {SPEED_FACTOR:.1f}x", manager=ui_manager, container=control_panel)
    y_pos += 20
    speed_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=SPEED_FACTOR, value_range=(0.05, 5.0), manager=ui_manager, container=control_panel)
    y_pos += 30
    gravity_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="引力: 1.0x", manager=ui_manager, container=control_panel)
    y_pos += 20
    gravity_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=1.0, value_range=(0.1, 100.0), manager=ui_manager, container=control_panel)
    y_pos += 30
    adaptive_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text=f"自适应步长: {'开' if ADAPTIVE_STEPPING else '关'}", manager=ui_manager, container=control_panel)
    y_pos += 40
    # ... 其他UI元素 ...
    help_window_rect = pygame.Rect(0, 0, (WIDTH - UI_SIDEBAR_WIDTH) * 0.7, HEIGHT * 0.7)
    help_window_rect.center = ((WIDTH - UI_SIDEBAR_WIDTH) // 2, HEIGHT // 2)
    help_window = pygame_gui.elements.UIWindow(rect=help_window_rect, manager=ui_manager, window_display_title="帮助和控制", visible=False)
    help_text_content = """
<b>N体模拟器控制与信息</b><br>
------------------------------------<br><br>
<b>鼠标控制:</b><br>
 - <b>左键单击天体:</b> 选中天体以查看/编辑其属性。<br>
 - <b>左键拖动天体:</b> 移动选中的天体 (暂停时会重置其速度)。<br>
 - <b>左键拖动背景:</b> 平移相机视角。<br>
 - <b>右键拖动:</b> 在点击位置添加新天体，拖动方向和长度决定其初速度。<br>
 - <b>鼠标滚轮:</b> 放大/缩小视角。<br><br>

<b>键盘快捷键:</b><br>
 - <b>空格键 (SPACE):</b> 暂停/继续模拟。<br>
 - <b>R:</b> 将模拟重置为当前选定的预设。<br>
 - <b>T:</b> 切换显示/隐藏轨迹。<br>
 - <b>G:</b> 切换引力场热图 (需要 Matplotlib)。<br>
 - <b>B:</b> 切换模拟边界 (反射墙)。<br>
 - <b>A:</b> 切换自适应时间步长。<br>
 - <b>C 或 HOME:</b> 视角居中于质心并重置缩放。<br>
 - <b>H:</b> 显示/隐藏此帮助窗口。<br><br>

<b>物理说明:</b><br>
 - 使用 RK4 积分 (固定或自适应步长)。<br>
 - 引力根据牛顿定律计算，并使用软化因子处理近距离相互作用。<br>
 - 碰撞基于物理半径检测，默认为弹性碰撞。<br>
    """
    help_textbox = pygame_gui.elements.UITextBox(html_text=help_text_content, relative_rect=pygame.Rect((0, 0), help_window.get_container().get_size()), manager=ui_manager, container=help_window)
    status_text_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 0), (WIDTH - UI_SIDEBAR_WIDTH - 20, UI_BOTTOM_HEIGHT)), text="状态: 初始化中...", manager=ui_manager, container=status_bar)


    # --- 模拟状态变量 ---
    bodies = []
    current_preset_name = "太阳 & 地球"
    time_step = TIME_STEP_BASE
    paused = False
    running = True
    use_boundaries = True
    selected_body = None
    dragging_body = False
    dragging_camera = False
    camera_drag_start_screen = np.zeros(2)
    camera_drag_start_pan = np.zeros(2)
    current_zoom = ZOOM_BASE
    current_pan = INITIAL_PAN_OFFSET.copy()
    target_zoom = ZOOM_BASE
    target_pan = INITIAL_PAN_OFFSET.copy()
    simulation_time = 0.0
    next_body_mass = DEFAULT_NEXT_BODY_MASS
    next_body_radius_pixels = DEFAULT_NEXT_BODY_RADIUS_PIXELS
    frame_times = deque(maxlen=60)
    color_options = [EARTH_COLOR, MARS_COLOR, VENUS_COLOR, MERCURY_COLOR, GAS_COLOR, ICE_COLOR, STAR_COLORS[1], STAR_COLORS[2]]
    color_index = 0
    adding_body_state = 0
    add_body_start_screen = np.zeros(2)
    gravity_multiplier = 1.0


    # --- 辅助函数 ---
    def load_preset(preset_name):
        nonlocal bodies, simulation_time, time_step, current_preset_name, selected_body
        nonlocal target_zoom, target_pan, current_zoom, current_pan, gravity_multiplier
        nonlocal softening_length_override

        if preset_name not in PRESETS:
            print(f"错误: 未找到预设 '{preset_name}'。")
            return

        preset_data = PRESETS[preset_name]
        preset_soft = PRESET_SOFTENING_LENGTHS.get(preset_name, C.SOFTENING_LENGTH)
        new_soft = softening_length_override if softening_length_override is not None else preset_soft
        C.SOFTENING_LENGTH = float(new_soft)
        C.SOFTENING_FACTOR_SQ = C.SOFTENING_LENGTH ** 2
        
        bodies.clear()
        Body.ID_counter = 0
        for body_config in preset_data:
            # 使用 Body.from_meters 方法创建天体，确保单位正确
            # 预设中的 'x' 和 'y' 已经是米，所以我们直接使用
            pos_m = np.array([
                body_config.get("x", 0.0),
                body_config.get("y", 0.0),
                0.0
            ])
            vel_m_s = np.array([
                body_config.get("vx", 0.0),
                body_config.get("vy", 0.0),
                0.0
            ])
            
            # 注意：from_meters 会将以米为单位的位置转换为内部模拟单位
            bodies.append(Body.from_meters(
                mass=body_config.get("mass", EARTH_MASS),
                pos_m=pos_m,
                vel_m_s=vel_m_s,
                color=body_config.get("color", EARTH_COLOR),
                radius=body_config.get("radius", DEFAULT_NEXT_BODY_RADIUS_PIXELS),
                fixed=body_config.get("fixed", False),
                name=body_config.get("name", f"Body_{Body.ID_counter}"),
                show_trail=SHOW_TRAILS
            ))

        simulation_time = 0.0
        time_step = TIME_STEP_BASE
        current_preset_name = preset_name
        selected_body = None
        # ... (重置UI元素) ...
        
        # 重置相机视角
        target_zoom = ZOOM_BASE
        com_pos, _ = calculate_center_of_mass(bodies)
        if com_pos is not None:
             screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
             # com_pos 是模拟单位，直接使用
             target_pan = screen_center - com_pos[:2] * target_zoom
        else:
             target_pan = INITIAL_PAN_OFFSET.copy()
        current_zoom = target_zoom
        current_pan = target_pan.copy()

    def process_input_events():
        """处理所有用户输入（UI、鼠标、键盘）。"""
        nonlocal running, paused, use_boundaries, SHOW_TRAILS, SHOW_GRAV_FIELD, ADAPTIVE_STEPPING
        nonlocal selected_body, dragging_body, dragging_camera, camera_drag_start_screen, camera_drag_start_pan
        nonlocal target_zoom, target_pan, adding_body_state, add_body_start_screen, color_index
        nonlocal gravity_multiplier, next_body_mass, next_body_radius_pixels

        mouse_screen_pos = np.array(pygame.mouse.get_pos())
        mouse_over_ui = control_panel_rect.collidepoint(mouse_screen_pos) or status_bar_rect.collidepoint(mouse_screen_pos)
        mouse_world_pos_m = (mouse_screen_pos - current_pan) / (current_zoom + 1e-18) * C.SPACE_SCALE if not mouse_over_ui else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return

            ui_manager.process_events(event)

            if event.type == pygame.USEREVENT:
                # 处理所有UI事件... (此部分逻辑与原文件相同, 此处省略)
                pass

            # 处理鼠标和键盘事件
            if not ui_manager.get_focus_set():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not mouse_over_ui:
                        if event.button == 3: # 右键开始添加天体
                            add_body_start_screen = mouse_screen_pos.copy()
                            adding_body_state = 1
                            status_text_label.set_text("拖动以设置速度...")
                        # ... 其他鼠标按钮事件 ...
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3: # 右键释放，完成添加
                        if adding_body_state == 1 and not mouse_over_ui:
                            # 开始位置（米）
                            start_pos_m = (add_body_start_screen - current_pan) / (current_zoom + 1e-18) * C.SPACE_SCALE
                            # 结束位置（米）
                            end_pos_m = (mouse_screen_pos - current_pan) / (current_zoom + 1e-18) * C.SPACE_SCALE
                            # 计算速度
                            vel_m_s = (end_pos_m - start_pos_m) / C.SPACE_SCALE * VELOCITY_DRAG_SCALE
                            
                            new_body = Body.from_meters(
                                mass=next_body_mass,
                                pos_m=np.array([start_pos_m[0], start_pos_m[1], 0.0]),
                                vel_m_s=np.array([vel_m_s[0], vel_m_s[1], 0.0]),
                                color=color_options[color_index],
                                radius=next_body_radius_pixels,
                                name=f"Body_{Body.ID_counter}",
                                show_trail=SHOW_TRAILS
                            )
                            bodies.append(new_body)
                            status_text_label.set_text(f"已添加 {new_body.name}")
                            color_index = (color_index + 1) % len(color_options)
                        adding_body_state = 0
                    # ... 其他鼠标按钮事件 ...
                # ... 其他事件 ...

    def update_physics():
        """更新物理状态，包括积分和碰撞。"""
        nonlocal simulation_time, time_step
        if not paused and bodies:
            current_g = INITIAL_G * gravity_multiplier
            sim_dt_propose = time_step * SPEED_FACTOR
            world_bounds_sim = get_world_bounds_sim(current_zoom, current_pan) if use_boundaries else None
            
            # ... (物理更新与原文件相同) ...
            if ADAPTIVE_STEPPING:
                time_advanced, next_dt_suggestion = adaptive_rk4_step(bodies, sim_dt_propose, current_g, ERROR_TOLERANCE, use_boundaries, world_bounds_sim)
                time_step = next_dt_suggestion
                simulation_time += time_advanced
            else:
                new_positions, new_velocities = perform_rk4_step(bodies, sim_dt_propose, current_g)
                for i, body in enumerate(bodies):
                    if not body.fixed:
                        body.update_physics_state(new_positions[i], new_velocities[i])
                        if use_boundaries and world_bounds_sim is not None:
                            body.handle_boundary_collision(world_bounds_sim)
                simulation_time += sim_dt_propose
            
            # 处理碰撞
            indices_to_remove = detect_and_handle_collisions(bodies, merge_on_collision=C.MERGE_ON_COLLISION)
            if indices_to_remove:
                for index in indices_to_remove:
                    if 0 <= index < len(bodies):
                        del bodies[index]

    def draw_scene():
        """绘制所有屏幕元素。"""
        screen.fill(BLACK)
        if SHOW_GRAV_FIELD:
            render_gravitational_field(screen, bodies, INITIAL_G * gravity_multiplier, current_zoom, current_pan)
        
        # 绘制天体和轨迹
        for i, body in enumerate(bodies):
            if i >= MAX_DISPLAY_BODIES: break
            show_labels = current_zoom > (ZOOM_BASE * 0.1)
            body.draw(screen, current_zoom, current_pan, draw_labels=show_labels)
        
        # ... (绘制速度预览线、质心标记等) ...

        ui_manager.draw_ui(screen)
        # ... (绘制FPS) ...
        pygame.display.flip()

    # --- 初始化 ---
    load_preset(current_preset_name)

    # --- 主循环 ---
    while running:
        real_dt = clock.tick(60) / 1000.0
        
        process_input_events()
        
        # 平滑相机移动
        current_zoom += (target_zoom - current_zoom) * CAMERA_SMOOTHING
        current_pan += (target_pan - current_pan) * CAMERA_SMOOTHING

        update_physics()
        
        # 更新轨迹
        for body in bodies:
            body.update_trail(current_zoom, current_pan)

        draw_scene()

    pygame.quit()


# --- 程序入口 ---
if __name__ == "__main__":
    # ... (命令行参数解析与原文件相同) ...
    try:
        main()
    except Exception as e:
        print(f"\n--- 模拟运行时错误 ---")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        input("按回车键退出...")
