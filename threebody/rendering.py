"""渲染助手和天体类。

这里定义的 :class:`Body` 类用于交互式 pygame 模拟。它扩展了
简单的物理天体，增加了颜色、半径和轨迹管理等额外的视觉属性。
"""
from collections import deque
import pygame
import pygame.gfxdraw
import numpy as np
from . import constants as C
from .jit import apply_boundary_conditions_jit


class Body:
    """代表一个具有物理和视觉属性的天体。"""
    ID_counter = 0

    def __init__(self, mass, pos, vel, color, radius,
                 max_trail_length=C.DEFAULT_TRAIL_LENGTH,
                 fixed=False, name=None, show_trail=True):
        """
        创建一个天体，将其位置/速度存储为三维向量。
        位置 pos 的单位是内部模拟单位，而不是米。
        """
        self.mass = float(mass)
        # 确保位置和速度是3D向量
        p = np.asarray(pos, dtype=float).reshape(-1)
        if p.size < 3:
            p = np.pad(p, (0, 3 - p.size))
        self.pos = p[:3] # 内部存储的是模拟单位

        v = np.asarray(vel, dtype=float).reshape(-1)
        if v.size < 3:
            v = np.pad(v, (0, 3 - v.size))
        self.vel = v[:3] # 速度单位是 m/s

        self.acc = np.zeros(3, dtype=np.float64)
        self.fixed = fixed
        self.color = color
        self.radius_pixels = max(1, int(radius))
        self.show_trail = show_trail
        self.max_trail_length = int(max_trail_length)
        self.trail = deque(maxlen=self.max_trail_length)
        self.visible = True
        self.id = Body.ID_counter
        Body.ID_counter += 1
        self.name = name if name else f"Body {self.id}"
        self.last_screen_pos = np.zeros(2)

    @staticmethod
    def from_meters(mass, pos_m, vel_m_s, color, radius,
                    max_trail_length=C.DEFAULT_TRAIL_LENGTH, fixed=False,
                    name=None, show_trail=True):
        """
        使用米为单位的坐标创建一个天体。
        这是一个非常重要的辅助函数，它能确保单位的正确转换。
        如果你有以米为单位的真实世界坐标，请使用此函数创建天体。
        
        参数:
            pos_m (array-like): 以米为单位的位置坐标。
            vel_m_s (array-like): 以米/秒为单位的速度。
        """
        # 将以米为单位的位置，通过除以 SPACE_SCALE 转换为内部模拟单位
        pos_sim = np.asarray(pos_m, dtype=float) / C.SPACE_SCALE
        return Body(
            mass,
            pos_sim, # 传入转换后的模拟单位位置
            vel_m_s,
            color,
            radius,
            max_trail_length=max_trail_length,
            fixed=fixed,
            name=name,
            show_trail=show_trail,
        )

    def update_physics_state(self, new_pos_sim, new_vel_m_s):
        """更新天体的物理状态。位置是模拟单位，速度是m/s。"""
        if not self.fixed:
            p = np.asarray(new_pos_sim, dtype=float).reshape(-1)
            if p.size < 3:
                p = np.pad(p, (0, 3 - p.size))
            self.pos = p[:3]
            v = np.asarray(new_vel_m_s, dtype=float).reshape(-1)
            if v.size < 3:
                v = np.pad(v, (0, 3 - v.size))
            self.vel = v[:3]

    def update_trail(self, zoom, pan_offset):
        """更新轨迹点。"""
        if not self.show_trail or not self.visible:
            if len(self.trail) > 0:
                self.trail.clear()
            return
        # self.pos 是模拟单位，乘以 zoom 得到屏幕像素单位
        screen_pos = self.pos[:2] * zoom + pan_offset
        self.last_screen_pos = screen_pos
        self.trail.append(screen_pos.copy())
        
    # ... 其他方法与原文件相同 ...
    def clear_trail(self):
        self.trail.clear()

    def set_trail_length(self, length):
        clamped = max(C.MIN_TRAIL_LENGTH, min(int(length), C.MAX_TRAIL_LENGTH))
        self.max_trail_length = clamped
        self.trail = deque(self.trail, maxlen=self.max_trail_length)

    def draw(self, screen, zoom, pan_offset, draw_labels):
        if not self.visible: return
        screen_pos = self.last_screen_pos
        # ... 绘制逻辑 ...
        pass
    
    def get_screen_pos(self, zoom, pan_offset):
        screen_pos = self.pos[:2] * zoom + pan_offset
        return (int(screen_pos[0]), int(screen_pos[1]))

    def handle_boundary_collision(self, bounds_sim, elasticity=0.8):
        if self.fixed: return
        new_pos, new_vel = apply_boundary_conditions_jit(self.pos[:2], self.vel[:2], bounds_sim, elasticity)
        if not np.array_equal(new_pos, self.pos[:2]) or not np.array_equal(new_vel, self.vel[:2]):
            self.pos[:2] = new_pos
            self.vel[:2] = new_vel

def render_gravitational_field(screen, bodies, g_constant, zoom, pan_offset):
    # ... 函数实现与原文件相同 ...
    pass
