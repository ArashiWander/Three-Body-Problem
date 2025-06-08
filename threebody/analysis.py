import numpy as np
import pygame
from collections import deque
from . import constants as C
from .physics_utils import calculate_system_energies
import csv
import os

def calculate_orbital_elements(body, central_body):
    """计算并返回一个天体相对于中心天体的轨道根数。"""
    if body is None or central_body is None or body.mass <= 0:
        return {
            'semi_major_axis': 0, 'eccentricity': 0, 'period': 0,
            'periapsis': 0, 'apoapsis': 0, 'speed': 0
        }

    r_vec = body.pos - central_body.pos
    v_vec = body.vel - central_body.vel

    r = np.linalg.norm(r_vec) * C.SPACE_SCALE
    v = np.linalg.norm(v_vec)
    
    mu = C.G_REAL * (central_body.mass + body.mass)

    specific_orbital_energy = v**2 / 2 - mu / r
    
    h_vec = np.cross(r_vec * C.SPACE_SCALE, v_vec)
    h = np.linalg.norm(h_vec)

    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec * C.SPACE_SCALE / r)
    eccentricity = np.linalg.norm(e_vec)

    if abs(specific_orbital_energy) < 1e-9: # 抛物线轨道
        semi_major_axis = float('inf')
        period = float('inf')
    else:
        semi_major_axis = -mu / (2 * specific_orbital_energy)
        if semi_major_axis > 0: # 椭圆轨道
             period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        else: # 双曲线轨道
            period = float('inf')

    periapsis = semi_major_axis * (1 - eccentricity) if semi_major_axis > 0 else 0
    apoapsis = semi_major_axis * (1 + eccentricity) if semi_major_axis > 0 else 0
    
    return {
        'semi_major_axis': semi_major_axis,
        'eccentricity': eccentricity,
        'period': period,
        'periapsis': periapsis,
        'apoapsis': apoapsis,
        'speed': v
    }

class EnergyMonitor:
    """监控并绘制系统总能量的变化。"""
    def __init__(self, max_points=500):
        self.history = deque(maxlen=max_points)
        self.initial_energy = None

    def set_initial_energy(self, bodies, g_constant):
        _, _, self.initial_energy = calculate_system_energies(bodies, g_constant)
        self.history.clear()

    def update(self, bodies, g_constant):
        if self.initial_energy is None or abs(self.initial_energy) < 1e-12:
            return
        _, _, current_energy = calculate_system_energies(bodies, g_constant)
        drift = ((current_energy - self.initial_energy) / self.initial_energy) * 100
        self.history.append(drift)

    def draw(self, surface):
        if len(self.history) < 2:
            return
        
        width, height = surface.get_size()
        points = []
        max_drift = max(abs(p) for p in self.history) if self.history else 1e-9
        max_drift = max(max_drift, 1e-9) # 避免除以零

        for i, drift in enumerate(self.history):
            x = (i / (self.history.maxlen - 1)) * width
            y = height / 2 - (drift / max_drift) * (height / 2 - 5)
            points.append((x, y))
        
        pygame.draw.lines(surface, (255, 100, 100), False, points, 2)
        
        # 绘制0%基准线
        pygame.draw.line(surface, (100, 100, 100), (0, height / 2), (width, height / 2), 1)

        # 显示最大漂移值
        font = pygame.font.Font(None, 18)
        text = font.render(f"能量漂移: {self.history[-1]:.3e} %", True, (200, 200, 200))
        surface.blit(text, (10, 5))

    def export_csv(self, file, delimiter=","):
        """Export the recorded energy drift history to a CSV file.

        Parameters
        ----------
        file : str or file-like
            Destination filename or open file object.
        delimiter : str, optional
            Delimiter used between columns (default is ',').
        """
        close = False
        if isinstance(file, (str, bytes, os.PathLike)):
            f = open(file, "w", newline="")
            close = True
        else:
            f = file
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["step", "energy_drift_percent"])
        for i, drift in enumerate(self.history):
            writer.writerow([i, drift])
        if close:
            f.close()
