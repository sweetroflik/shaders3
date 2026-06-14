"""
Ящерица с двумя позвоночниками (тело и хвост) и модификациями.

Модификации:
1б - следование за мышью с запозданием (ускорение зависит от расстояния)
4б - упрощенное шагание конечностей (моментальное передвижение)
6 - второй позвоночник (хвост)

Класс SDLizard реализует:
- Основной позвоночник (тело) с 12 сегментами
- Второй позвоночник (хвост) с 8 сегментами
- Четыре пары конечностей (ноги)
- Глаза
- Цветной слой для визуализации
"""

from typing import Iterable

import numpy as np
import taichi as ti
from taichi import math as tm

from core import LARGE_DIST, smoothmin, rot, to_field, hash21
from sdf import sd_circle, sd_segment, sd_moon
from spine import SpineShader, SDSpine

# Параметры основного позвоночника (тело ящерицы)
BODY_RADII = np.array([3.5, 4.0, 4.2, 4.0, 3.8, 3.5, 3.2, 2.8, 2.2, 1.5, 1.0, 0.5]) * 0.01
BODY_LENGTHS = (0.06,) * (len(BODY_RADII) - 1)

# Параметры хвоста (второй позвоночник)
TAIL_RADII = np.array([1.2, 1.4, 1.3, 1.2, 1.0, 0.8, 0.5, 0.2]) * 0.01
TAIL_LENGTHS = (0.05,) * (len(TAIL_RADII) - 1)

# Индексы ног (на каких сегментах тела расположены)
LIMBS_IDX = (3, 5, 7, 9)  # 4 ноги

# Параметры ног
LIMBS_LEN = np.array([4.5, 4.5, 4.0, 4.0])  # длина каждой ноги
LIMBS_ANG = np.array([np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4])  # угол ног


@ti.data_oriented
class SDLizard(SDSpine):
    """
    Класс для расчета и визуализации ящерицы.
    
    Ящерица состоит из:
    - Основного позвоночника (тело)
    - Второго позвоночника (хвост)
    - Четырех ног
    - Двух глаз
    
    Поддерживает модификации:
    - Следование за мышью с запозданием
    - Шагание конечностей
    """
    
    def __init__(
        self,
        body_lengths: np.ndarray | Iterable[float] = BODY_LENGTHS,
        body_radii: np.ndarray | list[float] = BODY_RADII,
        tail_lengths: np.ndarray | Iterable[float] = TAIL_LENGTHS,
        tail_radii: np.ndarray | list[float] = TAIL_RADII,
        head: tm.vec2 = tm.vec2(0.0, 0.0),
        align: tm.vec2 = tm.vec2(1.0, 0.0),
        limbs_idx: np.ndarray | Iterable[int] = LIMBS_IDX,
        limbs_len: np.ndarray | Iterable[float] = LIMBS_LEN,
        limbs_ang: np.ndarray | Iterable[float] = LIMBS_ANG,
        body_smooth: ti.f32 = 0.1,
    ):
        """
        Инициализация ящерицы.
        
        :param body_lengths: длины сегментов тела
        :param body_radii: радиусы узлов тела
        :param tail_lengths: длины сегментов хвоста
        :param tail_radii: радиусы узлов хвоста
        :param head: начальная позиция головы
        :param align: направление начального движения
        :param limbs_idx: индексы сегментов, где расположены ноги
        :param limbs_len: длины ног
        :param limbs_ang: углы ног
        :param body_smooth: коэффициент сглаживания для тела
        """
        super().__init__(body_lengths, head=head, align=align)
        
        assert len(body_radii) == self.n, "Body radius should be defined for each node"
        self.body_radii = to_field(body_radii)
        self.body_smooth = body_smooth
        
        # Хвост (второй позвоночник)
        self.tail_n = len(tail_lengths) + 1
        self.tail_lengths = to_field(tail_lengths)
        self.tail_nodes = ti.Vector.field(2, dtype=ti.f32, shape=self.tail_n)
        self.tail_radii = to_field(tail_radii)
        
        # Ноги
        self.limbs_idx = to_field(limbs_idx, nptype=np.int32, titype=ti.i32)
        self.limbs_len = to_field(limbs_len)
        self.limbs_ang = to_field(limbs_ang)
        self.n_limbs = len(limbs_len)
        
        # Для модификации 4б - шагание ног
        # Каждая нога имеет текущую и целевую позицию
        self.limb_positions = ti.Vector.field(2, dtype=ti.f32, shape=self.n_limbs)
        self.limb_targets = ti.Vector.field(2, dtype=ti.f32, shape=self.n_limbs)
        self.limb_timers = ti.field(dtype=ti.f32, shape=self.n_limbs)
        
        # Для модификации 1б - следование за мышью с запозданием
        self.target_pos = tm.vec2(0.0, 0.0)
        self.current_pos = head
        
        # Глаза
        self.eye_offset = 0.04
        self.eye_size = 0.012

    @ti.func
    def update_body(self):
        """
        Обновляет позиции узлов тела на основе движения головы.
        """
        for i in ti.static(range(1, self.n)):
            delta = tm.normalize(self.nodes[i] - self.nodes[i - 1])
            self.nodes[i] = self.nodes[i - 1] + delta * self.lengths[i - 1]

        for i in ti.static(range(self.n - 1)):
            link = tm.normalize(self.nodes[i] - self.nodes[i + 1])
            self.links[i] = link
            self.normals[i] = tm.vec2(-link.y, link.x)

        for i in ti.static(range(self.n - 2)):
            s = tm.cross(self.links[i], self.links[i + 1])
            dot = self.links[i] @ self.links[i + 1]
            self.curvature[i] = s * dot

    @ti.func
    def update_tail(self):
        """
        Обновляет позиции узлов хвоста.
        Хвост начинается от конца тела.
        """
        # Хвост начинается от последнего у��ла тела
        self.tail_nodes[0] = self.nodes[self.n - 1]
        
        # Используем направление последнего сегмента тела для начального направления хвоста
        tail_align = self.links[self.n - 2]
        
        for i in ti.static(range(1, self.tail_n)):
            self.tail_nodes[i] = self.tail_nodes[i - 1] + tail_align * self.tail_lengths[i - 1]

    @ti.func
    def update_limbs(self, t: ti.f32, dt: ti.f32 = 0.016):
        """
        Обновляет позиции ног с эффектом шагания.
        Модификация 4б: упрощенное шагание (моментальное передвижение).
        
        :param t: текущее время
        :param dt: шаг времени
        """
        step_duration = 0.3  # длительность одного шага
        
        for j in ti.static(range(self.n_limbs)):
            # Определяем фазу шага для каждой ноги
            # Передние ноги движутся в противофазе с задними
            phase_offset = 0.0 if j < 2 else ti.pi
            phase = ti.fmod(t * 2.0 * ti.pi / step_duration + phase_offset, 2.0 * ti.pi)
            
            # Движение ноги: фаза 0 - шаг вперед, фаза pi - шаг назад
            is_stepping = phase > ti.pi
            
            # Вычисляем целевую позицию ноги
            limb_idx = self.limbs_idx[j]
            attach_point = self.nodes[limb_idx]
            side = 1.0 if j % 2 == 0 else -1.0
            
            # Целевая позиция ноги на земле
            forward_offset = tm.normalize(self.links[limb_idx]) * self.limbs_len[j] * 0.5
            side_offset = side * self.body_radii[limb_idx] * self.limbs_len[j] * 0.8
            
            if is_stepping:
                self.limb_targets[j] = attach_point + forward_offset + tm.vec2(side_offset * 0.5, -0.15)
            else:
                self.limb_targets[j] = attach_point + forward_offset + tm.vec2(side_offset, -0.15)
            
            # Текущая позиция ноги движется к целевой
            self.limb_positions[j] = self.limb_targets[j]

    @ti.func
    def calc_distance(self, uv: tm.vec2) -> ti.f32:
        """
        Вычисляет расстояние до поверхности ящерицы.
        Включает: тело, хвост, ноги, глаза.
        
        :param uv: координата в пространстве
        :return: расстояние до ящерицы
        """
        d = LARGE_DIST
        
        # Тело - сглаженное объединение кругов
        for i in ti.static(range(self.n)):
            d = smoothmin(d, sd_circle(uv - self.nodes[i], self.body_radii[i]), self.body_smooth)
        
        # Хвост - сглаженное объединение кругов
        tail_smooth = 0.08
        for i in ti.static(range(self.tail_n)):
            d = smoothmin(d, sd_circle(uv - self.tail_nodes[i], self.tail_radii[i]), tail_smooth)
        
        # Ноги
        for j in ti.static(range(self.n_limbs)):
            limb_idx = self.limbs_idx[j]
            attach_point = self.nodes[limb_idx]
            foot_pos = self.limb_positions[j]
            
            cur_d = sd_segment(uv, attach_point, foot_pos) - 0.01
            d = ti.min(d, cur_d)
        
        # Глаза
        eye_y_offset = self.body_radii[0] * 1.2
        left_eye = self.nodes[0] + tm.vec2(-self.eye_offset, eye_y_offset)
        right_eye = self.nodes[0] + tm.vec2(self.eye_offset, eye_y_offset)
        
        d = ti.min(d, sd_circle(uv - left_eye, self.eye_size))
        d = ti.min(d, sd_circle(uv - right_eye, self.eye_size))
        
        return d


class LizardShader(SpineShader):
    """
    Шейдер для отрисовки ящерицы с поддержкой модификаций.
    
    Модификации:
    1б - следование за мышью с запозданием
    4б - шагание ног
    6 - два позвоночника (тело и хвост)
    """
    
    def __init__(
        self,
        lizard: SDLizard,
        smooth: ti.f32 = 0.005,
        scale: ti.f32 = 1.0,
        color: tm.vec3 = None,
        bgcolor: tm.vec3 = None,
        title: str = "Lizard Shader",
        res: tuple[int, int] | None = None,
        gamma: float = 2.2,
    ):
        """
        Инициализация шейдера ящерицы.
        
        :param lizard: объект SDLizard для визуализации
        :param smooth: коэффициент сглаживания границ
        :param scale: масштаб изображения
        :param color: цвет ящерицы
        :param bgcolor: цвет фона
        :param title: название окна
        :param res: разрешение экрана
        :param gamma: гамма-коррекция
        """
        if color is None:
            from colors import heatmap_gradient
            color = heatmap_gradient(0.6)  # Зеленовато-коричневый цвет
        if bgcolor is None:
            from colors import black
            bgcolor = black
            
        super().__init__(
            lizard,
            smooth=smooth,
            scale=scale,
            color=color,
            bgcolor=bgcolor,
            title=title,
            res=res,
            gamma=gamma,
        )
        self.lizard = lizard
        
        # Для модификации 1б - параметры следования за мышью
        self.follow_speed = 0.05  # коэффициент ускорения
        self.last_cursor_time = 0.0

    @ti.kernel
    def init(self):
        """
        Инициализация шейдера.
        Размещает узлы позвоночников в начальных позициях.
        """
        self.lizard.place_nodes()
        self.lizard.update_tail()
        
        for i in ti.static(range(self.lizard.n_limbs)):
            self.lizard.limb_positions[i] = self.lizard.nodes[self.lizard.limbs_idx[i]]
            self.lizard.limb_targets[i] = self.lizard.nodes[self.lizard.limbs_idx[i]]

    @ti.kernel
    def calculate(self, t: ti.f32, cursor: tm.vec2):
        """
        Основная функция расчета.
        Выполняет:
        1. Обновление позиции головы на основе следования за мышью
        2. Обновление тела
        3. Обновление хвоста
        4. Обновление ног
        
        Модификация 1б: ускорение зависит от расстояния до мыши.
        
        :param t: текущее время
        :param cursor: позиция курсора мыши
        """
        # Модификация 1б - следование за мышью с запозданием
        # Ускорение зависит от расстояния до указателя мыши
        target_direction = tm.normalize(cursor - self.lizard.current_pos)
        distance_to_cursor = (cursor - self.lizard.current_pos).norm()
        
        # Скорость движения возрастает с расстоянием до мыши
        speed = self.follow_speed * tm.clamp(distance_to_cursor * 2.0, 0.1, 1.0)
        
        # Обновляем позицию ��оловы ящерицы
        self.lizard.nodes[0] += target_direction * speed
        self.lizard.current_pos = self.lizard.nodes[0]
        
        # Обновляем выравнивание (направление позвоночника)
        self.lizard.align = tm.normalize(target_direction)
        
        # Обновляем тело
        self.lizard.update_body()
        
        # Обновляем хвост
        self.lizard.update_tail()
        
        # Обновляем ноги (модификация 4б)
        self.lizard.update_limbs(t)


if __name__ == "__main__":
    ti.init(arch=ti.opengl)

    # Создаем ящерицу
    lizard = SDLizard(body_smooth=0.1)
    
    # Создаем шейдер
    shader = LizardShader(lizard, scale=2)
    
    # Запускаем главный цикл
    shader.main_loop()
