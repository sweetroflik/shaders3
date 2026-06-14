from typing import Iterable

import numpy as np
import taichi as ti
from taichi import math as tm

from core import LARGE_DIST, to_field, rot
from sdf import sd_circle, sd_segment, sd_cut_disk, sd_cut_disk_aligned, sd_moon
from colors import black, white
from sdf_shader import SDBase, SDFShader

LENGTHS_1 = (0.1,) * 10


@ti.data_oriented
class Spine:
    def __init__(
        self,
        lengths: np.ndarray | Iterable[float] = LENGTHS_1,
        head: tm.vec2 = tm.vec2(0.0, 0.0),
        align: tm.vec2 = tm.vec2(1.0, 0.0),
    ):
        n = len(lengths) + 1
        assert n > 1, "Spine should contain at least 2 nodes"
        self.n = n
        self.lengths = to_field(lengths)
        self.nodes = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.links = ti.Vector.field(2, dtype=ti.f32, shape=n - 1)
        self.normals = ti.Vector.field(2, dtype=ti.f32, shape=n - 1)
        self.curvature = ti.field(dtype=ti.f32, shape=n - 2)
        self.head = head
        self.align = align

    @ti.func
    def place_head(self, head: tm.vec2):
        self.nodes[0] = head

    @ti.func
    def place_nodes(self):
        for i in ti.static(range(1, self.n)):
            self.nodes[i] = self.nodes[i - 1] + self.lengths[i - 1] * self.align

    @ti.func
    def update_nodes(self):
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
    def move_to(self, v: tm.vec2):
        self.place_head(v)
        self.update_nodes()


@ti.data_oriented
class SDSpine(Spine, SDBase):
    def __init__(
        self,
        lengths: np.ndarray | Iterable[float] = LENGTHS_1,
        head: tm.vec2 = tm.vec2(0.0, 0.0),
        align: tm.vec2 = tm.vec2(1.0, 0.0),
        r: ti.f32 = 0.1,  # node size
        w: ti.f32 = 0.01,  # link width
        g: ti.f32 = 0.05,  # normal length
    ):
        super().__init__(lengths=lengths, head=head, align=align)
        self.r = r
        self.w = w
        self.g = g

    @ti.func
    def calc_distance(self, uv: tm.vec2) -> ti.f32:
        d = LARGE_DIST
        for i in ti.static(range(self.n)):
            d = min(d, sd_circle(uv - self.nodes[i], self.r))

        for j in ti.static(range(self.n - 1)):
            d = min(d, sd_segment(uv, self.nodes[j], self.nodes[j + 1]) - self.w)
            center = (self.nodes[j] + self.nodes[j + 1]) * 0.5
            left = center + self.g * self.normals[j]
            right = center - self.g * self.normals[j]
            # d = min(d, sd_segment(uv, center, left) - self.w)
            # d = min(d, sd_segment(uv, center, right) - self.w)
            d = min(d, sd_segment(uv, left, right) - self.w)

        # fin test
        i = 2
        p = (self.nodes[i] + self.nodes[i + 1]) * 0.5
        c = self.curvature[i]
        a = tm.atan2(self.normals[i].y, self.normals[i].x)
        if c > 0:
            a += tm.pi
        m = rot(-a)

        ra = 0.5
        rb = 2.0
        di = tm.mix(rb - ra, rb * 1.05, abs(c))
        pp = m @ (uv - p)
        d = min(d, sd_moon(tm.vec2(pp.x + di - rb, pp.y), di, ra, rb))

        return d


class SpineShader(SDFShader):
    def __init__(
        self,
        spine: SDSpine,
        smooth: ti.f32 = 0.005,
        scale: ti.f32 = 1.0,
        color: tm.vec3 = white,
        bgcolor: tm.vec3 = black,
        title: str = "SDSpine simple shader",
        res: tuple[int, int] | None = None,
        gamma: float = 2.2,
    ):
        super().__init__(
            spine,
            smooth=smooth,
            scale=scale,
            color=color,
            bgcolor=bgcolor,
            title=title,
            res=res,
            gamma=gamma,
        )
        self.spine = spine

    @ti.kernel
    def init(self):
        self.spine.place_nodes()

    @ti.kernel
    def calculate(self, t: ti.f32, cursor: tm.vec2):
        w = tm.vec2(2.37, 4.68)
        v = ti.sin(0.1 * t * w)
        self.spine.move_to(v)


if __name__ == "__main__":
    ti.init(arch=ti.opengl)
    # ti.init(arch=ti.cpu)

    s = 0.05
    spine = SDSpine(lengths=np.array([1, 2, 5, 2, 1]*2) * s, r=0.01, w=0, g=0.05)
    shader = SpineShader(spine, scale=2)
    shader.main_loop()
