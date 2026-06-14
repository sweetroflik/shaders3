from typing import Iterable

import numpy as np
import taichi as ti
from taichi import math as tm

from core import LARGE_DIST, smoothmin, rot, to_field
from sdf import sd_circle, sd_segment, sd_moon
from spine import SpineShader, SDSpine

CHAIN_1 = np.array([1, 2, 3, 4, 3.5, 3, 2, 2, 1.5, 1, 1, 1, 0.3, 0.3, 0.3, 0.3]) * 0.02
LENGTHS_1 = (0.05,) * (len(CHAIN_1) - 1)
LIMBS_IDX_1 = (2, 10)
LIMBS_LEN_1 = np.array([4.0, 3.0])
LIMBS_ANG_1 = -np.array([np.pi / 6, np.pi/12])

FINS_IDX_1 = (5,)


@ti.data_oriented
class SDAnimal(SDSpine):
    def __init__(
        self,
        lengths: np.ndarray | Iterable[float] = LENGTHS_1,
        radii: np.ndarray | list[float] = CHAIN_1,
        head: tm.vec2 = tm.vec2(0.0, 0.0),
        align: tm.vec2 = tm.vec2(1.0, 0.0),
        limbs_idx: np.ndarray | Iterable[int] = LIMBS_IDX_1,
        limbs_len: np.ndarray | Iterable[float] = LIMBS_LEN_1,
        limbs_ang: np.ndarray | Iterable[float] = LIMBS_ANG_1,
        fins_idx: np.ndarray | Iterable[int] = FINS_IDX_1,
        body_smooth: ti.f32 = 0.1
    ):
        super().__init__(lengths, head=head, align=align)
        assert len(radii) == self.n, "Radius should be defined for each node"
        self.radii = to_field(radii)
        self.limbs_idx = to_field(limbs_idx, nptype=np.int32, titype=ti.i32)
        self.limbs_len = to_field(limbs_len)
        self.limbs_ang = to_field(limbs_ang)
        self.body_smooth = body_smooth
        self.fins_idx = to_field(fins_idx, nptype=np.int32, titype=ti.i32)

    @ti.func
    def calc_distance(self, uv: tm.vec2) -> ti.f32:
        d = LARGE_DIST
        for i in ti.static(range(self.n)):
            d = smoothmin(d, sd_circle(uv - self.nodes[i], self.radii[i]), self.body_smooth)

        for j in ti.static(range(self.limbs_idx.shape[0])):
            i = self.limbs_idx[j]
            limb_len = self.limbs_len[j]
            a = self.nodes[i]
            for s in ti.static((1, -1)):
                b = a + s * self.normals[i + 1] * self.radii[i] * limb_len
                v = rot(-s * self.limbs_ang[j]) @ (b - a)
                b = a + v
                cur_d = sd_segment(uv, a, b) - 0.02
                d = ti.min(d, cur_d)

        # d = SDSpine.calc_distance(self, uv)

        for j in ti.static(range(self.fins_idx.shape[0])):
            i = self.fins_idx[j]
            p = (self.nodes[i] + self.nodes[i + 1]) * 0.5
            c = self.curvature[i]
            a = tm.atan2(self.normals[i].y, self.normals[i].x)
            if c > 0:
                a += tm.pi
            m = rot(-a)
            ra = 0.15
            rb = 0.5
            di = tm.mix(rb - ra, rb, abs(c))
            pp = m @ (uv - p)
            d = max(d, -sd_moon(tm.vec2(pp.x + di - rb, pp.y), di, ra, rb) + 0.01)

        # for i in ti.static(range(self.n - 1)):
        #     node = self.nodes[i + 1]
        #     normal = self.normals[i]
        #     d = ti.min(d, sd_circle(uv - node + normal, self.node_size * 0.5))
        #     d = ti.min(d, sd_circle(uv - node - normal, self.node_size * 0.5))

        return d


if __name__ == "__main__":
    ti.init(arch=ti.opengl)

    animal = SDAnimal(body_smooth=0.1)
    shader = SpineShader(animal, scale=2)
    shader.main_loop()
