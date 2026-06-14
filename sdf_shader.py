import taichi as ti
import taichi.math as tm

from colors import white, black
from gui import BaseShader


@ti.data_oriented
class SDBase:

    @ti.func
    def calc_distance(self, uv: tm.vec2) -> ti.f32:
        raise NotImplementedError()


class SDFShader(BaseShader):
    def __init__(
        self,
        sdf: SDBase,
        smooth: ti.f32 = 0.005,
        scale: ti.f32 = 1.0,
        color: tm.vec3 = white,
        bgcolor: tm.vec3 = black,
        title: str = "SDF simple shader",
        res: tuple[int, int] | None = None,
        gamma: float = 2.2,
    ):
        super().__init__(title, res=res, gamma=gamma)
        self.sdf = sdf
        self.smooth = smooth
        self.scale = scale
        self.color = color
        self.bgcolor = bgcolor

    @ti.func
    def main_image(self, uv: tm.vec2, t: ti.f32, cursor: tm.vec2):
        uv *= self.scale
        d = self.sdf.calc_distance(uv)
        alpha = tm.smoothstep(self.smooth, 0.0, d)
        col = tm.mix(self.bgcolor, self.color, alpha)
        return col
