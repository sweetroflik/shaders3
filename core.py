from typing import Type, Iterable

import numpy as np
import taichi as ti
import taichi.math as tm

LARGE_DIST = 1000.0


@ti.func
def smoothmin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * (1.0 / 4.0)


@ti.func
def smoothmax(a, b, k):
    return smoothmin(a, b, -k)


@ti.func
def smoothmin3(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * h * k * (1.0 / 6.0)


@ti.func
def skewsin(x, t):
    return ti.atan2(t * ti.sin(x), (1.0 - t * ti.cos(x))) / t


@ti.func
def hash1(n):
    return tm.fract(ti.sin(n * 43758.5453))


@ti.func
def hash21(p):
    q = tm.fract(p * tm.vec2(123.34, 345.56))
    q += q @ (q + 34.23)
    return tm.fract(q.x * q.y)


@ti.func
def hash22(p):
    x = hash21(p)
    y = hash21(p + x)
    return tm.vec2(x, y)


@ti.func
def rot(a):
    c = ti.cos(a)
    s = ti.sin(a)
    return tm.mat2([c, -s], [s, c])


@ti.func
def fract_floor(x):
    fl = ti.floor(x)
    return tm.vec2(x - fl, fl)


@ti.func
def length(x):
    return ti.sqrt(x @ x)


def to_field(
    data: np.ndarray | Iterable, nptype: Type = np.float32, titype: Type = ti.f32
) -> ti.ScalarField:
    arr = np.array(data, dtype=nptype)
    field = ti.field(titype, shape=arr.size)
    field.from_numpy(arr)
    return field
