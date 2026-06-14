import taichi as ti
import taichi.math as tm

SQRT3 = tm.sqrt(3)


@ti.func
def sd_circle(p, r):
    return p.norm() - r


@ti.func
def sd_segment(p, a, b):
    pa = p - a
    ba = b - a
    h = tm.clamp((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return (pa - ba * h).norm()


@ti.func
def sd_box(p, b):
    d = abs(p) - b
    return ti.max(d, 0.0).norm() + ti.min(ti.max(d.x, d.y), 0.0)


@ti.func
def sd_roundbox(p, b, r):
    rr = r.zw
    if p.x > 0.0:
        rr = r.xy
    if p.y < 0.0:
        rr.x = rr.y
    q = ti.abs(p) - b + rr[0]
    return ti.min(ti.max(q[0], q[1]), 0.0) + ti.max(q, 0.0).norm() - rr[0]


@ti.func
def sd_trapezoid(p, r1, r2, he):
    k1 = tm.vec2(r2, he)
    k2 = tm.vec2(r2 - r1, 2.0 * he)
    pp = tm.vec2(abs(p[0]), p[1])
    ca = tm.vec2(pp[0] - ti.min(pp[0], r1 if pp[1] < 0.0 else r2), ti.abs(pp[1]) - he)
    cb = pp - k1 + k2 * tm.clamp(((k1 - pp) @ k2) / (k2 @ k2), 0.0, 1.0)
    s = -1.0 if cb[0] < 0.0 and ca[1] < 0.0 else 1.0
    return s * ti.sqrt(ti.min(ca @ ca, cb @ cb))


@ti.func
def sd_arc(p, sc, ra, rb):
    """
    in vec2 p, in vec2 sc, in float ra, float rb
    """
    p.x = abs(p.x)

    return (
        tm.length(p - sc * ra) if sc.y * p.x > sc.x * p.y else abs(tm.length(p) - ra)
    ) - rb


@ti.func
def sd_ellipse(p, ab):
    p_ = abs(p)
    ab_ = ab
    if p_.x > p_.y:
        p_ = p_.yx
        ab_ = ab_.yx
    l = ab_.y * ab_.y - ab_.x * ab_.x
    m = ab_.x * p_.x / l
    m2 = m * m
    n = ab_.y * p_.y / l
    n2 = n * n
    c = (m2 + n2 - 1.0) / 3.0
    c3 = c * c * c
    q = c3 + m2 * n2 * 2.0
    d = c3 + m2 * n2
    g = m + m * n2
    co = 0.0
    if d < 0.0:
        h = tm.acos(q / c3) / 3.0
        s = tm.cos(h)
        t = tm.sin(h) * SQRT3
        rx = tm.sqrt(-c * (s + t + 2.0) + m2)
        ry = tm.sqrt(-c * (s - t + 2.0) + m2)
        co = (ry + tm.sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0
    else:
        h = 2.0 * m * n * tm.sqrt(d)
        s = tm.sign(q + h) * tm.pow(abs(q + h), 1.0 / 3.0)
        u = tm.sign(q - h) * tm.pow(abs(q - h), 1.0 / 3.0)
        rx = -s - u - c * 4.0 + 2.0 * m2
        ry = (s - u) * SQRT3
        rm = tm.sqrt(rx * rx + ry * ry)
        co = (ry / tm.sqrt(rm - rx) + 2.0 * g / rm - m) / 2.0

    r = ab_ * tm.vec2(co, tm.sqrt(1.0 - co * co))
    return tm.length(r - p_) * tm.sign(p_.y - r.y)


@ti.func
def sd_cut_disk(p, r, h):  # p: tm.vec2, r: ti.f32, h: ti.f32
    p_ = p
    w = tm.sqrt(r * r - h * h)
    p_.x = abs(p_.x)
    s = max((h - r) * p_.x * p_.x + w * w * (h + r - 2.0 * p_.y), h * p_.x - w * p_.y)
    d = 0.0
    if s < 0.0:
        d = tm.length(p_) - r
    elif p_.x < w:
        d = h - p_.y
    else:
        d = tm.length(p_ - tm.vec2(w, h))
    return d


@ti.func
def sd_cut_disk_aligned(p, r, h):  # p: tm.vec2, r: ti.f32, h: ti.f32
    return sd_cut_disk(tm.vec2(p.x, p.y + h), r, h)


@ti.func
def sd_moon(p, d, ra, rb):  # p: tm.vec2, d: float, ra: float, rb: float
    p_ = p
    p_.y = abs(p_.y)
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = tm.sqrt(max(ra * ra - a * a, 0.0))
    dst = 0.0
    if d * (p_.x * b - p_.y * a) > d * d * max(b - p_.y, 0.0):
        dst = tm.length(p_ - tm.vec2(a, b))
    else:
        dst = max((tm.length(p_) - ra), -(tm.length(p_ - tm.vec2(d, 0)) - rb))
    return dst
