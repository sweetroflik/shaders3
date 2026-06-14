import taichi as ti
import taichi.math as tm

red = tm.vec3(1.0, 0.0, 0.0)
green = tm.vec3(0.0, 1.0, 0.0)
blue = tm.vec3(0.0, 0.0, 1.0)
black = tm.vec3(0.0)
white = tm.vec3(1.0)

# 1D color gradients from: https://www.shadertoy.com/view/4dsSzr


@ti.func
def hue_gradient(t: ti.f32):
    p = ti.abs(tm.fract(t + tm.vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0)
    return tm.clamp(p - 1.0, 0.0, 1.0)


@ti.func
def tech_gradient(t: ti.f32):
    return ti.pow(tm.vec3(t + 0.01), tm.vec3(120.0, 10.0, 180.0))


@ti.func
def fire_gradient(t: ti.f32):
    return ti.max(
        ti.pow(tm.vec3(ti.min(t * 1.02, 1.0)), tm.vec3(1.7, 25.0, 100.0)),
        tm.vec3(0.06 * pow(max(1.0 - abs(t - 0.35), 0.0), 5.0)),
    )


@ti.func
def desert_gradient(t: ti.f32):
    s = ti.sqrt(tm.clamp(1.0 - (t - 0.4) / 0.6, 0.0, 1.0))
    sky = ti.sqrt(
        tm.mix(tm.vec3(1.0), tm.vec3(0.0, 0.8, 1.0), tm.smoothstep(t, 0.4, 0.9))
        * tm.vec3(s, s, 1.0)
    )
    land = tm.mix(
        tm.vec3(0.7, 0.3, 0.0),
        tm.vec3(0.85, 0.75 + ti.max(0.8 - t * 20.0, 0.0), 0.5),
        (t / 0.4) ** 2,
    )
    return tm.clamp(sky if t > 0.4 else land, 0.0, 1.0) * tm.clamp(
        1.5 * (1.0 - ti.abs(t - 0.4)), 0.0, 1.0
    )


@ti.func
def electric_gradient(t: ti.f32):
    return tm.clamp(
        tm.vec3(t * 8.0 - 6.3, tm.smoothstep(t, 0.6, 0.9) ** 2, ti.pow(t, 3.0) * 1.7),
        0.0,
        1.0,
    )


@ti.func
def neon_gradient(t: ti.f32):
    return tm.clamp(
        tm.vec3(t * 1.3 + 0.1, (abs(0.43 - t) * 1.7) ** 2, (1.0 - t) * 1.7), 0.0, 1.0
    )


@ti.func
def heatmap_gradient(t: ti.f32):
    return tm.clamp(
        (pow(t, 1.5) * 0.8 + 0.2)
        * tm.vec3(
            tm.smoothstep(t, 0.0, 0.35) + t * 0.5,
            tm.smoothstep(t, 0.5, 1.0),
            ti.max(1.0 - t * 1.7, t * 7.0 - 6.0),
        ),
        0.0,
        1.0,
    )


@ti.func
def rainbow_gradient(t: ti.f32):
    c = 1.0 - ti.pow(
        ti.abs(tm.vec3(t) - tm.vec3(0.65, 0.5, 0.2)) * tm.vec3(3.0, 3.0, 5.0),
        tm.vec3(1.5, 1.3, 1.7),
    )
    c.r = max((0.15 - (ti.abs(t - 0.04) * 5.0) ** 2), c.r)
    c.g = tm.smoothstep(t, 0.04, 0.45) if t < 0.5 else c.g
    return tm.clamp(c, 0.0, 1.0)


@ti.func
def brightness_gradient(t: ti.f32):
    return tm.vec3(t * t)


@ti.func
def grayscale_gradient(t: ti.f32):
    return tm.vec3(t)


@ti.func
def stripe_gradient(t: ti.f32):
    return tm.vec3(ti.floor(t * 32.0) % 2.0) * 0.2 + 0.8


@ti.func
def ansi_gradient(t: ti.f32):
    return ti.floor(t * tm.vec3(8.0, 4.0, 2.0)) % 2.0


if __name__ == "__main__":
    import time

    ti.init(arch=ti.gpu)
    # ti.init(arch=ti.cpu)

    # Настройка размеров окна
    asp = 16 / 9  # отношение сторон
    h = 600  # высота в пикселях
    w = int(asp * h)  # ширина в пикселях
    res = w, h
    resf = tm.vec2(float(w), float(h))

    # Векторное поле, расположенное в памяти видеокарты (в случае ti.gpu)
    # Функция kernel записывает в это поле цвет каждого пикселя (RGB)
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

    @ti.func
    def show_all_gradientm(fragCoord):
        num_palettes = 12.0

        x = fragCoord.x / resf.x
        # j = x + (tm.fract(ti.sin(fragCoord.y * 7.5e2 + fragCoord.x * 6.4) * 1e2) - 0.5) * 0.005
        i = num_palettes * fragCoord.y / resf.y

        col = tm.vec3(0.0)
        if fragCoord.y % (resf.y / num_palettes) < ti.max(resf.y / 100.0, 3.0):
            col = tm.vec3(0.0)
        elif i > 11.0:
            col = hue_gradient(x)
        elif i > 10.0:
            col = tech_gradient(x)
        elif i > 9.0:
            col = fire_gradient(x)
        elif i > 8.0:
            col = desert_gradient(x)
        elif i > 7.0:
            col = electric_gradient(x)
        elif i > 6.0:
            col = neon_gradient(x)
        elif i > 5.0:
            col = heatmap_gradient(x)
        elif i > 4.0:
            col = rainbow_gradient(x)
        elif i > 3.0:
            col = brightness_gradient(x)
        elif i > 2.0:
            col = grayscale_gradient(x)
        elif i > 1.0:
            col = stripe_gradient(x)
        else:
            col = ansi_gradient(x)

        return tm.clamp(col ** (1 / 2.2), 0.0, 1.0)

    @ti.kernel
    def render(t: ti.f32):
        """
        Основная функция, внешний цикл которой автоматически распараллеливается.
        Выполняется на видеокарте (в случае ti.gpu)

        :param t: время, прошедшее от первого кадра
        :return:
        """

        for fragCoord in ti.grouped(pixels):
            # uv = (fragCoord.xy + tm.vec2(0.0, ti.sin(t * 8.0 + fragCoord.x * 0.02) * 100.0 * float(ti.max(ti.sin(t), 0.0)))) % resf.xy
            col = show_all_gradientm(fragCoord)
            pixels[fragCoord] = col

    gui = ti.GUI("Color gradients", res=res, fast_gui=True)
    start = time.time()

    while gui.running:

        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        t = time.time() - start
        render(t)
        gui.set_image(pixels)
        gui.show()

    gui.close()
