from __future__ import annotations

import numpy as np
from manimlib.renderer.shader_source import read_shader_file
from manimlib.renderer.uniform_block import uniform_block_dtype, COMMON_UNIFORMS, Uniforms
from custom.constants import CYAN, FW, FH
from manimlib.constants import UL, DL, UR, DR, FRAME_HEIGHT, RED, UP
from manimlib.mobject.types.surface import Surface
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.geometry import Polygon, RegularPolygon
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.shape_matchers import Underline

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import Self
    from manimlib.camera.camera import Camera


class Star(Polygon):
    def __init__(self, n: int = 6, inner_radius: float = 1.0, outer_radius: float = 2.0, **kwargs):
        inner_polygon = RegularPolygon(n=n)
        outer_polygon = inner_polygon.copy()

        inner_polygon.scale(inner_radius)
        outer_polygon.scale(outer_radius)

        inner_dots = inner_polygon.get_points()[1::2]
        outer_dots = outer_polygon.get_points()[1::2]

        points = []

        for i in range(n):
            points.append(inner_dots[i])
            points.append(outer_dots[i])
        points.append(inner_dots[0])

        super().__init__(*points, **kwargs)


class ShaderMobject(Mobject):
    def __init__(
        self,
        shader_file: str,
        data_dtype: np.dtype = [("point", np.float32, (3,))],
        height: float = FRAME_HEIGHT,
        aspect_ratio: float = 16 / 9,
        verts_per_record: int = 6,
        **kwargs,
    ):
        self.aspect_ratio = aspect_ratio
        self.shader_file = shader_file
        self.data_dtype = data_dtype
        self.verts_per_record = verts_per_record
        self.uniform_types = {*COMMON_UNIFORMS, ("iTime", 1), ("iMouse", 2), ("iResolution", 2)}
        self.uniform_dtype = uniform_block_dtype(*self.uniform_types)
        super().__init__(**kwargs)

        self.set_height(height, stretch=True)
        self.set_width(height * aspect_ratio, stretch=True)

    def init_data(self, length: int = 4) -> None:
        super().init_data(length=length)
        self.data["point"] = [UL, DL, UR, DR]

    def set_color(self, *args, **kwargs):
        return self

    def refresh(self, camera: Camera) -> None:
        read_shader_file.cache_clear()
        self.shader_code_replacements = dict(self.shader_code_replacements)
        camera.renderer.materials.clear()

    def add_uniforms(self, *uniforms: tuple[str, int]) -> None:
        self.uniform_types = self.uniform_types.union(set(uniforms))
        self.uniform_dtype = uniform_block_dtype(*self.uniform_types)
        self.uniforms = Uniforms(self.uniform_dtype)

    def set_uniform(self, uniform: dict = {}, **kwargs) -> Self:
        uniform.update(kwargs)
        if unif := [(k, np.array(v).size) for k, v in uniform.items() if k not in self.uniforms]:
            self.add_uniforms(*unif)
        return super().set_uniform(**uniform)


class TitleText(Text):
    def __init__(
        self,
        text: str,
        gr: list[str] = [CYAN, RED],
        font: str = "Lobster Two",
        underline: bool = True,
        **kwargs,
    ):
        self.gr = gr
        super().__init__(text, font=font, **kwargs)
        self.to_edge(UP, buff=0.5)
        self.set_color_by_gradient(*gr)

        if underline:
            self.add_underline()

    def add_underline(self):
        underline = self.underline = Underline(self, stroke_color=CYAN, stretch_factor=1)
        underline.set_stroke(width=4, opacity=1)
        underline.set_color_by_gradient(*self.gr)
        self.add(underline)


class Rectangle3D(Surface):
    def __init__(
        self,
        width: float = FW,
        height: float = FH,
        resolution=(101, 101),
        color=CYAN,
        **kwargs,
    ):
        super().__init__(
            color=color,
            u_range=(-width / 2, width / 2),
            v_range=(-height / 2, height / 2),
            resolution=resolution,
            **kwargs,
        )

    def uv_func(self, u: float, v: float) -> np.ndarray:
        return np.array([u, v, 0])
