import os
import re
import numpy as np
from custom.constants import CYAN, FW, FH
from manimlib.utils.shaders import get_shader_code_from_file
from manimlib.constants import UL, DL, UR, DR, FRAME_HEIGHT, RED, UP
from manimlib.mobject.types.surface import Surface
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.geometry import Polygon, RegularPolygon
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.shape_matchers import Underline


class Star(Polygon):
    def __init__(
        self, n: int = 6, inner_radius: float = 1.0, outer_radius: float = 2.0, **kwargs
    ):
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
        shader_folder: str,
        data_dtype: np.dtype = [("point", np.float32, (3,))],
        height: float = FRAME_HEIGHT,
        aspect_ratio: float = 16 / 9,
        **kwargs,
    ):
        self.aspect_ratio = aspect_ratio
        self.shader_folder = shader_folder
        self.data_dtype = data_dtype

        super().__init__(**kwargs)
        self.set_height(height, stretch=True)
        self.set_width(height * aspect_ratio, stretch=True)

    def init_data(self, length: int = 4) -> None:
        super().init_data(length=length)
        self.data["point"][:] = [UL, DL, UR, DR]

    def set_color(self, *args, **kwargs):
        return self

    @Mobject.affects_data
    def refresh(self) -> None:
        """
        This is used to reload the shaders files
        (frag.glsl, vert.glsl, geom.glsl) in the embed mode.
        """

        for shader_type in ["fragment", "vertex", "geometry"]:
            file_name = f"{shader_type[:4]}.glsl"
            filepath = os.path.join(self.shader_folder, file_name)

            if not os.path.exists(filepath):
                if shader_type == "geometry":
                    # most of the time, geom.glsl is not required
                    continue
                else:
                    raise FileNotFoundError(
                        f"{file_name} isn't found at the specified location."
                    )

            with open(filepath, "r") as f:
                refreshed_code = f.read()

            # taken directly from 3b1b/manim
            insertions = re.findall(
                r"^#INSERT .*\.glsl$", refreshed_code, flags=re.MULTILINE
            )

            for line in insertions:
                inserted_code = get_shader_code_from_file(
                    os.path.join("inserts", line.replace("#INSERT ", ""))
                )
                refreshed_code = refreshed_code.replace(line, inserted_code)

            self.shader_wrapper.program_code[f"{shader_type}_shader"] = refreshed_code
            self.shader_wrapper.init_vertex_objects()
            self.shader_wrapper.init_program()
            self.shader_wrapper.refresh_id()


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
        underline = self.underline = Underline(
            self, stroke_color=CYAN, stretch_factor=1
        )
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
