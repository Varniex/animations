from __future__ import annotations

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized
from scipy.ndimage import map_coordinates
from matplotlib.colors import LinearSegmentedColormap

from manimlib.constants import GREY_B, RIGHT, UP, DOWN
from custom.constants import FH, FW
from custom.objects import Rectangle3D
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.coordinate_systems import NumberPlane
from manimlib.mobject.vector_field import VectorField
from manimlib.mobject.types.dot_cloud import DotCloud, GlowDots
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VMobject, VGroup

from custom.utils import normalize_array

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
    from manimlib.constants import ManimColor


def normalize_array(array: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return (array - array.min()) / (array.max() - array.min() + epsilon)


# Thanks to https://github.com/ChrisWu1997/StableFluids-python for the fluid pipeline and examples.
def get_laplacian_matrix(nx: int, ny: int):
    main = -4 * np.ones(nx * ny)
    main[[0, ny - 1, -ny, -1]] = -2
    main[
        [
            *range(1, ny - 1),
            *range(-ny + 1, -1),
            *range(ny, (nx - 2) * ny + 1, ny),
            *range(2 * ny - 1, (nx - 1) * ny, ny),
        ]
    ] = -3
    side = np.ones(nx * ny - 1)
    side[[*range(ny - 1, nx * ny - 1, ny)]] = 0
    data = [np.ones(nx * ny - ny), side, main, side, np.ones(nx * ny - ny)]
    offset = [-ny, -1, 0, 1, ny]
    return sparse.diags(data, offset)


# Stable Fluids
class Fluid(VMobject):
    def __init__(
        self,
        size: float = 0.2,  # size of each cell
        diffusion: float = 0,  # diffusion
        viscosity: float = 0,  # viscosity,
        max_iter: int = 20,  # no of iteration per frame
        width: float = FW,  # size of the Fluid
        height: float = FH,  # size of the Fluid
        source_func=None,
        boundary_func=None,
        source_duration: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.h = size
        self.nx, self.ny = int(width / size), int(height / size)
        self.diff = diffusion
        self.visc = viscosity
        self.max_iter = max_iter

        self.source_fn = source_func
        self.source_duration = source_duration

        self.boundary_fn = boundary_func or self.fix_boundary
        self.init_fluid()

    def init_fluid(self, dt: float = 1 / 60) -> None:
        self.time = 0
        nx, ny = self.nx, self.ny
        h = self.h

        self.dgrid = np.zeros((nx + 2, ny + 2))
        self.ugrid = np.zeros((nx + 2, ny + 1))
        self.vgrid = np.zeros((nx + 1, ny + 2))
        self.pgrid = np.zeros((nx, ny))

        self.grid_indices = np.indices((self.nx + 2, self.ny + 2))
        self.u_idx = self.transform_coords(self.grid_indices[:, :, :-1], [-0.5, 0])
        self.v_idx = self.transform_coords(self.grid_indices[:, :-1], [0, -0.5])
        self.d_idx = self.transform_coords(self.grid_indices, [-0.5, -0.5])

        self.lap_matrix = get_laplacian_matrix(self.nx, self.ny)
        self.pressure_solver = factorized(self.lap_matrix.copy())
        self.sdgrid, self.sugrid, self.svgrid = self.source_fn(self)

        if self.diff:
            self.diffD_solver = factorized(
                np.eye(nx * ny) - self.diff * dt / h**2 * self.lap_matrix.copy()
            )

        if self.visc:
            self.diffU_solver = factorized(
                np.eye(nx * (ny + 1))
                - self.visc * dt / h**2 * get_laplacian_matrix(nx, ny + 1)
            )
            self.diffV_solver = factorized(
                np.eye((nx + 1) * ny)
                - self.visc * dt / h**2 * get_laplacian_matrix(nx + 1, ny)
            )

    def reset_fluid(self) -> None:
        self.time = 0
        self.dgrid *= 0
        self.ugrid *= 0
        self.vgrid *= 0
        self.pgrid *= 0

    @staticmethod
    def fix_boundary(self) -> None:
        for grid in [self.dgrid, self.ugrid, self.vgrid]:
            grid[0, :] = 0
            grid[-1, :] = 0
            grid[:, 0] = 0
            grid[:, -1] = 0

    def map_coords(self, ip: np.ndarray, coords: np.ndarray) -> np.ndarray:
        return map_coordinates(ip, coords, order=1, prefilter=False)

    def set_boundary(self) -> None:
        self.boundary_fn(self)

    @property
    def density(self) -> np.ndarray:
        return self.dgrid[1:-1, 1:-1]

    @property
    def velocity(self) -> np.ndarray:
        u = (self.ugrid[1:-1, 1:] + self.ugrid[1:-1, :-1]) / 2
        v = (self.vgrid[1:, 1:-1] + self.vgrid[:-1, 1:-1]) / 2
        return np.stack([v, u], axis=-1)

    def get_vorticity(self, vel: np.ndarray = None) -> np.ndarray:
        vel = vel if vel is not None else self.velocity.reshape(-1, 2)
        ux = vel[:, 0].reshape(self.nx, self.ny)
        uy = vel[:, 1].reshape(self.nx, self.ny)
        duy_dx = np.gradient(uy, self.d_idx[..., 0][1:-1][:, 0], axis=0)
        dux_dy = np.gradient(ux, self.d_idx[..., 1][0][1:-1], axis=1)
        return duy_dx - dux_dy

    def interpolate_grid(self, i, j, inter_type: Literal[0, 1, 2]) -> np.ndarray:
        match inter_type:
            case 0:
                i = np.clip(i, 1, self.nx)
                j = np.clip(j, 1, self.ny)
                return self.map_coords(self.dgrid, np.stack([i, j]))

            case 1:
                i = np.clip(i, 1, self.nx)
                j = np.clip(j - 0.5, 0, self.ny)
                return self.map_coords(self.ugrid, np.stack([i, j]))

            case 2:
                i = np.clip(i - 0.5, 0, self.nx)
                j = np.clip(j, 1, self.ny)
                return self.map_coords(self.vgrid, np.stack([i, j]))

    def transform_coords(self, coords: np.ndarray, offset: np.ndarray) -> np.ndarray:
        minxy = np.array([-self.width / 2, -self.height / 2])[None, None]
        offset = np.array(offset)[None, None]
        coords = coords.transpose((1, 2, 0)).astype(float) + offset
        return coords * self.h + minxy

    def solve_pressure(self) -> np.ndarray:
        div = (
            self.ugrid[1:-1, 1:]
            - self.ugrid[1:-1, :-1]
            + self.vgrid[1:, 1:-1]
            - self.vgrid[:-1, 1:-1]
        ) / self.h

        rhs = div.flatten() * self.h**2
        return self.pressure_solver(rhs).reshape(self.nx, self.ny)

    def add_source(self) -> None:
        if self.sdgrid is not None:
            self.dgrid += self.sdgrid
        if self.sugrid is not None:
            self.ugrid += self.sugrid[..., 1] if self.sugrid.ndim == 3 else self.sugrid
        if self.svgrid is not None:
            self.vgrid += self.svgrid[..., 0] if self.svgrid.ndim == 3 else self.svgrid
        self.set_boundary()

    def diffuse(self, diffuse_type: Literal[0, 1]) -> None:
        nx, ny = self.nx, self.ny
        if diffuse_type == 0:
            self.dgrid[1:-1, 1:-1] = self.diffD_solver(
                self.dgrid[1:-1, 1:-1].flatten()
            ).reshape(nx, ny)

        elif diffuse_type == 1:
            self.ugrid[1:-1, :] = self.diffU_solver(self.ugrid[1:-1].flatten()).reshape(
                nx, ny + 1
            )
            self.vgrid[:, 1:-1] = self.diffV_solver(
                self.vgrid[:, 1:-1].flatten()
            ).reshape(nx + 1, ny)
        else:
            raise ValueError("diffuse_type must be '0: density' or '1: velocity'")

    def project(self) -> None:
        self.set_boundary()

        self.pgrid = pgrid = self.solve_pressure()
        self.ugrid[1:-1, 1:-1] -= (pgrid[:, 1:] - pgrid[:, :-1]) / self.h
        self.vgrid[1:-1, 1:-1] -= (pgrid[1:, :] - pgrid[:-1, :]) / self.h

        self.set_boundary()

    def advect(self, advect_type: Literal[0, 1], dt: float = 1 / 60) -> None:
        dt0 = dt / self.h

        def get_ij_back(ij):
            i_back = ij[0] - dt0 * self.interpolate_grid(ij[0], ij[1], 2)
            j_back = ij[1] - dt0 * self.interpolate_grid(ij[0], ij[1], 1)
            return i_back, j_back

        if advect_type == 0:
            ij = self.grid_indices[:, 1:-1, 1:-1].astype(float)
            ib, jb = get_ij_back(ij)
            self.dgrid[1:-1, 1:-1] = self.interpolate_grid(ib, jb, 0)

        elif advect_type == 1:
            ij = self.grid_indices[:, 1:-1, :-1].astype(float)
            ij[1] += 0.5
            ib, jb = get_ij_back(ij)
            self.ugrid[1:-1, :] = self.interpolate_grid(ib, jb, 1)

            ij = self.grid_indices[:, :-1, 1:-1].astype(float)
            ij[0] += 0.5
            ib, jb = get_ij_back(ij)
            self.vgrid[:, 1:-1] = self.interpolate_grid(ib, jb, 2)
        else:
            raise ValueError("advect_type must be '0: density' or '1: velocity'")

    def start_simulation(self) -> None:
        self.add_updater(self.update_fluid)

    def stop_simulation(self) -> None:
        self.remove_updater(self.update_fluid)

    @staticmethod
    def update_fluid(self, dt: float = 1 / 60) -> None:
        if self.time <= self.source_duration and self.source_fn:
            self.add_source()
        # velocity step
        self.advect(1, dt)

        if self.visc:
            self.diffuse(1)
        self.project()

        # density step
        if self.diff:
            self.diffuse(0)

        self.advect(0, dt)
        self.time += dt

    def i2c(self, i: int, j: int) -> tuple[float, float]:
        """short for index to coordinate"""
        x = i / self.nx * self.width - self.width / 2
        y = j / self.ny * self.height - self.height / 2
        return x, y

    def c2i(self, x: float, y: float) -> tuple[int, int]:
        """short for coordinate to index"""
        i = np.array((x / self.width + 0.5) * self.nx).astype(int)
        j = np.array((y / self.height + 0.5) * self.ny).astype(int)
        return i, j

    def get_velocity_at(self, x: float, y: float) -> np.ndarray:
        i, j = self.c2i(x, y)
        i = np.clip(i, 0, self.nx - 1)
        j = np.clip(j, 0, self.ny - 1)
        return self.velocity[i, j]

    def get_grid(self, cell_size: float = 0.5, stroke_width: float = 2) -> NumberPlane:
        return NumberPlane(
            x_range=(-int(self.width / 2), int(self.width / 2), cell_size),
            y_range=(-int(self.height / 2), int(self.height / 2), cell_size),
            faded_line_ratio=0,
        ).set_stroke(GREY_B, stroke_width)

    def get_colorbar(self, text: str, vis):
        # the colorbar would adapt the color map of the visualization
        bar = Rectangle3D(0.25, self.height / 1.35, (2, self.ny))
        bar.to_edge(RIGHT, buff=1).fix_in_frame()
        bar.set_color_by_rgb_func(lambda p: vis.cmap(normalize_array(p[:, 1]))[..., :3])
        tex = VGroup(
            Text(f"{i} {text}", font_size=24) for i in ["low", "high"]
        ).fix_in_frame()
        tex[0].next_to(bar, DOWN, buff=0.15)
        tex[1].next_to(bar, UP, buff=0.15)
        return bar.add(tex)

    def visualize(
        self,
        vis_type: Literal["density", "velocity", "field", "energy"] = "density",
        opacity: float = 1.35,
        color_map: str | list[ManimColor] = "coolwarm",
        ranges: tuple[tuple[float, float]] = None,
        num_dots_per_frame: int = 100,
        stream_duration: float = 10,
        cell_size: float = 0.8,
        stroke_width: float = 2,
        make_3d: bool = False,
    ) -> Rectangle3D | DotCloud | VectorField:
        hfw, hfh = self.width / 2, self.height / 2
        h = self.h

        if not isinstance(color_map, str):
            cm = LinearSegmentedColormap.from_list("custom_cm", color_map)
            plt.colormaps.register(cm, name="custom_cm", force=True)
            color_map = "custom_cm"
        cmap = plt.get_cmap(color_map)

        def colorize(vis: Mobject) -> np.ndarray:
            pts = vis.get_points() - vis.get_center()
            vels = self.get_velocity_at(pts[:, 0], pts[:, 1])
            vels_norm = np.sqrt(vels[:, 0] ** 2 + vels[:, 1] ** 2)
            normalized_vel = normalize_array(vels_norm)
            if make_3d:
                pts[:, 2] = 2 * normalized_vel
            rgba = cmap(normalized_vel)

            if vis_type.startswith("d"):
                rgba[..., 3] = (
                    opacity * normalize_array(arr)
                    if (arr := self.density.flatten()).any()
                    else opacity / normalized_vel
                )
            elif vis_type.startswith("e"):
                rgba[..., 3] = opacity * vels_norm
            vis.set_rgba_array(rgba)

        if vis_type.startswith("d") or vis_type.startswith("e"):
            vis = Rectangle3D(2 * hfw - h, 2 * hfh - h, (self.nx, self.ny))
        elif vis_type.startswith("v"):
            stream_duration += self.time
            num_dots = int(np.sqrt(num_dots_per_frame))
            if ranges:
                xy00 = self.c2i(ranges[0][0], ranges[1][0])
                xy11 = self.c2i(ranges[0][1], ranges[1][1])
                ranges = [(xy00[0], xy11[0], num_dots), (xy00[1], xy11[1], num_dots)]
            else:
                ranges = [(0, self.nx, num_dots), (0, self.ny, num_dots)]

            def update_dots(dots: DotCloud, dt: float = 1 / 60) -> None:
                pt = dots.get_points()
                x, y = pt[:, 0], pt[:, 1]
                vel = np.hstack((self.get_velocity_at(x, y), np.zeros((len(x), 1))))
                new_pt = pt + vel * dt
                if self.time < stream_duration:
                    new_pt = np.vstack((points, new_pt))
                new_pt = list(
                    filter(
                        lambda x: (-hfw < x[0] < hfw) and (-hfh < x[1] < hfh), new_pt
                    )
                )
                dots.set_points(new_pt)

            points = [
                [*self.i2c(i, j), 0]
                for i in np.linspace(*ranges[0])
                for j in np.linspace(*ranges[1])
            ]
            vis = GlowDots(points=points).add_updater(update_dots)
        elif vis_type.startswith("f"):

            def field_func(x):
                vel = self.get_velocity_at(x[:, 0], x[:, 1]) + 0.01
                return np.hstack((vel, np.zeros((len(vel), 1))))

            grid = self.get_grid(cell_size, stroke_width)
            vis = VectorField(field_func, grid, tip_width_ratio=3, color_map=cmap)
            vis.always.update_vectors()
        else:
            raise ValueError(f"The {vis_type} hasn't implemented yet!")

        vis.cmap = cmap
        return vis if vis_type.startswith("f") else vis.add_updater(colorize)
