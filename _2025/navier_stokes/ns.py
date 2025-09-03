from manim_imports import *
from _2025.navier_stokes.fluid_simulation import Fluid

TITLE_CMAP = [CYAN, WHITE, VIOLET]
NS_CMAP = {"u": YELLOW_B, " p ": TEAL, "f_e": BLUE, "\\nu": RED_B, "\\rho": VIOLET}
POS_CMAP = {"x": CYAN, "y": RED_B, "z": YELLOW, "p": TEAL, r"\rho": VIOLET}

ACC = r"{\partial u \over \partial t}"
ADVEC = r"(u \cdot \nabla)u"
DIFF = r"\nu \nabla^2 u"
PRESS = r"\nabla p "

ADVEC_1D = r"u {\partial u \over \partial x}"
DIFF_1D = r"\nu {\partial^2 u \over \partial x^2}"
PRESS_1D = r"{\partial p \over \partial x}"
DU_DX = r"{\partial u \over \partial x}"
D2U_DX2 = r"{\partial^2 u \over \partial x^2}"

NS1 = f"{ACC} + {ADVEC} = {DIFF} - {{1 \\over \\rho}} {PRESS} + f_e"
NS2 = r"\nabla \cdot u = 0"
NS3 = r"u(\mathbf{x}, t_0) = u_0"
NS_REDUCED = f"{ACC} + {ADVEC} = {DIFF} - {PRESS}"
NS_1D = f"{ACC} + {ADVEC_1D} = {DIFF_1D} - {PRESS_1D}"
NS_NEWTON = (
    f"\\rho \\left({ACC} + {ADVEC} \\right) = \\rho {DIFF} - {PRESS} + \\rho f_e"
)


def focus_on(
    main_mob: Mobject,
    *mobs: Mobject,
    opacity: float = 0.2,
    only_stroke: bool = False,
    **kwargs,
) -> AnimationGroup:
    # find a better way
    if only_stroke:
        return AnimationGroup(
            main_mob.animate.set_stroke(opacity=opacity),
            *[mob.animate.set_stroke(opacity=1) for mob in mobs],
            **kwargs,
        )
    return AnimationGroup(
        main_mob.animate.set_opacity(opacity),
        *[mob.animate.set_opacity(1) for mob in mobs],
        **kwargs,
    )


def get_tex(*tex: str, t2c: dict = NS_CMAP, isolate: list = None, **kwargs) -> Tex:
    isolate = list(t2c.keys()) if isolate is None else isolate
    return Tex(*tex, t2c=t2c, isolate=isolate, **kwargs)


def get_group_tex(*eqs: str, **kwargs) -> VGroup[Tex]:
    return VGroup(get_tex(eq, **kwargs) for eq in eqs)


def get_ns_eqs() -> VGroup[Tex]:
    return get_group_tex(NS1, NS2, NS3, NS_REDUCED, isolate=[ACC, ADVEC, DIFF, PRESS])


def get_ns_1d() -> Tex:
    return get_tex(NS_1D, isolate=[ACC, ADVEC_1D, DIFF_1D, PRESS_1D])


def get_ns_newton() -> Tex:
    return get_tex(NS_NEWTON, isolate=[r"\rho"])


#  Special Cases
def vortex_fn(fluid: Fluid) -> tuple:
    rd = np.linalg.norm(fluid.d_idx, 2, axis=-1)
    radius = 1
    den = np.zeros(fluid.d_idx.shape[:-1])
    den[rd < radius] = 0.5

    w = 2.5
    ru = np.linalg.norm(fluid.u_idx, 2, axis=-1)
    rv = np.linalg.norm(fluid.v_idx, 2, axis=-1)
    u = w * fluid.u_idx[..., 0]
    v = -w * fluid.v_idx[..., 1]
    u[~(ru < radius)] = 0
    v[~(rv < radius)] = 0
    return den, u, v


def taylor_vortex(fluid: Fluid) -> tuple:
    amp = 2
    u = np.zeros_like(fluid.u_idx)
    v = np.zeros_like(fluid.v_idx)
    u[..., 1] = -np.cos(4 * fluid.u_idx[..., 0]) * np.sin(4 * fluid.u_idx[..., 1]) * amp
    v[..., 0] = np.sin(4 * fluid.v_idx[..., 0]) * np.cos(4 * fluid.v_idx[..., 1]) * amp
    return None, u, v


def inflow_fn(fluid: Fluid) -> tuple[np.ndarray]:
    pt = np.array([np.cos(-PI / 2), np.sin(-PI / 2)])
    nm = -pt  # normal
    pt *= 3.5
    blob_radius = 0.15
    vel = 2.5

    den = np.zeros(fluid.d_idx.shape[:-1])
    vu = np.zeros_like(fluid.u_idx)
    vv = np.zeros_like(fluid.v_idx)

    dmask = np.linalg.norm(fluid.d_idx - pt[None, None], 2, axis=-1) <= blob_radius
    den[dmask] = 0.1
    umask = np.linalg.norm(fluid.u_idx - pt[None, None], 2, axis=-1) <= blob_radius
    vu[umask] += nm[None] * vel
    vmask = np.linalg.norm(fluid.v_idx - pt[None, None], 2, axis=-1) <= blob_radius
    vv[vmask] += nm[None] * vel

    return den, vu, vv


def sphere_obstacle(coords) -> np.ndarray:
    center = np.array([-4.5, 0])
    radius = 0.5
    center = center.reshape(*[1] * len(coords.shape[:-1]), 2)
    return np.linalg.norm(coords - center, 2, axis=-1) - radius


def karman_boundary(fluid: Fluid) -> None:
    fluid.ugrid[sphere_obstacle(fluid.u_idx) < 0] = 0
    fluid.vgrid[sphere_obstacle(fluid.v_idx) < 0] = 0

    fluid.vgrid[0, :] = 2.5
    fluid.vgrid[:, 0] = 0
    fluid.vgrid[:, -1] = 0

    for grid in [fluid.ugrid, fluid.dgrid]:
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0


def karman_fn(fluid: Fluid) -> tuple:
    vel = np.zeros_like(fluid.v_idx)
    vel[..., 0] = 2.5
    return None, None, vel


class MagnifyingGlass(VGroup):
    def __init__(self, show_ob: Mobject, hide_ob: Mobject, **kwargs):
        lens = Circle(radius=1.8)
        h1 = Rectangle(0.1, 0.65).next_to(lens, DOWN, 0)
        h1.set_fill(GREY_B, 1)
        h2 = Rectangle(width=h1.get_width() * 4.5, height=1.65)
        h2.next_to(h1, DOWN, 0).set_fill(GREY_B, 1).round_corners(0.1)
        super().__init__(lens, h1, h2, **kwargs)

        self.set_stroke(GREY_B, 20)
        self.rotate(45 * DEG, about_point=lens.get_center())
        self.show_ob = show_ob.add_updater(lambda m: self.mask_ob(m))
        self.hide_ob = hide_ob.add_updater(lambda m: self.mask_ob(m, False))

    def mask_ob(self, ob: Mobject, show: bool = True) -> None:
        mobs = [ob] if ob.has_points() else ob.family_members_with_points()
        for mob in mobs:
            dtype = "rgba"
            if isinstance(mob, VMobject):
                dtype = "stroke_rgba"
                if mob.has_fill():
                    dtype = "fill_rgba"
            rgba = mob.data[dtype]
            opa = np.zeros(len(rgba))
            pts = mob.get_points() - self[0].get_center()
            mask = sqrt((pts**2).sum(1)) <= self[0].get_width() / 2
            opa[mask if show else ~mask] = 1
            rgba[:, 3] *= opa

    def follow_cursor(self, cursor: Point):
        self.add_updater(lambda g: g.shift(-(g[0].get_center() - cursor.get_center())))
        return self


KARMAN_CONFIG = {
    "source_func": karman_fn,
    "boundary_func": karman_boundary,
    "source_duration": 1,
    "height": 5,
}

INFLOW_CONFIG = {
    "source_func": inflow_fn,
    "source_duration": 10,
    "width": 8,
}

VORTEX_CONFIG = {
    "source_func": vortex_fn,
    "source_duration": 2,
    "width": 8,
}

TAYLOR_VORTEX_CONFIG = {
    "source_func": taylor_vortex,
    "source_duration": 2,
    "width": 8,
}


class Introduction(Scene):
    def construct(self):
        quote = TexText(
            r"``One could perhaps describe the situation by saying that\\",
            r"God is a Mathematician of a very high order and He used\\",
            "very advanced mathematics in constructing the Universe.''",
            font_size=48,
            t2c={"high order": YELLOW_B, "advanced mathematics": CYAN},
        )
        paul = Text("~ Paul Dirac", font_size=42)
        paul.next_to(quote, DOWN, buff=0.5).shift(4 * RIGHT)

        self.play(LaggedStart(Write(quote), FadeIn(paul), lag_ratio=0.5), run_time=5)
        self.wait(2)

        fluid = Fluid(size=0.08, **KARMAN_CONFIG)
        self.add(fluid)

        sphere = Circle(radius=0.55).move_to(4.5 * LEFT)
        sphere.set_fill(BLACK, opacity=1).set_stroke(BLACK, width=0)
        vis = fluid.visualize(color_map="coolwarm_r", opacity=0.15)
        vis.z_index = -1
        vis.clear_updaters()
        vis.set_color_by_gradient(PURE_RED, INDIGO).set_opacity(0.35)

        skip_time = 10
        for _ in range(skip_time * self.camera.fps):
            fluid.update_fluid(fluid, 1 / self.camera.fps)
        dots = fluid.visualize(
            "velocity",
            color_map="coolwarm_r",
            ranges=[(-7, -6.5), (-0.5, 0.5)],
            num_dots_per_frame=100,
            stream_duration=40,
        )
        dots.set_radius(0.025)
        dots.set_glow_factor(0.08)
        dots.z_index = 0

        fluid.start_simulation()
        self.add(dots)
        self.play(*map(FadeOut, [quote, paul]), FadeIn(vis), FadeIn(sphere))
        self.wait(10)

        rect = RoundedRectangle(11, 7)
        rect.set_fill(BLACK, 0.7)
        ns_eq = get_ns_eqs()[:3].arrange(DOWN, buff=0.5)
        title = TitleText("Navier - Stokes Equation", TITLE_CMAP).shift(0.5 * DOWN)

        self.play(Write(rect))
        self.play(Write(title), Wavy(ns_eq))
        self.wait(5)
        self.play(FlashAround(ns_eq[0], buff=0.35), run_time=2)
        self.wait(5)
        self.play(*map(FadeOut, [title, rect, ns_eq]))
        self.wait(5)


class Introduction2(Scene):
    def construct(self):
        frame = self.frame
        fluid = Fluid(0.1, **INFLOW_CONFIG)
        density = fluid.visualize("d", 1.5)
        self.add(fluid, density)

        fluid.start_simulation()
        self.wait()
        fluid.stop_simulation()

        initial_fluid = fluid.copy().clear_updaters()
        initial_density = initial_fluid.visualize("d", 1.5).clear_updaters()
        initial_vector = initial_fluid.visualize("f").clear_updaters()
        initial_vector.f_always.move_to(initial_density.get_center)
        self.remove(density)
        self.add(initial_density)

        rect = RoundedRectangle(
            stroke_color=WHITE,
            stroke_width=4,
            stroke_opacity=1,
            fill_opacity=0,
            corner_radius=0.25,
        ).surround(initial_density)

        ar = Arrow(LEFT, RIGHT, thickness=6).set_color(YELLOW)
        show_anim = Group(initial_density, ar, density)
        show_anim[1].next_to(show_anim[2], LEFT, buff=1)
        show_anim[0].next_to(show_anim[1], LEFT, buff=1)

        r1 = rect.copy().move_to(initial_density)
        r2 = rect.copy().move_to(density)

        vector = fluid.visualize("f", "bwr")
        time_txt = VGroup(
            TexText(f"{t} \\\\ t: 0.00", font_size=64).next_to(
                show_anim[i * 2], DOWN, buff=0.5
            )
            for i, t in enumerate(
                [
                    r"Initial Condition  $\rightarrow u(\mathbf{x}, 0) = u_0$",
                    "Prediction",
                ]
            )
        )
        sol_txt = Tex(r"u(\mathbf{x}, t)").next_to(ar, UP).scale(1.35)
        frame.move_to(show_anim[0])
        self.play(ShowCreation(initial_vector))
        self.wait()
        self.add(initial_vector, vector)

        self.play(
            frame.animate.set_width(show_anim[:3].get_width() + 4).move_to(show_anim),
            *map(FadeIn, [show_anim[1:], time_txt, r1, r2]),
            Write(sol_txt),
            run_time=2,
        )
        self.wait()
        sim_time_track = time_txt[1].make_number_changeable("0.00")
        time_track = ValueTracker(self.time)
        sim_time_track.f_always.set_value(time_track.get_value)

        fluid.start_simulation()
        self.play(time_track.animate.increment_value(5), run_time=5, rate_func=linear)
        fluid.stop_simulation()
        self.wait()

        self.play(
            density.copy().clear_updaters().animate.move_to(initial_density),
            vector.copy().clear_updaters().animate.move_to(initial_vector),
            *map(FadeOut, [initial_density, initial_vector]),
            time_track.animate.set_value(0),
            rate_func=linear,
        )

        fluid.start_simulation()
        self.play(time_track.animate.increment_value(5), run_time=5, rate_func=linear)

        ns_eq = get_ns_eqs()[0].fix_in_frame()
        rect = RoundedRectangle().surround(ns_eq).fix_in_frame()
        rect.set_stroke(WHITE).set_fill(BLACK, 0.8)
        self.play(
            ShowCreation(rect),
            Write(ns_eq),
            time_track.animate.increment_value(1),
            rate_func=linear,
        )
        self.play(time_track.animate.increment_value(5), run_time=5, rate_func=linear)
        fluid.stop_simulation()
        self.wait()


class ApplicationsOfNavierStokes(Scene):
    def construct(self):
        apps = VGroup(
            Text(t, font="Vesper Libre", font_size=30)
            for t in [
                "Weather Prediction",
                "Blood Flow Simulation",
                "Pollution Dispertion",
                "Industrial Fluids",
                "Aerodynamics",
                "Oceanography",
                "Computational Fluid Dynamics",
                "Magnetohydrodynamics",
            ]
        )
        apps.arrange_in_grid(4, h_buff=5.5, v_buff=1.5, fill_rows_first=True)
        ns_txt = Text("Navier - Stokes Equation", font="Delius Swash Caps")
        circle = Circle(stroke_color=GREY).move_to(ns_txt)
        circle.set_width(ns_txt.get_width() + 1, stretch=True)
        ns_txt.add(circle)
        self.play(Write(ns_txt))
        self.play(Wavy(apps))
        self.wait(2)


class TaoStrategies(Scene):
    def construct(self):
        img = ImageMobject("terence_tao")
        img.set_width(2).to_corner(UR, buff=0.25)

        title = TitleText("Strategies proposed by Terence Tao", gr=TITLE_CMAP)
        self.play(Write(title), FadeIn(img))
        self.wait()

        strategies = VGroup(
            Text(t)
            for t in [
                "1. Solve the Navier-Stokes equation exactly and explicitly.",
                "2. Discover a new globally controlled quantity which is both\n     coercive & either critical or subcritical.",
                "3. Discover a new method which yields global smooth solutions.",
            ]
        ).scale(0.9)
        strategies.arrange(DOWN, buff=0.5, aligned_edge=LEFT).next_to(title, DOWN, 1)
        info = Text(
            "*We'll see critical & subcritical quantities later in the video.",
            font="Delius Swash Caps",
        )
        info.scale(0.8).to_edge(DOWN, buff=0.35)
        self.play(Write(strategies), ApplyWave(img))
        self.wait()
        self.play(focus_on(strategies, strategies[1]), Write(info))
        self.wait()


class ShowingDivergentField(Scene):
    def construct(self):
        particles_charge = [1, -1]
        particles = VGroup(Dot() for _ in range(2)).arrange(RIGHT, buff=7)

        def div_func(p: np.ndarray):
            result = np.zeros_like(p)
            pos = array([par.get_center()[:2] for par in particles])
            for x, s in zip(pos, particles_charge):
                new_p = p - x
                if p.ndim != 1:
                    norm_p = np.linalg.norm(new_p, axis=1).reshape(-1, 1)
                else:
                    norm_p = np.linalg.norm(new_p)

                radius = 0.25 * np.ones_like(norm_p)

                nn_p = s * np.where(
                    norm_p > radius, new_p / norm_p**2, new_p / radius**2
                )
                result += nn_p
            return result

        nb = NumberPlane()
        vec = VectorField(div_func, nb).set_stroke(opacity=0.75)
        stream = StreamLines(div_func, nb, density=2.5)
        st = AnimatedStreamLines(stream)

        angles = linspace(0, TAU, 8, endpoint=False)
        out_arrows = VGroup(
            Vector(1.5 * RIGHT, buff=0.35).rotate_about_origin(ang) for ang in angles
        ).move_to(particles[0])
        in_arrows = VGroup(
            Arrow(1.5 * RIGHT, ORIGIN, buff=0.35).rotate_about_origin(ang)
            for ang in angles
        ).move_to(particles[1])
        self.add(vec, st)
        self.wait(10)

        pts = VGroup(Tex(i) for i in ["A", "B"])
        pts[0].next_to(out_arrows, DOWN, 0.5).add_background_rectangle(BLACK, 0.8)
        pts[1].next_to(in_arrows, DOWN, 0.5).add_background_rectangle(BLACK, 0.8)

        self.play(
            vec.animate.set_stroke(opacity=0.5),
            st.animate.set_stroke(opacity=0.5),
            Write(out_arrows),
            Write(pts[0]),
        )
        self.wait()
        self.play(Write(in_arrows), Write(pts[1]))
        self.wait()
        info_txt = TexText(
            r"...and we don't want such kind of a fluid flow.\\So, $\nabla \cdot u = 0$ ensures that.",
            t2c={" u ": YELLOW_B},
        )
        info_txt.to_edge(DOWN, 0.5).add_background_rectangle(BLACK, 0.5)
        self.play(Write(info_txt))
        self.wait()


class AboutNSAssumptions(Scene):
    def construct(self):
        title = TitleText("Navier-Stokes Equation", TITLE_CMAP)
        title.save_state()
        self.play(Write(title))
        self.wait()

        # Navier-Stokes is Newton's 2nd Law
        ns_eq = get_ns_eqs()[:3].arrange(DOWN, buff=0.5)
        desc = VGroup(
            TexText(i, font_size=34)
            for i in [
                "$u$: velocity of fluid",
                "$p$: pressure of fluid",
                "$\\nu$: viscosity of fluid",
                "$\\rho$: density of fluid",
                "$f$: external forces like gravity",
                r"$u(\mathbf{x}, t_0) = u_0$: initial condition",
            ]
        ).arrange_in_grid(3, fill_rows_first=False, aligned_edge=LEFT)
        desc.to_edge(DOWN, buff=0.35)
        self.play(Write(ns_eq), FadeIn(desc), run_time=2)
        self.wait(2)

        self.play(focus_on(ns_eq, ns_eq[-1]), focus_on(desc, desc[-1]))
        self.play(FlashAround(ns_eq[-1]), run_time=2)
        self.wait()
        self.play(FadeOut(desc), focus_on(ns_eq, ns_eq[1]))
        self.wait(2)

        ## Assumptions
        # 1. Divergence-less field

        divergence_tex = get_group_tex(
            r"\left({\partial \over \partial x}\hat{i} + {\partial \over \partial y}\hat{j} + {\partial \over \partial z}\hat{k}\right) \cdot (u_x \hat{i} + u_y\hat{j} + u_z \hat{k}) = 0",
            r"{\partial u_x \over \partial x} + {\partial u_y \over \partial y} + {\partial u_z \over \partial z} = 0",
        ).arrange(DOWN, buff=0.5)
        divergence_tex.shift(0.5 * DOWN)
        divergence_title = TitleText("Divergence - less Field")

        self.play(
            ReplacementTransform(title, divergence_title),
            ns_eq[1].animate.shift(2.35 * UP),
            ns_eq[0::2].animate.set_opacity(0),
            run_time=2,
        )
        self.wait()
        br = BraceText(ns_eq[1][r"\nabla \cdot u"], "Divergence")
        self.play(br.creation_anim(), focus_on(ns_eq[1], ns_eq[1][:3]))
        self.wait()
        self.play(*map(FadeOut, [br.label, br.brace]))
        self.wait()

        den_const = get_tex(r"\Rightarrow \rho \ \text{constant}")
        den_const.next_to(ns_eq[1], RIGHT).shift(0.05 * DOWN)

        self.play(ns_eq[1].animate.set_opacity(1))
        self.play(Write(den_const))
        self.wait()

        self.play(
            LaggedStart(
                *[
                    TransformMatchingShapes(
                        ns_eq[1][i].copy(),
                        divergence_tex[0][j],
                    )
                    for i, j in zip(
                        ["\\nabla", "\\cdot", "u", "= 0"],
                        [
                            r"\left({\partial \over \partial x}\hat{i} + {\partial \over \partial y}\hat{j} + {\partial \over \partial z}\hat{k}\right)",
                            "\\cdot",
                            r"(u_x \hat{i} + u_y\hat{j} + u_z \hat{k})",
                            "= 0",
                        ],
                    )
                ],
                lag_ratio=1,
            )
        )
        self.wait()
        self.play(Write(divergence_tex[1]))
        self.wait()

        more_about_txt = TexText(
            r"partial derivatives\\(*more about them later in video)"
        )
        more_about_txt.scale(0.6).to_corner(DL, buff=0.25)
        ar = Arrow(
            more_about_txt["derivatives"].get_corner(UR),
            divergence_tex[1].get_corner(DL),
        )

        self.play(
            *map(
                FlashAround,
                [
                    divergence_tex[1][f"{{\\partial u_{i} \\over \\partial {i}}}"]
                    for i in ["x", "y", "z"]
                ],
            ),
            Write(more_about_txt),
            GrowArrow(ar),
            run_time=2,
        )
        self.wait()
        self.remove_all_except(self.frame)

        note_txt = TexText(r"A Note on $x$ and $\mathbf{x}$")
        note_txt.to_edge(UP, buff=1).set_width(5)
        rect_x = divergence_tex[1][r"{\partial u_x \over \partial x}"].copy()
        rect_x.move_to(2 * LEFT)
        rect_boldx = ns_eq[-1].copy().set_opacity(1).next_to(rect_x, RIGHT, buff=2)

        boldx_tex = get_tex(r"\mathbf{x} = x \hat{i} + y \hat{j} + z \hat{k}")
        boldx_tex.move_to(2 * DOWN)

        self.play(*map(Write, [note_txt, rect_x, rect_boldx]))
        self.wait()
        self.play(FlashAround(rect_x))
        self.wait()
        self.play(FlashAround(rect_boldx))
        self.wait()
        self.play(Write(boldx_tex))
        self.wait()
        self.remove_all_except(self.frame)

        title.restore()
        ns_eq[0].set_opacity(1).center()
        self.play(Write(title), Write(ns_eq[0]))
        self.wait()

        vecs = get_group_tex(
            r"u = u_x \hat{i} + u_y \hat{j} + u_z \hat{k}",
            r"p = p_x \hat{i} + p_y \hat{j} + p_z \hat{k}",
            t2c={"u": YELLOW_B, "p": TEAL},
        ).arrange(DOWN, buff=0.5)
        vecs.next_to(ns_eq[0], DOWN, 0.5)
        self.play(Write(vecs))
        self.wait()
        self.play(
            FadeOut(vecs),
            focus_on(ns_eq[0], ns_eq[0]["\\nu"], ns_eq[0]["\\rho"]),
            focus_on(desc, desc[2:4], opacity=0),
        )
        self.wait()
        self.play(FadeOut(desc[2:4]), ns_eq[0].animate.set_opacity(1))

        brace_d = BraceText(ns_eq[0], "valid for any $d$-dimensional space")
        self.play(brace_d.creation_anim())
        self.wait(2)
        self.play(FadeOut(brace_d.brace), FadeOut(brace_d.label))
        self.wait()

        # 2. Newtonian Fluid
        desc.set_opacity(1)
        self.play(FadeIn(desc))
        self.wait()

        self.play(focus_on(ns_eq[0], ns_eq[0]["\\nu"]), focus_on(desc, desc[2]))
        self.wait()

        vis_title = TitleText("Viscosity")
        rects = VGroup(RoundedRectangle(height=3, width=5.3) for _ in range(2)).arrange(
            RIGHT, buff=1.5
        )
        texts = VGroup(
            TexText(t, t2c=NS_CMAP)
            for t in [
                r"honey\\(high $\nu$, slow flow)",
                r"water\\(low $\nu$, high flow)",
            ]
        )
        for i in range(2):
            texts[i].next_to(rects[i], DOWN, buff=0.5)

        self.play(
            *map(FadeOut, [desc, ns_eq[0]]),
            TransformMatchingShapes(title, vis_title),
            ShowCreation(rects),
            Write(texts),
        )
        self.wait()
        self.play(*map(FadeOut, [rects, texts]))
        self.wait()

        axes = Axes((0, 10, 1), (0, 6, 1))
        shear_txt = Text("Shear Rate").to_edge(DOWN, buff=0.4)
        visc_txt = (
            TexText("Viscosity ($\\nu$)").rotate_about_origin(PI / 2).to_edge(LEFT, 1.5)
        )
        newt_gr = axes.get_graph(lambda x: 3).set_color(CYAN)
        non_newt_gr = axes.get_graph(
            lambda x: x ** (-0.35), x_range=(0.01, 10, 0.1)
        ).set_color(RED_B)
        label1 = axes.get_graph_label(newt_gr, Text("Newtonian Fluid"))
        label2 = axes.get_graph_label(non_newt_gr, Text("Non-Newtonian Fluid"))

        self.play(*map(Write, [axes, shear_txt, visc_txt]))
        self.play(*map(ShowCreation, [newt_gr, non_newt_gr, label1, label2]))
        self.wait()
        self.remove_all_except(self.frame)

        # 3. Continuum Hypothesis
        fluid = Fluid(size=0.02, **INFLOW_CONFIG)
        self.add(fluid)

        dens = fluid.visualize()
        cb = fluid.get_colorbar("velocity", dens)
        dots = fluid.visualize("v", ranges=[(-0.25, 0.25), (-3.95, -3.85)])
        dots.set_radius(0.03)
        self.add(dens)
        glass = MagnifyingGlass(dots, dens)
        glass.move_to(9 * RIGHT + 5 * UP)
        self.add(dots, glass)

        fluid.start_simulation()
        self.play(ShowCreation(cb))

        img = ImageMobject("teaspoon").scale(0.6).shift(1.5 * UP)
        rect = RoundedRectangle().surround(img, buff=0.15)
        rect.set_stroke(WHITE, 4, 1).set_fill(opacity=0)
        img.add_to_back(rect)

        tea_txt = TexText(
            r"One teaspoon of water\\contains $\approx 1.67 \times 10^{23}$ water particles"
        )
        tea_txt.add_background_rectangle(BLACK, 0.7).next_to(rect, DOWN)
        self.play(*map(FadeIn, [img, tea_txt]))
        self.wait()
        self.play(*map(FadeOut, [img, tea_txt]))
        self.wait()

        self.play(
            glass.animate.shift(-glass[0].get_center()).set_anim_args(path_arc=-PI / 2)
        )
        self.wait()
        self.play(glass.animate.move_to(2 * UL))
        self.play(glass.animate.move_to(2 * UR))
        self.play(
            glass.animate.shift(-glass[0].get_center()).set_anim_args(path_arc=-PI / 2)
        )
        self.wait()
        self.play(FadeOut(glass), FadeOut(dens))
        dots.remove_updater(dots.updaters[-1])
        dens.remove_updater(dens.updaters[-1])
        self.wait(2)
        self.play(dots.animate.set_radius(0.18))
        self.add(dens)
        self.remove(dots)
        self.remove_all_except(self.frame)

        # 4. Isothermal
        title.restore()
        assumptions_txt = Text("Assumptions:").shift(DOWN)
        assumptions_txt.set_color_by_gradient(DARK_BLUE, WHITE, DARK_BLUE)
        ns_eq[0].set_opacity(1).shift(UP)
        self.play(*map(Write, [title, ns_eq[0]]))
        self.wait()

        assumptions = VGroup(
            TexText(t, t2c={"$\\rho$": VIOLET, "$\\nu$": RED_B}, font_size=35)
            for t in [
                r"1. Density ($\rho$) is constant",
                r"2. Fluid is Newtonian ($\nu$ is constant)",
                "3. Continuous Fluid",
                "4. Isothermal.",
            ]
        )
        assumptions.arrange_in_grid(v_buff=0.35, aligned_edge=LEFT)
        assumptions.next_to(assumptions_txt, DOWN, buff=0.5)
        self.play(
            LaggedStartMap(
                Write, VGroup(*[assumptions_txt, *assumptions]), lag_ratio=1, run_time=5
            )
        )
        self.wait()
        self.play(
            *map(FadeOut, [assumptions, assumptions_txt]), ns_eq[0].animate.center()
        )
        self.wait()

        # # Newton
        ns_ma = get_ns_newton()
        ns_eq[0].save_state()
        # show
        self.play(TransformMatchingStrings(ns_eq[0], ns_ma, path_arc=-PI / 2))

        b0 = BraceLabel(ns_ma[0], "m")
        b1 = BraceLabel(ns_ma[2:14], "a")
        b2 = BraceLabel(ns_ma[-12:], "F").shift(0.35 * DOWN)

        self.play(*map(lambda b: b.creation_anim(), [b0, b1, b2]))
        self.wait()
        self.play(WiggleOutThenIn(ns_ma[-12:]))
        self.wait()
        self.play(WiggleOutThenIn(ns_ma[2:14]))
        self.wait()

        self.play(
            *map(
                lambda b: AnimationGroup(FadeOut(b.brace), FadeOut(b.label)),
                [b0, b1, b2],
            ),
        )
        ns_eq[0].restore()
        self.play(TransformMatchingTex(ns_ma, ns_eq[0], path_arc=-PI / 2))
        self.wait()

        # each term brace
        braces_term = VGroup(
            BraceText(ns_eq[0][ACC], r"acceleration\\of fluid", UP, label_scale=0.7),
            BraceText(ns_eq[0][ADVEC], "advection", label_scale=0.7),
            BraceText(ns_eq[0][DIFF], "diffusion", UP, label_scale=0.7),
            BraceText(ns_eq[0][PRESS], r"pressure", UP, label_scale=0.7),
            BraceText(
                ns_eq[0]["f_e"],
                r"external\\forces\\(eg. gravity)",
                DOWN,
                label_scale=0.7,
            ),
            BraceText(
                ns_eq[0][f"{DIFF} - {{1 \\over \\rho}} {PRESS}"],
                "internal forces",
                DOWN,
                label_scale=0.7,
            ),
        )
        self.play(*[b.creation_anim() for b in braces_term])
        self.wait()

        # acc term
        term_info = VGroup(
            Text(i)
            for i in [
                "deals with how fluid's velocity changes with time",
                "transports the fluid in the direction of flow",
                "balances out the velocity profile",
                "initiates and maintains fluid motion",
                "sets up boundary condition",
            ]
        )
        term_info.next_to(ns_eq[0], DOWN, 1).shift(0.5 * LEFT)
        self.remove(braces_term)
        self.play(focus_on(ns_eq[0], ns_eq[0][ACC]), FadeIn(braces_term[0]))
        self.play(Write(term_info[0]))
        self.wait()

        # advection, diffusion, pressure, external force
        for i, term in enumerate([ADVEC, DIFF, PRESS, "f_e"]):
            self.play(
                FadeOut(braces_term[i]),
                FadeIn(braces_term[i + 1]),
                focus_on(ns_eq[0], ns_eq[0][term]),
                TransformMatchingStrings(term_info[i], term_info[i + 1]),
            )
            self.wait()

        self.play(
            ns_eq[0].animate.set_opacity(1),
            *map(FadeOut, [braces_term[-2], term_info[-1]]),
        )
        self.wait()

        # important terms
        self.play(
            focus_on(ns_eq[0], ns_eq[0][ADVEC], ns_eq[0][DIFF]),
            FadeIn(braces_term[1:3]),
        )
        self.wait()
        self.play(FadeOut(braces_term[1:3]))
        self.wait()

        # simplifying eq
        self.play(focus_on(ns_eq[0], ns_eq[0][:-8], ns_eq[0][-5:], opacity=0))
        self.wait()
        self.play(FlashAround(ns_eq[0]["\\nu"], 1.5), run_time=2)
        self.wait()
        self.play(focus_on(ns_eq[0], ns_eq[0][:-8], ns_eq[0][-5:-3], opacity=0))
        self.play(FlashAround(ns_eq[0][ADVEC], buff=0.15), run_time=2)
        self.wait()

        ns_sim = get_ns_eqs()[3]
        ns_sim.match_width(ns_eq[0][:-2]).move_to(ns_eq[0][:-2])
        self.play(TransformMatchingTex(ns_eq[0], ns_sim))
        self.remove(ns_eq[0])
        self.play(ns_sim.animate.shift(UP))

        ns_1d = get_ns_1d()
        ns_1d.match_width(ns_sim).scale(0.95).match_x(ns_sim).shift(DOWN)

        nsi = [ACC, "+", ADVEC, "=", DIFF, "-", PRESS]
        nsj = [ACC, "+", ADVEC_1D, "=", DIFF_1D, "-", PRESS_1D]
        self.play(
            TransformMatchingStrings(
                ns_sim.copy(),
                ns_1d,
                key_map={i: j for i, j in zip(nsi, nsj)},
            ),
            run_time=2,
        )
        self.wait(2)

        self.play(
            FadeOut(ns_sim),
            ns_1d.animate.center(),
            Transform(
                title, TitleText("Navier-Stokes Equation in 1D", [BLUE, WHITE, BLUE])
            ),
            run_time=2,
        )
        self.wait()


class FocusingOnDifficulty(Scene):
    def construct(self):
        self.sim_time = 0
        self.boundary_cn = True
        hw = int(FW / 2)
        ns_1d = get_ns_1d()
        self.play(FadeIn(ns_1d))
        self.wait()
        self.play(ns_1d.animate.to_corner(UR, 0.75))
        self.wait()

        def update_dots_stream(dots: GlowDots, dt: float = 1 / 60):
            pts = dots.get_points()
            x = pts[:, 0]
            x += 2 * dt
            pts[:, 0] = -hw * (x >= hw + 1) + x * (x < hw + 1)
            dots.set_points(pts)

        def update_vector_field(field: VGroup, dt: float = 1 / 60):
            for vec in field:
                vec.shift(dt * RIGHT)
                if vec.get_x() >= hw + 1:
                    vec.set_x(-hw)

        points = array([[x, 0, 0] for x in linspace(-hw, hw + 1, 200)])
        dots_stream = GlowDots(points=points, color=CYAN, radius=0.35, glow_factor=1.35)
        dots_discrete = DotCloud(points[::10], radius=0.15, color=CYAN)

        lines = VGroup(Line(LEFT_SIDE, RIGHT_SIDE) for _ in range(2))
        lines[0].set_y(dots_stream.get_top()[1])
        lines[1].set_y(dots_stream.get_bottom()[1])

        # adding arrow
        vector_field = VGroup(
            Arrow(
                0.5 * LEFT,
                0.5 * RIGHT,
                tip_angle=PI / 4,
                max_tip_length_to_length_ratio=0.25,
            ).next_to(dot, DR, buff=0.75)
            for dot in dots_discrete.get_points()[::2]
        ).set_color_by_gradient(VIOLET, INDIGO, RED, CYAN)

        vector_field.add_updater(update_vector_field)
        dots_stream.add_updater(update_dots_stream)
        dots_discrete.add_updater(update_dots_stream)

        fluid_flow = Group(lines, dots_stream, dots_discrete)
        flow_stream_txt = Text("Flow of Fluid").move_to(1.5 * DOWN)

        self.play(
            FadeIn(dots_stream),
            Write(flow_stream_txt),
            FadeIn(vector_field),
            GrowFromEdge(lines, LEFT),
        )
        self.wait(4)
        self.play(FadeIn(dots_discrete), dots_stream.animate.set_opacity(0.1))
        self.wait(4)

        num_points = 10
        self.nu = 0.5  # diffusion coefficient
        self.dx = 0.5  # step size
        self.time_factor = 1  # it slows the time down by a factor
        self.alpha = 0  # it decides the advection: None -> self advection

        axes = self.get_axes((0, 10), (0, 5), num_points=num_points)
        gr_discrete = self.get_graph(
            lambda x: 2.25 + sin(1.5 * x) + cos(x / 2), discrete=True, color=RED_B
        )
        gr = self.get_graph(self.graph_func)
        gr.save_state()

        self.play(
            FadeOut(flow_stream_txt),
            fluid_flow.animate.match_y(axes.x_axis),
            FadeOut(vector_field),
        )
        self.wait()
        dots_stream.clear_updaters()
        dots_discrete.clear_updaters()
        func = gr.get_function()

        # stop the time
        self.wait()

        # moving points
        p2n = axes.x_axis.p2n
        new_stream_pts = dots_stream.get_points().copy()
        new_stream_pts = axes.c2p((x := p2n(new_stream_pts)), func(x))

        new_discrete_pts = dots_discrete.get_points().copy()
        new_discrete_pts = axes.c2p((x := p2n(new_discrete_pts)), func(x))
        self.play(
            FadeOut(lines),
            dots_stream.animate.set_points(new_stream_pts),
            dots_discrete.animate.set_points(new_discrete_pts),
            Write(axes),
        )
        self.wait(2)

        # tracing the point along the x-axis
        pt = Point().move_to(axes.c2p(0, 0))
        h_line = axes.get_h_line_to_graph(p2n(pt.get_center()), gr)
        v_line = axes.get_v_line_to_graph(p2n(pt.get_center()), gr)
        self.add(h_line, v_line)

        h_line.add_updater(
            lambda h: h.become(axes.get_h_line_to_graph(p2n(pt.get_center()), gr))
        )
        v_line.add_updater(
            lambda v: v.become(axes.get_v_line_to_graph(p2n(pt.get_center()), gr))
        )
        pt.move_to(axes.c2p(0, 0))
        self.play(
            pt.animate.move_to(axes.c2p(10, 0)),
            ShowCreation(gr),
            FadeOut(dots_stream, run_time=2),
            FadeOut(dots_discrete, run_time=2),
            run_time=5,
        )
        self.play(pt.animate.move_to(axes.c2p(0, 0)), run_time=2)
        self.wait()
        self.remove(h_line, v_line)

        acc = ns_1d[ACC]
        advec = ns_1d[ADVEC_1D]
        diff = ns_1d[DIFF_1D]
        press = ns_1d[PRESS_1D]
        du_dx = ns_1d[DU_DX]
        d2u_dx2 = ns_1d[D2U_DX2]

        self.play(*map(FlashAround, [acc, du_dx, d2u_dx2, press]), run_time=2)
        self.play(focus_on(ns_1d, du_dx))

        fr = self.frame
        fr.save_state()
        gr_pts = gr.get_points()
        slope = Line(gr_pts[40], gr_pts[45]).set_stroke(YELLOW, 6)
        delta_x21 = DashedLine(
            slope.get_start(), [slope.get_end()[0], *slope.get_start()[1:]]
        )
        delta_u21 = DashedLine(delta_x21.get_end(), slope.get_end())

        delta_is_txt = Tex(r"\Delta \rightarrow \text{change}").scale(2)
        delta_is_txt.fix_in_frame().to_edge(UP, 2)

        deltau_txt, deltax_txt, deltat_txt = VGroup(
            Tex(f"\\Delta {x}", font_size=40) for x in ["u", "x", "t"]
        )
        deltau_txt.always.next_to(delta_u21, RIGHT, buff=0.15)
        deltax_txt.always.next_to(delta_x21, DOWN, buff=0.15)

        self.play(fr.animate.scale(1 / 2))
        self.wait()
        self.play(GrowFromEdge(delta_x21, LEFT), Write(deltax_txt), Write(delta_is_txt))
        self.wait()
        self.play(
            GrowFromEdge(delta_u21, DOWN), Write(deltau_txt), FadeOut(delta_is_txt)
        )
        self.wait()
        self.play(ShowCreation(slope), gr.animate.set_stroke(opacity=0.37))
        self.wait()

        delta_x21.add_updater(
            lambda d: d.become(
                DashedLine(
                    slope.get_start(), [slope.get_end()[0], *slope.get_start()[1:]]
                )
            )
        )
        delta_u21.add_updater(
            lambda d: d.become(DashedLine(delta_x21.get_end(), slope.get_end()))
        )
        delta_u_delta_x = Tex(r"{\Delta u \over \Delta x}").next_to(slope, UL, buff=0.2)
        du_dx = Tex(
            r"\lim_{\Delta x \rightarrow 0}{\Delta u \over \Delta x} = {du \over dx}"
        )
        du_dx.shift(
            delta_u_delta_x.get_center()
            - du_dx["{\Delta u \over \Delta x}"].get_center()
        )
        pu_px = Tex(
            r"\lim_{\Delta x \rightarrow 0}{\Delta u \over \Delta x} = {\partial u \over \partial x}"
        ).move_to(du_dx)
        pu_pt = Tex(
            r"\lim_{\Delta t \rightarrow 0}{\Delta u \over \Delta t} = {\partial u \over \partial t}"
        ).move_to(du_dx)
        u_txt = get_tex(r"u \equiv u(\mathbf{x}, t)").next_to(pu_px, UP, buff=0.2)

        self.play(
            TransformMatchingShapes(
                deltau_txt.copy(), delta_u_delta_x[r"\Delta u"], path_arc=PI / 2
            ),
            TransformMatchingShapes(
                deltax_txt.copy(), delta_u_delta_x[r"\Delta x"], path_arc=PI / 2
            ),
            Write(delta_u_delta_x["\\over"]),
        )
        self.wait()
        self.play(
            slope.animate.scale(0.05, about_point=slope.get_start()),
            deltau_txt.animate.scale(0.1),
            deltax_txt.animate.scale(0.1),
            TransformMatchingStrings(delta_u_delta_x, du_dx),
            rate_func=there_and_back,
            run_time=5,
        )
        self.wait()
        self.play(Write(u_txt), TransformMatchingStrings(du_dx, pu_px))
        self.wait()
        self.play(Restore(fr), gr.animate.set_stroke(opacity=1))
        self.wait()
        self.play(
            focus_on(ns_1d, acc),
            FadeOut(u_txt),
            pu_px.animate.shift(UR),
            *map(FadeOut, [slope, delta_u21, delta_x21, deltau_txt, deltax_txt]),
        )
        self.wait()
        pu_pt.move_to(pu_px)
        self.play(TransformMatchingStrings(pu_px, pu_pt))
        self.wait()
        ns_copy = ns_1d.copy()
        self.play(FadeOut(pu_pt))
        self.wait()

        self.play(
            ns_1d.animate.center().set_width(8), gr.animate.set_stroke(opacity=0.2)
        )
        self.play(focus_on(ns_1d, ns_1d[ACC], ns_1d[PRESS_1D]))
        self.wait()

        # diffusion term
        self.play(focus_on(ns_1d, diff))
        self.wait()
        self.play(
            ns_1d.animate.move_to(ns_copy).match_width(ns_copy),
            gr.animate.set_stroke(opacity=1),
        )
        self.wait()

        focused_particles = gr_discrete[1:4]
        h_lines = VGroup(axes.get_h_line(p.get_center()) for p in focused_particles)
        v_lines = VGroup(axes.get_v_line(p.get_center()) for p in focused_particles)

        u_txts, p_txts = VGroup(
            VGroup(
                Tex(f"{n}_{x}").next_to(line[x - 1], direc, buff=0.25)
                for x in [1, 2, 3]
            )
            for n, direc, line in zip(["u", "p"], [LEFT, DOWN], [h_lines, v_lines])
        )

        self.play(FadeOut(gr), FadeIn(gr_discrete), run_time=2)
        self.wait()
        self.play(
            focus_on(gr_discrete, focused_particles, opacity=0),
            Write(p_txts),
            Write(v_lines),
        )
        self.play(*[GrowFromEdge(l, RIGHT) for l in h_lines], Write(u_txts))
        self.wait()

        for i, (h_line, v_line, u_txt) in enumerate(zip(h_lines, v_lines, u_txts)):
            center = focused_particles[i].get_center()
            h_line.f_always.set_y(focused_particles[i].get_y)
            v_line.always.put_start_and_end_on(v_line.get_bottom(), center)
            u_txt.always.next_to(h_line, LEFT, 0.25)

        self.play(CircleIndicate(focused_particles[1]), run_time=2)
        self.play(*map(CircleIndicate, focused_particles[::2]), run_time=2)

        brace_x_diff = VGroup(
            BraceLabel(p_txts[:2], "\Delta x = 1", UP),
            BraceLabel(p_txts[1:], "\Delta x = 1", UP),
        )

        for br in brace_x_diff:
            br.brace.stretch(0.7, dim=0)

        self.play(brace_x_diff[1].creation_anim())
        self.wait()
        self.play(ReplacementTransform(brace_x_diff[1], brace_x_diff[0]))
        self.wait()
        self.play(FadeOut(brace_x_diff[0]))
        self.wait()

        diff_calcs_txt = get_group_tex(
            r"{u_3 + u_1 \over 2} - u_2",
            r"{1 \over 2} [(u_3 - u_2) - (u_2 - u_1)]",
            r"{1 \over 2} [\Delta u_{32} - \Delta u_{21}]",
            r"{1 \over 2} {\partial^2 u \over \partial x^2}",
        )
        diff_calcs_txt.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        diff_calcs_txt.next_to(ns_1d, DOWN, buff=0.25)
        diff_calcs_txt[1].next_to(diff_calcs_txt[0], DOWN, buff=0.35, aligned_edge=LEFT)
        diff_txt = TexText(r"Diffusion Term $\rightarrow$").next_to(
            diff_calcs_txt[0], LEFT, 0.25
        )
        u_txt_copy = u_txts.copy()

        self.play(
            Write(diff_txt), TransformMatchingShapes(u_txt_copy, diff_calcs_txt[0])
        )
        self.remove(u_txt_copy)
        self.wait()
        self.play(TransformMatchingStrings(diff_calcs_txt[0].copy(), diff_calcs_txt[1]))
        self.wait()

        braces_diff = VGroup(
            BraceLabel(diff_calcs_txt[1]["u_3 - u_2"], r"\Delta u_{32}"),
            BraceLabel(diff_calcs_txt[1]["u_2 - u_1"], r"\Delta u_{21}"),
        )

        for br in braces_diff:
            br.label["u"].set_color(YELLOW_B)
            br.label["u"].set_color(YELLOW_B)

        self.play(braces_diff[0].creation_anim(), braces_diff[1].creation_anim())
        self.wait()
        self.play(
            *[
                TransformMatchingShapes(
                    diff_calcs_txt[1][i].copy(),
                    diff_calcs_txt[2][j],
                )
                for i, j in zip(
                    [r"{1 \over 2} [", "(u_3 - u_2)", "- (u_2 - u_1)", "]"],
                    [r"{1 \over 2} [", r"\Delta u_{32}", r"- \Delta u_{21}", "]"],
                )
            ],
            *map(
                lambda br: AnimationGroup(FadeOut(br.label), FadeOut(br.brace)),
                braces_diff,
            ),
        )
        self.wait()

        br_diff_diff = BraceLabel(
            diff_calcs_txt[2][r"\Delta u_{32} - \Delta u_{21}"], r"\Delta \Delta u_{31}"
        )
        br_diff_diff.label["u"].set_color(YELLOW_B)

        self.play(br_diff_diff.creation_anim())
        self.wait()
        self.play(FadeOut(br_diff_diff.brace), FadeOut(br_diff_diff.label))
        self.wait()
        self.play(Write(diff_calcs_txt[3]))
        self.wait()

        self.play(
            FadeOut(diff_calcs_txt[1:-1]),
            diff_calcs_txt[-1].animate.next_to(
                diff_calcs_txt[0], DOWN, buff=0.25, aligned_edge=LEFT
            ),
        )
        ddux_in3d = get_tex(
            r"+ \ \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}"
        )
        ddux_in3d.next_to(diff_calcs_txt[-1], RIGHT, buff=0.18, aligned_edge=TOP)
        lap_br = BraceLabel([diff_calcs_txt[-1][3:], ddux_in3d], "\\nabla^2 u")
        lap_br = lap_br.change_label(r"\nabla^2 u", t2c={"u": YELLOW_B})
        lap_txt = Text('"Laplacian"').next_to(lap_br, DOWN, buff=0.35)

        self.play(
            Write(ddux_in3d),
            focus_on(diff_calcs_txt[-1], diff_calcs_txt[-1][3:], opacity=0),
        )
        self.wait()
        self.play(FadeIn(lap_br), Write(lap_txt))
        self.wait()
        self.play(
            focus_on(ns_1d, acc, diff, ns_1d["="]),
            *map(
                FadeOut,
                [
                    diff_calcs_txt[-1],
                    diff_calcs_txt[0],
                    diff_txt,
                    ddux_in3d,
                    lap_br,
                    lap_txt,
                ],
            ),
        )
        self.wait()

        gr_discrete.save_state()
        self.play(focused_particles[0].animate.shift(4 * DOWN))
        self.wait()

        def update_p2_diff(p, dt=1 / 60):
            p1, p2, p3 = [pt.get_center().copy() for pt in focused_particles]
            p2[1] += self.nu * (p1[1] + p3[1] - 2 * p2[1]) * dt
            p.move_to(p2)

        focused_particles[1].add_updater(update_p2_diff)
        self.wait(2)
        focused_particles[1].clear_updaters()

        self.play(
            focused_particles[0].animate.shift(4 * UP),
            focused_particles[2].animate.shift(4 * UP),
        )
        focused_particles[1].add_updater(update_p2_diff)
        self.wait(2)
        focused_particles[1].clear_updaters()
        self.wait()

        self.play(
            *map(FadeOut, [h_lines, v_lines, u_txts, p_txts]),
            gr_discrete.animate.restore().set_opacity(1),
        )
        gr_discrete.add_updater(self.update_graph)
        self.wait(4)

        # diff simulate
        self.play(FadeOut(gr_discrete), ShowCreation(gr))
        self.time_factor = 0.2
        gr.add_updater(self.update_graph)
        self.wait(10)

        gr_discrete.clear_updaters()
        gr.clear_updaters()
        self.time_factor = 1

        self.remove(ns_1d)
        self.add(ns_1d)
        self.play(
            ns_1d.animate.set_opacity(0.2),
            FadeOut(gr),
            Restore(gr_discrete),
            advec.animate.set_opacity(1),
        )
        gr.restore()
        self.play(
            focus_on(focused_particles, focused_particles[:2], opacity=0),
            *map(Write, [u_txts[:2], h_lines[:2], v_lines[:2], p_txts[:2]]),
            run_time=2,
        )
        self.wait()

        self.play(FadeIn(brace_x_diff[0]))
        self.wait()
        self.play(FadeOut(brace_x_diff[0]))
        self.wait()

        advec_txt = get_group_tex(
            r"\text{Advection Term} \rightarrow u_2 (u_2 - u_1)",
            r"\text{Advection Term} \rightarrow u_2 \Delta u_{21}",
        )
        advec_txt.arrange(ORIGIN, aligned_edge=LEFT).next_to(
            ns_1d, DOWN, buff=0.5
        ).shift(2 * LEFT)

        self.play(Write(advec_txt[0]))
        self.wait()
        self.play(TransformMatchingStrings(advec_txt[0], advec_txt[1]))
        self.wait()

        advec_eq = get_group_tex(f"{ACC} + {ADVEC_1D} = 0", f"{ACC} = -{ADVEC_1D}")
        advec_eq.arrange(ORIGIN, aligned_edge=LEFT).next_to(advec_txt, DOWN, buff=0.5)
        self.play(focus_on(ns_1d, acc, advec), Write(advec_eq[0]))
        self.wait()
        self.play(TransformMatchingStrings(advec_eq[0], advec_eq[1], path_arc=PI / 2))
        self.wait()

        def update_p2_advec(p, dt=1 / 60):
            p1, p2 = [
                self.axes.p2c(pt.get_center().copy()) for pt in focused_particles[:2]
            ]
            p2[1] -= p2[1] * (p2[1] - p1[1]) * dt
            p.move_to(self.axes.c2p(*p2))

        focused_particles[1].add_updater(update_p2_advec)
        self.wait(2)
        focused_particles[1].clear_updaters()

        self.play(focused_particles[0].animate.shift(4 * DOWN))

        focused_particles[1].add_updater(update_p2_advec)
        self.wait(2)
        focused_particles[1].clear_updaters()

        self.play(
            *map(FadeOut, [h_lines, v_lines, u_txts, p_txts, advec_txt, advec_eq]),
            FadeOut(gr_discrete),
            ShowCreation(gr),
        )
        gr_discrete.restore()

        hill = VMobject()
        hill.set_points_smoothly(gr.get_points()[35:65])
        hill.data["stroke_rgba"][:30] = np.array([*hex_to_rgb(RED), 1])
        hill.data["stroke_rgba"][30:] = np.array([*hex_to_rgb(YELLOW), 1])
        hill.set_stroke(width=8)

        down_arr = VGroup(
            Arrow(
                hill.get_points()[i],
                hill.get_points()[i] + 0.5 * DOWN,
                buff=0.1,
                thickness=1.35,
            )
            for i in range(2, 30, 6)
        ).set_color(RED)

        up_arr = VGroup(
            Arrow(
                hill.get_points()[i],
                hill.get_points()[i] + 0.5 * UP,
                buff=0.1,
                thickness=1.35,
            )
            for i in range(35, 59, 6)
        ).set_color(YELLOW)

        self.play(
            fr.animate.set_width(hill.get_width() * 2).move_to(hill),
            ShowCreation(hill),
            gr.animate.set_stroke(opacity=0.5),
            run_time=2,
        )
        self.wait()
        self.play(Write(up_arr), Write(down_arr))
        self.wait()

        self.boundary_cn = False
        self.nu = 0
        self.alpha = None
        hill.add_updater(self.update_graph)
        self.play(FadeOut(down_arr), FadeOut(up_arr))
        self.wait(6)
        hill.clear_updaters()

        steep_curve_txt = Text("Steep Curve\n").scale(0.5).shift(2 * DOWN)
        steep_ar = Arrow(
            steep_curve_txt.get_corner(UR),
            steep_curve_txt.get_corner(UR) + 0.75 * UR,
            buff=0.1,
            thickness=2,
        ).set_color(TEAL)

        self.play(Write(steep_curve_txt), GrowArrow(steep_ar))
        self.wait()
        self.play(
            *map(FadeOut, [steep_curve_txt, steep_ar, hill]),
            fr.animate.set_width(FW).center(),
            gr.animate.set_stroke(opacity=1),
        )
        self.boundary_cn = True
        gr.save_state()
        gr.add_updater(self.update_graph)
        self.wait(5)
        gr.clear_updaters()

        self.play(FlashAround(ns_1d[ADVEC_1D]))
        self.wait()

        self.play(ns_1d.animate.set_opacity(1))
        self.wait()

        self.nu = 0.5
        gr.add_updater(self.update_graph)
        self.wait(10)
        gr.clear_updaters()

    def get_axes(
        self, x_range=(0, 10), y_range=(0, 6), num_points=10, labels=["x", "u"]
    ):
        self.axes_x_range = np.linspace(*x_range, num_points * 4)
        dx = self.axes_x_range[1] - self.axes_x_range[0]
        self.axes = Axes(
            (*x_range, max(4 * dx, 1)),
            (*y_range, 1),
            height=6,
            axis_config=dict(include_tip=True),
        )
        self.axes.add(self.axes.get_axis_labels(*labels))
        return self.axes

    def get_graph(self, func, color=CYAN, discrete: bool = False):
        ax = self.axes
        self.graph_func = func

        if discrete:
            pts = array([[x, func(x)] for x in self.axes_x_range])[::4]
            return VGroup(
                Dot(point=pt, radius=0.12, fill_color=color)
                for pt in ax.c2p(pts[:, 0], pts[:, 1])
            )

        return ax.get_graph(func, x_range=ax.x_range).set_color(CYAN)

    def update_graph_points(self, pts, dt: float = 1 / 60) -> None:
        x, y = self.axes.p2c(pts)
        dx = self.dx

        y_change = np.zeros_like(y)
        alpha = y[1:-1] if self.alpha is None else self.alpha

        advec_eq = -alpha * (y[1:-1] - y[:-2]) / dx
        diff_eq = self.nu * (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2

        y_change[1:-1] = (advec_eq + diff_eq) * dt

        if self.boundary_cn:
            # (Periodic) Boundary Conditions
            alpha = y[0] if self.alpha is None else self.alpha
            y_change[0] = (
                -alpha * (y[0] - y[-2]) / dx
                + self.nu * 2 * (y[1] - 2 * y[0] + y[-2]) / dx**2
            ) * dt
            y_change[-1] = y_change[0]
        y += y_change
        return self.axes.c2p(x, y)

    def update_graph(self, gr, dt: float = 1 / 60) -> None:
        discrete = isinstance(gr, VGroup)
        dt /= self.time_factor
        self.sim_time += dt

        if discrete:
            pts = np.array([g.get_center() for g in gr])
        else:
            pts = gr.get_points()

        pts = self.update_graph_points(pts, dt)

        if discrete:
            for i, pt in enumerate(pts):
                gr[i].move_to(pt)
        else:
            gr.set_points(pts)


class ScalingOfEquation(Scene):
    def construct(self):
        ns_eq = get_ns_eqs()[3].shift(UP)
        definitions = get_tex(r"u \equiv u(\mathbf{x}, t)\\ p \equiv p(\mathbf{x}, t)")
        definitions.next_to(ns_eq, DOWN, 0.75)

        self.play(LaggedStart(Write(ns_eq), Write(definitions), lag_ratio=0.6))
        self.wait()

        title = TitleText("Scaling the Equation", TITLE_CMAP)
        scale_txt = Text("Scale Invariant").next_to(ns_eq[ACC], DL, buff=0.5)
        ar = Arrow(scale_txt.get_top(), ns_eq[ACC].get_left(), path_arc=-PI / 2)
        self.play(Write(title))
        self.wait(2)
        self.play(FlashAround(ns_eq), Write(scale_txt), Write(ar), run_time=2)
        self.wait(2)
        self.play(FadeOut(scale_txt), FadeOut(ar), FadeOut(ns_eq), FadeOut(definitions))
        self.wait(2)

        scale_qty = get_group_tex(
            r"\mathbf{x} \mapsto \mathbf{y} = {\mathbf{x} \over \lambda}",
            r"t \mapsto s = \lambda^{\alpha} t",
            r"u \mapsto u_{\lambda} = \lambda^{\beta} u",
            r" p \mapsto p _{\lambda} = \lambda^{\gamma} p ",
        )

        lambda_ge_0 = Tex(r"\lambda > 0").next_to(scale_qty[0], RIGHT, 2)
        self.play(Write(scale_qty[0]))
        self.play(Write(lambda_ge_0))
        self.wait(2)

        coarse_grid = NumberPlane(
            (0, 7, 1),
            (0, 7, 1),
            faded_line_ratio=0,
        ).prepare_for_nonlinear_transform()
        coarse_grid.set_stroke(CYAN, 2)
        fine_grid = NumberPlane(
            (0, 7, 0.5),
            (0, 7, 0.5),
            faded_line_ratio=0,
            background_line_style=dict(
                stroke_color=RED_A, stroke_width=2, stroke_opacity=1
            ),
        ).prepare_for_nonlinear_transform()
        fine_grid.set_stroke(RED_A, 2)

        self.play(
            *map(FadeOut, [title, lambda_ge_0]),
            scale_qty[0].animate.to_edge(RIGHT, buff=1),
            ShowCreation(coarse_grid),
        )
        self.wait()
        lambda_tracker = ValueTracker(2)

        coarse_grid.save_state()
        fine_grid.save_state()

        def update_fine_grid(grid: NumberPlane):
            grid.restore()
            grid.x_axis.x_step = 1 / lambda_tracker.get_value()
            grid.y_axis.x_step = 1 / lambda_tracker.get_value()

            grid.remove(grid.background_lines, grid.faded_lines)
            grid.init_background_lines()
            grid.prepare_for_nonlinear_transform()

        coarse_grid.add_updater(lambda m: m.restore())
        fine_grid.add_updater(update_fine_grid)

        glass = MagnifyingGlass(fine_grid, coarse_grid)
        glass.move_to(8.2 * RIGHT + 5.25 * UP)
        self.add(fine_grid, coarse_grid, glass)

        self.play(
            glass.animate.shift(-glass[0].get_center()).set_anim_args(path_arc=-PI / 2),
            run_time=2,
        )
        self.play(glass.animate.scale(1.5))

        for i in [0.5, 4, 8]:
            self.play(lambda_tracker.animate.set_value(i))
            self.wait()

        self.play(
            glass.animate.move_to(8 * RIGHT + 4 * UP).set_anim_args(path_arc=-PI / 2)
        )
        self.wait()
        coarse_grid.clear_updaters()
        fine_grid.clear_updaters()
        self.play(*map(FadeOut, [coarse_grid, fine_grid, glass]))

        self.play(
            scale_qty.animate.arrange(DOWN, buff=0.5, aligned_edge=LEFT).set_opacity(1),
            run_time=2,
        )
        self.wait(2)

        u_lambda_form = get_tex(
            r"u_{\lambda}(\mathbf{x}, t) = \lambda^{\beta} u \left({\mathbf{x} \over \lambda}, \lambda^{\alpha} t \right) = \lambda^{\beta} u(\mathbf{y}, s)"
        )
        self.play(
            TransformMatchingTex(scale_qty[2].copy(), u_lambda_form),
            scale_qty.animate.set_width(2.35)
            .arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            .to_corner(UR, buff=0.5),
        )
        self.wait(2)
        self.play(focus_on(u_lambda_form, u_lambda_form[:20]))
        self.wait()
        scale_qty.set_opacity(1)
        self.play(focus_on(u_lambda_form, u_lambda_form[:20], opacity=0))

        acc_term = get_group_tex(
            r"{\partial  \over \partial t} u_{\lambda}(\mathbf{x}, t) = {\partial \over \partial t} \left(\lambda^{\beta} u\left({\mathbf{x} \over \lambda}, \lambda^{\alpha} t \right) \right)",
            r"{\partial  \over \partial t} u_{\lambda}(\mathbf{x}, t) = \lambda^{\beta}{\partial \over \partial t} \left( u\left({\mathbf{x} \over \lambda}, \lambda^{\alpha} t \right) \right)",
            r"{\partial  \over \partial t} u_{\lambda}(\mathbf{x}, t) = \lambda^{\beta} \cdot \lambda^{\alpha}{\partial u \over \partial t}\left({\mathbf{x} \over \lambda}, \lambda^{\alpha} t \right)",
            r"{\partial  \over \partial t} u_{\lambda}(\mathbf{x}, t) = \lambda^{\beta + \alpha} {\partial u \over \partial t}\left({\mathbf{x} \over \lambda}, \lambda^{\alpha} t \right)",
        )
        acc_term[1:].set_opacity(0)
        # acc term
        self.play(
            Transform(
                u_lambda_form[r"u_{\lambda}(\mathbf{x}, t)"],
                acc_term[0][r"u_{\lambda}(\mathbf{x}, t)"],
            ),
            Transform(u_lambda_form[r"="], acc_term[0][r"="]),
            u_lambda_form[8:20].animate.move_to(
                acc_term[0][
                    r"\lambda^{\beta} u\left({\mathbf{x} \over \lambda}, \lambda^{\alpha} t \right)"
                ].get_center()
                + 0.27 * LEFT
            ),
            FadeIn(acc_term[0]),
        )
        self.wait()
        self.remove(u_lambda_form)
        self.play(acc_term.animate.arrange(DOWN, buff=0.5, aligned_edge=LEFT))
        self.wait()

        acc_term.set_opacity(1)
        self.remove(acc_term[1:])
        self.play(TransformMatchingTex(acc_term[0].copy(), acc_term[1]))
        self.wait()

        acc_term[2][14:17].set_opacity(0)
        self.play(
            TransformMatchingTex(
                acc_term[1].copy(),
                acc_term[2],
            )
        )
        self.play(
            ApplyMethod(
                (sp := acc_term[2][27:29].copy()).move_to,
                acc_term[2][15:17],
                path_arc=PI / 2,
            ),
            acc_term[2][14:15].animate.set_opacity(1),
        )
        self.wait(2)
        acc_term[15:17].set_opacity(1)

        self.play(TransformMatchingTex(acc_term[2].copy(), acc_term[3]))
        self.wait(2)
        self.play(
            sp.animate.set_opacity(0),
            acc_term[:3].animate.set_opacity(0),
            acc_term[3].animate.to_edge(UP, buff=1),
        )
        self.wait()

        scaled_terms = get_group_tex(
            r"\nabla u \mapsto {1 \over \lambda}\nabla u = \lambda^{-1}\nabla u",
            r"\nabla^2 u \mapsto {1 \over \lambda^2}\nabla^2 u = \lambda^{-2} \nabla^2 u",
            r"(\lambda^{\beta}u \cdot \lambda^{-1}\nabla)\lambda^{\beta} u = \lambda^{2 \beta - 1} (u \cdot \nabla)u",
            r"(\lambda^{-2} \nabla^2) \lambda^{\beta} u = \lambda^{\beta - 2} \nabla^2 u",
            r"(\lambda^{-1} \nabla) \lambda^{\gamma} p = \lambda^{\gamma - 1} \nabla p ",
        )
        scaled_terms[:2].arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        scaled_terms[2:].arrange(DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(scaled_terms[:2]), run_time=2)
        self.wait(2)
        self.play(
            *map(
                FlashAround,
                [scaled_terms[0]["\\lambda^{-1}"], scaled_terms[1]["\\lambda^{-2}"]],
            ),
            run_time=2,
        )
        self.wait(2)
        self.play(FadeOut(scaled_terms[:2]), Write(scaled_terms[2:]))
        self.wait(2)

        self.play(
            focus_on(acc_term[3], acc_term[3][12:16]),
            focus_on(scaled_terms[2], scaled_terms[2][14:19]),
            focus_on(scaled_terms[3], scaled_terms[3][11:15]),
            focus_on(scaled_terms[4], scaled_terms[4][10:14]),
        )
        self.wait()
        equating_terms = get_tex(
            r"\beta + \alpha = 2 \beta - 1 = \beta - 2 = \gamma - 1"
        ).next_to(scaled_terms, DOWN, buff=0.75)
        # c5
        self.play(
            TransformFromCopy(acc_term[3][13:16], equating_terms[:3], path_arc=PI / 2),
            TransformFromCopy(
                scaled_terms[2][15:19], equating_terms[4:8], path_arc=PI / 2
            ),
            TransformFromCopy(
                scaled_terms[3][12:15], equating_terms[9:12], path_arc=PI / 2
            ),
            TransformFromCopy(
                scaled_terms[4][11:14], equating_terms[-3:], path_arc=PI / 2
            ),
            run_time=2,
        )
        self.play(Write(equating_terms["="]))
        self.wait(2)

        final_values = get_tex(r"\beta = -1; \ \alpha = -2; \ \gamma = -2")
        final_values.next_to(equating_terms, DOWN, 0.5)
        self.play(Write(final_values), run_time=2)
        self.wait(2)

        final_scale_qty = get_group_tex(
            r"\mathbf{x} \mapsto \mathbf{y} = {\mathbf{x} \over \lambda}",
            r"t \mapsto s = {t \over \lambda^2}",
            r"u \mapsto u_{\lambda} = {1 \over \lambda} u",
            r" p \mapsto p _{\lambda} = {1 \over \lambda^2} p ",
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        final_scale_qty.to_edge(RIGHT, buff=2)
        self.play(
            *map(FadeOut, [acc_term[3], equating_terms, scaled_terms[2:]]),
            final_values.animate.to_edge(UP, buff=0.5),
            scale_qty.animate.match_width(final_scale_qty)
            .arrange(DOWN, buff=0.8, aligned_edge=LEFT)
            .center()
            .to_edge(LEFT, buff=2),
            run_time=2,
        )
        ar = Arrow(LEFT, RIGHT).set_color(CYAN)
        self.wait()
        self.play(Write(final_scale_qty), GrowArrow(ar))
        self.wait(2)

        final_u_lambda = get_tex(
            r"u{_\lambda} (\mathbf{x}, t) = {1 \over \lambda}u\left({\mathbf{x} \over \lambda}, {t \over \lambda^2}\right)"
        )
        self.remove_all_except(self.frame)
        self.play(Write(final_u_lambda))
        self.wait()

        # u and u_lambda
        def get_fine_squares(idx: int):
            lamb = int(1 / fine_grid.x_axis.x_step)  # has to be an int?
            frc = fine_grid.x_range[1] * lamb  # rows or columns

            cri = idx // coarse_grid.x_range[1]  # row index
            cci = idx % coarse_grid.x_range[1]  # column index

            frir = range(cri * lamb, cri * lamb + lamb)  # row index range
            fcir = range(cci * lamb, cci * lamb + lamb)  # column index range

            return VGroup(fine_squares[frc * r + c] for r in frir for c in fcir)

        coarse_squares = self.get_squares(coarse_grid)
        fine_squares = self.get_squares(fine_grid)

        self.remove(coarse_grid, fine_grid)
        # self.add(coarse_squares, fine_squares)

        u_grids = VGroup(fine_squares, coarse_squares)
        u_grids.arrange(RIGHT, buff=1)
        u_grids.set_width(FW / 1.1).shift(0.5 * UP)

        u_txt, u_lambda_txt = get_group_tex(
            "u",
            r"u_{\lambda}; \left({1 \over \lambda} \text{smaller}, {1 \over \lambda^2} \text{slower}\right)",
        )
        u_txt.next_to(coarse_squares, DOWN, buff=0.5)
        u_lambda_txt.next_to(fine_squares, DOWN, buff=0.15)

        self.play(
            LaggedStart(*map(Write, coarse_squares)),
            LaggedStart(
                *(Write(get_fine_squares(i)) for i in range(len(coarse_squares)))
            ),
            FadeOut(final_u_lambda),
            *map(Write, [u_txt, u_lambda_txt]),
            run_time=2,
        )
        self.wait()

        self.play(
            LaggedStart(*map(Indicate, coarse_squares)),
            LaggedStart(
                *(Indicate(get_fine_squares(i)) for i in range(len(coarse_squares)))
            ),
            run_time=4,
        )
        self.wait()

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        coarse_squares[i].animate.set_fill(opacity=0.65),
                        get_fine_squares(i).animate.set_fill(opacity=0.65),
                    )
                    for i in np.random.randint(0, 45, 5)
                ],
                lag_ratio=0.9,
            ),
            run_time=2,
            rate_func=there_and_back,
        )
        self.wait(2)
        self.play(
            coarse_squares.animate.set_fill(opacity=0),
            fine_squares.animate.set_fill(opacity=0),
        )
        self.wait()
        self.remove_all_except(self.frame)

    @staticmethod
    def get_squares(grid: NumberPlane, step: int = None) -> VGroup:
        x_max = grid.x_range[1]
        step = int(1 / grid.x_axis.x_step) if step is None else step
        gsquares = VGroup(
            Square(side_length=step).set_stroke(width=2)
            for _ in range((x_max * step) ** 2)
        )
        gsquares.arrange_in_grid(buff=0).match_width(grid)
        return gsquares.match_color(grid)


class LpNorm(Scene):
    def construct(self):
        # measure vector lengths
        vecs = VGroup(Arrow(DOWN, UP, buff=0) for _ in range(2))
        vecs.arrange(RIGHT, buff=1).set_color_by_gradient(CYAN, RED_B)
        t1 = Tex(r"v_1 = -2 \hat{i} + 5 \hat {j}", isolate=["v_2"])
        t2 = Tex(r"v_2 = 4 \hat{i} + 3 \hat {j}", isolate=["v_1"])
        t1.next_to(vecs[0], LEFT, buff=0.5)
        t2.next_to(vecs[1], RIGHT, buff=0.5)

        axes = Axes((-14, 14, 2), (-8, 8, 2), height=FH, width=FW)
        is_great = Tex(r"\overset{?}{>}")

        self.play(Write(vecs), Write(is_great))
        self.wait()
        self.play(Write(t1), Write(t2))
        self.wait()
        v1 = axes.c2p(-2, 5)
        v2 = axes.c2p(4, 3)
        self.play(
            ShowCreation(axes),
            vecs[0].animate.put_start_and_end_on(axes.c2p(0, 0), v1),
            vecs[1].animate.put_start_and_end_on(axes.c2p(0, 0), v2),
            t1.animate.next_to(v1, UL),
            t2.animate.next_to(v2, UR),
            FadeOut(is_great),
        )

        t1_norm = VGroup(
            Tex(i, isolate=["v_1"])
            for i in [r"|v_1| = \sqrt{2^2 + 5^2}", r"|v_1| = \sqrt{29}"]
        )
        t1_norm.arrange(ORIGIN, aligned_edge=LEFT).move_to(4.5 * LEFT + DOWN)
        t2_norm = VGroup(
            Tex(j, isolate=["v_2"])
            for j in [r"|v_2| = \sqrt{4^2 + 3^2}", r"|v_2| = \sqrt{25}"]
        )
        t2_norm.arrange(ORIGIN, aligned_edge=LEFT).move_to(4.5 * LEFT + 2 * DOWN)
        cmap = POS_CMAP
        cmap.update({"Euclidean": WHITE})
        eucl_norm = get_group_tex(
            r"\text{Euclidean Norm:}",
            r"v = v_x \hat{i} + v_y \hat{j} + v_z \hat{k}",
            r"|v| = \sqrt{v_x^2 + v_y^2 + v_z^2}",
            t2c=cmap,
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        eucl_norm.to_edge(RIGHT, 1).shift(2 * DOWN)
        self.play(Write(eucl_norm), run_time=2)
        self.wait()
        self.play(
            TransformMatchingTex(t1.copy(), t1_norm[0], path_arc=-PI / 2),
            TransformMatchingTex(t2.copy(), t2_norm[0], path_arc=-PI / 2),
        )
        # final form
        self.play(
            TransformMatchingStrings(
                t1_norm[0],
                t1_norm[1],
                key_map={"2^2 + 5^2": "29"},
                matched_keys=["\\sqrt"],
            ),
            TransformMatchingStrings(
                t2_norm[0],
                t2_norm[1],
                key_map={"4^2 + 3^2": "25"},
                matched_keys=["\\sqrt"],
            ),
        )
        self.wait()

        v1_v2 = Tex(r"|v_1| > |v_2|").next_to(t2_norm, DOWN, 0.5, aligned_edge=LEFT)
        self.play(Write(v1_v2))
        self.wait()

        g1 = axes.get_graph(lambda x: 2 * sin(1.5 * x) + 3 * cos(x / 2))
        g1.set_color(CYAN)
        g2 = axes.get_graph(lambda x: 3 * cos(1.5 * x) + 2 * sin(x / 2))
        g2.set_color(RED_B)

        graph_labels = VGroup(
            axes.get_graph_label(g1, "f(x)"),
            axes.get_graph_label(g2, "g(x)"),
        )
        graph_labels.arrange(RIGHT, buff=4).move_to(3 * UP)
        self.play(*map(FadeOut, [v1_v2, t1, t2, t2_norm, t1_norm, eucl_norm, vecs]))
        self.wait()
        self.play(ShowCreation(g1), ShowCreation(g2), Write(graph_labels))
        self.wait()

        self.play(
            *map(FadeOut, [g1, g2, axes]),
            graph_labels.animate.arrange(RIGHT, buff=2).set_y(0),
            Write(is_great.shift(0.05 * UP)),
        )

        norm_labels = VGroup(
            Tex(f"||{eq}(x)||_{{L^p}}", t2c={"f(x)": CYAN, "g(x)": RED_B}).move_to(
                graph_labels[i]
            )
            for i, eq in enumerate(["f", "g"])
        )
        # transforming
        lp_txt = TexText("$L^p$ Norm", font_size=78).to_edge(UP, buff=1)
        self.play(
            *[
                TransformMatchingStrings(i, j)
                for i, j in zip(graph_labels, norm_labels)
            ],
            Write(lp_txt),
        )
        self.play(FadeOut(norm_labels), FadeOut(is_great))
        self.wait()

        # start
        u_lp = get_tex(
            r"\tilde{u}(p, t) \equiv ||u(\cdot, t)||_{L^p} = \left(\int_{\Omega} |u(\mathbf{x}, t)|^p d\mathbf{x}\right)^ {1 / p}"
        ).shift(0.5 * UP)
        self.play(Write(u_lp[8:]))
        self.wait()
        self.play(Write(u_lp[:8]))
        self.wait()

        describe_dx = Tex(
            r"d\mathbf{x} = \begin{cases} dx, & \text{for } d = 1 \\ dx\cdot dy, & \text{for } d = 2 \\dx\cdot dy\cdot dz, & \text{for } d = 3 \end{cases} "
        )
        describe_dx.next_to(u_lp, DOWN, 0.5)
        self.play(Write(describe_dx))
        self.wait()
        self.play(FadeOut(describe_dx))
        self.wait()

        omega_txt = TexText("represents all space").next_to(u_lp[23:24], DOWN, 0.5)
        self.play(focus_on(u_lp, u_lp[23:24]), Write(omega_txt))
        self.wait()
        self.play(focus_on(u_lp, u_lp[22:35]), FadeOut(omega_txt))
        self.wait()
        self.play(focus_on(u_lp, u_lp[22:35], u_lp[10:16]))
        self.wait()
        self.play(focus_on(u_lp, u_lp[:7], u_lp[22:35], u_lp[10:16]))
        self.wait()

        u_norm = get_tex(
            r"|u(\mathbf{x}, t)| = \sqrt{u_x^2(\mathbf{x}, t) + u_y^2(\mathbf{x}, t) + u_z^2(\mathbf{x}, t)}"
        )
        u_norm.move_to(omega_txt).shift(0.25 * DOWN)
        self.play(focus_on(u_lp, u_lp[24:32]), Write(u_norm))
        self.wait()
        self.play(focus_on(u_lp, u_lp[-3:]), FadeOut(u_norm))
        self.wait()
        self.play(u_lp.animate.set_opacity(1))

        # p = 1
        ups = get_group_tex(
            r"\tilde{u}(1, t) = \int_{\Omega} |u(\mathbf{x}, t)| d\mathbf{x}",
            r"\tilde{u}(2, t) = \left(\int_{\Omega} |u(\mathbf{x}, t)|^2 d\mathbf{x}\right)^ {1 / 2}",
            isolate=[r"\tilde{u}(1, t)", r"\tilde{u}(2, t)"],
        ).arrange(ORIGIN, aligned_edge=LEFT)
        ups[1].shift(0.1 * UP)

        self.play(FadeOut(u_lp))
        self.play(Write(ups[0]))

        u_1t = r"\tilde{u}(1, t)"
        u_2t = r"\tilde{u}(2, t)"
        # p = 1 to 2
        self.play(
            TransformMatchingShapes(
                ups[0],
                ups[1],
                matched_pairs=[
                    (ups[0][u_1t], ups[1][u_2t]),
                    *[
                        (ups[0][i], ups[1][i])
                        for i in [
                            r"\int_{\Omega}",
                            r"|u(\mathbf{x}, t)|",
                            r"d\mathbf{x}",
                            "=",
                        ]
                    ],
                ],
            ),
        )
        self.wait()

        # highlight the features of p=2
        ke = get_group_tex(
            r"\int_{\Omega} u \cdot \left[\partial_t u + (u \cdot \nabla)u - \nu \nabla^2u + \nabla p \right]d\mathbf{x} = 0",
            r"\frac{1}{2} \frac{d}{dt}\int_{\Omega} |u(\mathbf{x}, t)|^2d\mathbf{x} = - \nu \int_{\Omega}|\nabla u(\mathbf{x}, t)| ^ 2 d\mathbf{x} < 0",
        )
        ke.arrange(DOWN, buff=0.5).next_to(ups, DOWN, -1)
        ke[1]["0"].set_color(YELLOW)
        ke_desc = Text("Kinetic Energy decreases with time.").next_to(ke, DOWN, 0.5)
        arr = Arrow(
            ke[0].get_left() + 0.15 * LEFT,
            ke[1].get_left() + 0.25 * LEFT,
            path_arc=2 * PI / 2,
        ).set_color(YELLOW)
        ke[0].add_background_rectangle(BLACK, buff=0.5, opacity=0.9)
        underline = Underline(ke_desc["decreases"], 0.15, CYAN, 6, 1)

        ke_brace = BraceText(ups[1][9:22], "Kinetic Energy", UP)
        self.play(focus_on(ups[1], ups[1][9:22]), ke_brace.creation_anim(), run_time=2)
        self.wait()
        self.play(Write(ke[1]))
        self.wait()
        self.play(Write(ke_desc))
        self.play(GrowFromEdge(underline, LEFT))
        self.wait()
        self.play(
            ShowCreation(ke[0]),
            Write(arr),
            *map(FadeOut, [ke_brace.brace, ke_brace.label]),
            run_time=2,
        )
        self.wait()
        self.play(
            *map(FadeOut, [underline, arr, ke_desc, ke]), ups[1].animate.set_opacity(1)
        )
        self.wait()

        ke_infty = Tex(
            r"\lim_{t \rightarrow \infty} \tilde{ u }(2, t) = \text{some finite value}",
            t2c={" u ": YELLOW_B},
        )
        ke_infty.shift(
            (ups[1][u_2t].get_center() - ke_infty[r"\tilde{ u }(2, t)"].get_center())
        )

        # transform
        self.play(
            Write(ke_infty[:6]), TransformMatchingShapes(ups[1][8:], ke_infty[14:])
        )
        self.wait(2)
        self.remove_all_except(self.frame)


class AnalysingEquation(Scene):
    def construct(self):
        lp_txt = TexText("$L^p$ Norm of Scaled Solution", font_size=78)
        self.play(Write(lp_txt))
        self.wait(2)
        self.play(FadeOut(lp_txt))

        lp_norm_scaled = get_group_tex(
            r"\tilde{u}_{\lambda}(p, t) \equiv ||u_{\lambda}(\cdot, t)||_{L_p} = \left(\int_{\Omega} |u_{\lambda}(\mathbf{x}, t)|^p d\mathbf{x}\right)^{1/p}",
            r"\tilde{u}_{\lambda}(p, t) = \left(\int_{\Omega} |u_{\lambda}(\mathbf{x}, t)|^p d\mathbf{x}\right)^{1/p}",
            r"\tilde{u}_{\lambda}(p, t) = \left(\int_{\Omega} \left|{1 \over \lambda}u(\mathbf{y}, s)\right|^p d\mathbf{x}\right)^{1/p}",
            r"\tilde{u}_{\lambda}(p, t) = \left(\int_{\Omega} \left|{1 \over \lambda}u(\mathbf{y}, s)\right|^p \lambda^d d\mathbf{y}\right)^{1/p}",
            r"\tilde{u}_{\lambda}(p, t) = \lambda^{d/p-1}\tilde{u}(p, s)",
        )
        lp_norm_scaled[1:].arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(Write(lp_norm_scaled[0][9:]))
        self.wait()
        self.play(Write(lp_norm_scaled[0][:9]))
        self.wait()
        # transform
        self.play(
            Transform(lp_norm_scaled[0][:8], lp_norm_scaled[1][:8]),
            Transform(lp_norm_scaled[0][22:23], lp_norm_scaled[1][8:9]),
            Transform(lp_norm_scaled[0][-19:], lp_norm_scaled[1][-19:]),
            FadeOut(lp_norm_scaled[0][8:-20]),
        )
        self.wait()
        # t2
        self.play(
            Transform(lp_norm_scaled[1][:12].copy(), lp_norm_scaled[2][:12]),
            Transform(lp_norm_scaled[1][12:22].copy(), lp_norm_scaled[2][12:30]),
            Transform(lp_norm_scaled[1][-6:].copy(), lp_norm_scaled[2][-6:]),
        )

        showing_d_dimen = get_group_tex(
            r"\mathbf{y} = \frac{\mathbf{x}}{\lambda}",
            r"\mathbf{x} = \lambda \mathbf{y}",
            r"d\mathbf{x} = \lambda^d d\mathbf{y}",
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        showing_d_dimen.next_to(lp_norm_scaled[2], DOWN, 0.5)
        self.play(Write(showing_d_dimen))
        self.wait(2)

        br = BraceText(showing_d_dimen, r"for $d$-dimensions\\$d = 3$ for $3D$", RIGHT)
        self.play(br.creation_anim())
        self.wait(2)
        self.play(FadeOut(br.label), FadeOut(br.brace), FadeOut(showing_d_dimen))
        self.wait(2)

        # t3
        self.play(
            Transform(lp_norm_scaled[2][:30].copy(), lp_norm_scaled[3][:30]),
            Write(lp_norm_scaled[3][30:32]),
            Transform(lp_norm_scaled[2][-6:].copy(), lp_norm_scaled[3][-6:]),
        )
        self.wait(2)
        self.play(Write(lp_norm_scaled[4]))
        self.wait(2)
        self.clear()
        self.add(lp_norm_scaled[1:])
        self.play(
            lp_norm_scaled[1:4].animate.set_opacity(0),
            lp_norm_scaled[4].animate.center(),
        )
        self.wait()
        self.play(FlashAround(lp_norm_scaled[4]["d/p-1"]), run_time=2)
        self.wait()

        possibilities = get_group_tex(
            r"\frac{d}{p} - 1 > 0 \Rightarrow p < d",
            r"\frac{d}{p} - 1 < 0 \Rightarrow p > d",
            r"\frac{d}{p} - 1 = 0 \Rightarrow p = d",
            t2c={},
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        possibilities.set_width(2.8).move_to(5 * LEFT + 1.8 * DOWN)

        sim_tex = get_tex(r"\tilde{u}_{\lambda} = \lambda^{d/p-1}\tilde{u}")
        sim_tex.set_width(2.5).to_edge(RIGHT).shift(DOWN)
        self.play(TransformMatchingStrings(lp_norm_scaled[-1], sim_tex))
        self.wait()

        axes = Axes((0, 5, 1), (0, 5, 1), height=6, width=11)
        axes.add_coordinate_labels()
        labels = axes.get_axis_labels(
            "\\lambda", r"\frac{\tilde{u}_\lambda}{\tilde{u}}"
        )
        axes.add(labels)

        graphs_2d = VGroup(
            axes.get_graph(lambda x: self.get_graph_fn(x, p, 2), x_range=(0.01, 5, 0.1))
            for p in range(1, 4)
        ).set_color_by_gradient(RED_B, CYAN, YELLOW_B)

        graphs_3d = VGroup(
            axes.get_graph(lambda x: self.get_graph_fn(x, p, 3), x_range=(0.01, 5, 0.1))
            for p in range(1, 5)
        ).set_color_by_gradient(RED_B, VIOLET, CYAN, YELLOW_B)
        graphs_3d[0].set_points_smoothly(
            list(filter(lambda d: d[1] < axes.get_top()[1], graphs_3d[0].get_points()))
        )

        legend_2d = self.get_legend(graphs_2d, 2).shift(1.5 * DOWN)
        legend_3d = self.get_legend(graphs_3d, 3)

        all_2d = VGroup(axes.copy(), graphs_2d, legend_2d)
        all_3d = VGroup(axes.copy(), graphs_3d, legend_3d)

        all_info = VGroup(all_2d, all_3d)
        all_info.arrange(RIGHT, buff=0.35).set_width(FW / 1.1).to_edge(UP, buff=0.5)

        self.play(*map(Write, [*all_info]), run_time=5)
        self.wait()
        self.play(Write(possibilities))
        self.wait(2)

        cmap = {r"\downarrow": RED, r"\uparrow": TEAL, "=": CYAN, "u": YELLOW_B}
        p1 = Tex(r"\tilde{u}_{\lambda} \ \uparrow \ , \ \lambda \ \uparrow", t2c=cmap)
        p1.next_to(possibilities[0], RIGHT, 0.5)
        p2 = Tex(r"\tilde{u}_{\lambda} \ \downarrow \ , \ \lambda \ \uparrow", t2c=cmap)
        p2.next_to(possibilities[1], RIGHT, 0.5)
        p3 = Tex(r"\tilde{u}_{\lambda} \ = \ , \ \lambda \ \uparrow", t2c=cmap)
        p3.next_to(possibilities[2], RIGHT, 0.5)

        # first possibility
        self.play(
            focus_on(possibilities, possibilities[0]),
            focus_on(graphs_2d, graphs_2d[0], only_stroke=True),
            focus_on(graphs_3d, graphs_3d[:2], only_stroke=True),
        )
        self.wait()
        self.play(Write(p1))
        self.wait()

        super_txt = TexText(r"super-critical \ding{55}", t2c={r"\ding{55}": RED})
        super_txt.set_width(2.5).next_to(p1, RIGHT, buff=0.35)
        self.play(Write(super_txt[:-1]))
        self.wait(2)
        self.play(focus_on(all_info, all_2d[0], graphs_2d[0], only_stroke=True))
        self.play(focus_on(all_info, all_3d[0], graphs_3d[:2], only_stroke=True))
        self.wait()
        self.play(
            focus_on(
                all_info,
                all_2d[0::2],
                all_3d[0::2],
                graphs_2d[0],
                graphs_3d[:2],
                only_stroke=True,
            )
        )
        self.wait()
        self.play(Write(super_txt[-1]))
        self.wait(2)

        # second possibility
        self.play(
            focus_on(possibilities, possibilities[1]),
            focus_on(graphs_2d, graphs_2d[-1], only_stroke=True),
            focus_on(graphs_3d, graphs_3d[-1], only_stroke=True),
            super_txt.animate.set_opacity(0.2),
            p1.animate.set_opacity(0.2),
        )
        self.wait()
        self.play(Write(p2))
        self.wait()

        sub_txt = Text(r"sub-critical")
        sub_txt.set_width(2).next_to(p2, RIGHT, buff=0.35)
        self.play(Write(sub_txt))
        self.wait()

        # third possibility
        self.play(
            focus_on(possibilities, possibilities[2]),
            focus_on(graphs_2d, graphs_2d[1], only_stroke=True),
            focus_on(graphs_3d, graphs_3d[2], only_stroke=True),
            sub_txt.animate.set_opacity(0.2),
            p2.animate.set_opacity(0.2),
        )
        self.wait()
        self.play(Write(p3))
        self.wait()

        critic_txt = TexText(r"critical \ding{51}", t2c={r"\ding{51}": TEAL})
        critic_txt.set_width(1.8).next_to(p3, RIGHT, 0.35)
        self.play(Write(critic_txt[:-1]))
        self.wait()
        self.play(Write(critic_txt[-1]))
        self.wait()

        r3 = RoundedRectangle(12.5, 7).set_fill(BLACK, 0.9).set_stroke(WHITE, 4, 1)
        title = TitleText("Olga Ladyzhenskaya's Inequality").to_edge(UP, buff=1.5)
        title.underline.set_color(CYAN)
        info_p3 = get_group_tex(
            r"\text{In 2D}: \tilde{u}_{\lambda}(2, t)=\tilde{u}(2, s) \rightarrow \text{critical} \ (p = d = 2)",
            r"||u||_{L^4} \le C\sqrt{||u||_{L^2} ||\nabla u||_{L^2}}",
            r"\Rightarrow \tilde{u}(4, t) \le C\sqrt{\tilde{u}(2, t) \widetilde{\nabla u}(2, t)}",
        )
        info_p3[0]["critical"].set_color(TEAL)
        info_p3.arrange(DOWN, buff=0.75).next_to(title, DOWN, 0.75)
        info_p3[-1].align_to(info_p3[1], RIGHT)

        self.play(Write(r3))
        self.play(Write(info_p3[0]))
        self.wait(2)
        self.play(Write(title), Write(info_p3[1:]), run_time=4)
        self.wait()

        info_3d = get_tex(
            r"\text{In 3D}: \tilde{u}_{\lambda}(3, t)=\tilde{u}(3, s) \rightarrow \text{critical} \ (p = d = 3)"
        )
        info_3d["critical"].set_color(TEAL)
        info_3d.move_to(info_p3[0])
        self.play(TransformMatchingStrings(info_p3[0], info_3d))
        self.wait()
        info_p3[0].set_opacity(0)
        self.play(*map(FadeOut, [info_p3, title]))

        self.play(WiggleOutThenIn(info_3d[r"\tilde{u}(3, s)"], 1.15))
        self.wait(2)
        self.remove_all_except(self.frame)

    @staticmethod
    def get_graph_fn(x: float, p: float, d: float) -> float:
        return x ** (d / p - 1)

    @staticmethod
    def get_legend(graphs: VGroup, d: int = 2) -> VGroup:
        n = len(graphs)
        p_vals = VGroup(Tex(f"p: {i+1}") for i in range(n))
        p_vals.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        lines = VGroup(
            Line(stroke_color=gr.get_color(), stroke_width=8)
            .set_length(0.6)
            .next_to(p_vals[i], RIGHT, buff=0.35)
            for i, gr in enumerate(graphs)
        )

        d_txt = Tex(f"d: {d}").next_to(VGroup(p_vals, lines), UP, buff=0.5)
        info = VGroup(d_txt, p_vals, lines)
        rect = RoundedRectangle(corner_radius=0.1).surround(info, buff=0.25)

        line_sep = DashedLine().set_length(rect.get_width())
        line_sep.next_to(d_txt, DOWN, buff=0.25)
        rect.set_stroke(TEAL, width=6).set_fill(BLACK, 1).add(line_sep)
        info.add_to_back(rect).to_corner(UR, buff=0.5)
        return info
