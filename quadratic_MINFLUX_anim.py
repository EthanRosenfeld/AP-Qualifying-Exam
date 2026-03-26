from manim import *


class QuadraticMINFLUX(Scene):
    def construct(self):
        L = 4.0       # probe separation (axis units)
        x_mol = 0.4   # molecule position (slightly off MINFLUX centre)

        # ── Axes ──────────────────────────────────────────────────────────
        axes = Axes(
            x_range=[-L * 1.4, L * 1.4, L / 2],
            y_range=[0, 1.5, 0.5],
            x_length=11, y_length=5,
            axis_config={"include_tip": True, "color": BLACK,
                         "tick_size": 0.05, "stroke_color": BLACK},
        )
        axes.set_color(BLACK)
        x_lbl = axes.get_x_axis_label("x").set_color(BLACK).scale(1.4)
        y_lbl = axes.get_y_axis_label("I").set_color(BLACK).scale(1.4)

        # ── Molecule marker ───────────────────────────────────────────────
        mol_line = DashedLine(
            axes.c2p(x_mol, 0), axes.c2p(x_mol, 1.4),
            color=GOLD_D, dash_length=0.15,
        )
        mol_lbl = MathTex("x_0", color=GOLD_D, font_size=52).next_to(
            axes.c2p(x_mol, 0), DOWN, buff=0.3
        )

        # ── Beam-centre tracker ───────────────────────────────────────────
        bc = ValueTracker(x_mol)

        def I(x):
            """Quadratic intensity at x given current beam centre."""
            return ((x - bc.get_value()) / (L / 2)) ** 2

        # ── Quadratic curve ───────────────────────────────────────────────
        curve = always_redraw(lambda: axes.plot(
            lambda x: ((x - bc.get_value()) / (L / 2)) ** 2,
            x_range=[-L * 1.3, L * 1.3],
            color=BLUE, stroke_width=6,
        ))

        # Red dot at molecule position on the curve
        mol_dot = always_redraw(lambda: Dot(
            axes.c2p(x_mol, min(I(x_mol), 1.45)),
            color=RED, radius=0.13,
        ))

        # Blue triangle below x-axis marking beam centre
        beam_tri = always_redraw(lambda: (
            Triangle(fill_color=BLUE, fill_opacity=1, stroke_width=0)
            .scale(0.15)
            .next_to(axes.c2p(bc.get_value(), 0), DOWN, buff=0.05)
        ))

        # ── Build scene ───────────────────────────────────────────────────
        self.play(Create(axes), Write(x_lbl), Write(y_lbl), run_time=1)
        self.play(Create(mol_line), Write(mol_lbl))
        self.play(Create(curve), FadeIn(beam_tri))
        self.play(FadeIn(mol_dot))
        self.wait(0.5)

        # ── Move to x1 = +L/2 ────────────────────────────────────────────
        self.play(bc.animate.set_value(L / 2), run_time=1.5)
        self.wait(0.3)

        x1_lbl = MathTex(r"x_1 = +\tfrac{L}{2}", color=BLUE, font_size=42).next_to(
            axes.c2p(L / 2, 0), DOWN, buff=0.55
        )
        i1_ledger = MathTex(r"n_1 = A\!\left(x_0 - \tfrac{L}{2}\right)^2", color=RED, font_size=42).to_corner(UR).shift(DOWN * 0.6)

        self.play(Write(x1_lbl))
        self.play(Write(i1_ledger))
        self.wait(0.8)
        self.play(FadeOut(x1_lbl))

        # ── Move to x2 = -L/2 ────────────────────────────────────────────
        self.play(bc.animate.set_value(-L / 2), run_time=2.0)
        self.wait(0.3)

        x2_lbl = MathTex(r"x_2 = -\tfrac{L}{2}", color=BLUE, font_size=42).next_to(
            axes.c2p(-L / 2, 0), DOWN, buff=0.55
        )
        i2_ledger = MathTex(r"n_2 = A\!\left(x_0 + \tfrac{L}{2}\right)^2", color=GREEN_D, font_size=42).next_to(
            i1_ledger, DOWN, buff=0.25, aligned_edge=LEFT
        )

        self.play(Write(x2_lbl))
        self.play(Write(i2_ledger))
        self.wait(2)


if __name__ == "__main__":
    import os, glob, subprocess
    os.makedirs("figs", exist_ok=True)
    with tempconfig({
        "format": "mp4",
        "output_file": "quadratic_MINFLUX_anim",
        "media_dir": "figs",
        "pixel_height": 720,
        "pixel_width": 1280,
        "frame_rate": 15,
        "quality": "medium_quality",
        "background_color": WHITE,
    }):
        scene = QuadraticMINFLUX()
        scene.render()

    # Convert MP4 → GIF via ffmpeg two-pass palette for clean colours
    mp4_files = glob.glob(os.path.join("figs", "**", "quadratic_MINFLUX_anim.mp4"), recursive=True)
    if mp4_files:
        mp4     = mp4_files[0]
        gif     = os.path.join("figs", "quadratic_MINFLUX_anim.gif")
        palette = os.path.join("figs", "_palette.png")
        subprocess.run(["ffmpeg", "-y", "-i", mp4, "-vf", "palettegen=stats_mode=full", palette], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", mp4, "-i", palette,
                        "-filter_complex", "paletteuse=dither=none", gif], check=True)
        os.remove(palette)
