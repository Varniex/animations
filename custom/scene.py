import os
import numpy as np
from manimlib.extract_scene import manim_config
from custom.objects import ShaderMobject
from manimlib.utils.file_ops import guarantee_existence
from custom.constants import WGSL_TEMPLATE, FH
from manimlib.scene.interactive_scene import InteractiveScene


class ShaderScene(InteractiveScene):
    """
    An integration of WebGPU Shading Language (WGSL) with ManimGL.

    The path to the `shaders` must be relative to the python script.
    |- some_folder
    |---- python_script.py
    |---- shaders
    |-------- shader_file.wgsl

    If shader_file is "" (default), then it'll be set to the SceneName.
    """

    shader_file: str = ""
    shader_class: ShaderMobject = ShaderMobject
    height: float = FH

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # to avoid using self.wait() repeatedly
        self.hold_on_wait = True

        # necessary files should exist
        self.assure_shader_file_exists()
        self.init_shader()

        # some built-in uniforms like iTime, iResolution, iMouse
        # similar to Shadertoy
        self.init_uniforms()

    def setup(self) -> None:
        super().setup()
        self.add(self.shader)

    def assure_shader_file_exists(self) -> None:
        self.shader_file = (self.shader_file or self.__class__.__name__) + ".wgsl"
        scene_folder_path = os.path.dirname(os.path.abspath(manim_config.run.file_name))
        self.shader_folder_path = guarantee_existence(os.path.join(scene_folder_path, "shaders"))
        self.shader_file_path = os.path.join(self.shader_folder_path, self.shader_file)

        if not os.path.exists(self.shader_file_path):
            print(f"Generating '{self.shader_file}' with default code template.")
            with open(self.shader_file_path, "w") as f:
                f.write(WGSL_TEMPLATE)

    def init_shader(self) -> None:
        self.shader = self.shader_class(shader_file=self.shader_file_path, height=self.height)

    def init_uniforms(self) -> None:
        self.set_uniforms(
            lambda: {
                "iTime": self.time,
                "iResolution": np.array(self.camera.get_pixel_shape(), dtype=np.int32),
                "iMouse": self.mouse_point.get_center()[:2],
            }
        )

    def set_uniforms(self, uniforms) -> None:
        if isinstance(uniforms, dict):
            self.shader.set_uniform(**uniforms)
        else:
            self.shader.f_always.set_uniform(uniforms)

    def set_uniform(self, **uniforms) -> None:
        self.set_uniforms(uniforms)

    def refresh(self) -> None:
        """
        In the embed mode, this can be called to refresh the code
        without any need to restart the Scene.
        """
        self.shader.refresh(self.camera)

    def refresh_and_hold(self) -> None:
        """
        Refresh the Scene in loop.
        To exit the loop, press <spacebar> when Window is focused.
        """
        self.refresh()
        self.hold_loop()
