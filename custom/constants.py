import numpy as np
from manimlib.constants import *

# Basic Shader Templates
VERT_TEMPLATE = """#version 330

in vec3 point;
out vec3 xyz_coords;

#INSERT emit_gl_Position.glsl

void main() {
    xyz_coords = point;
    emit_gl_Position(point);
}
"""

FRAG_TEMPLATE = """#version 330

in vec3 xyz_coords;
out vec4 frag_color;

void main() {
    frag_color = vec4(0.0, 0.0, 1.0, 0.5);
}
"""

# Constants
FW: float = FRAME_WIDTH
FH: float = FRAME_HEIGHT

# Colors
NAVY_BLUE = "#0066CC"
DARK_BLUE = "#2A9DF4"
VIOLET = "#EE82EE"
INDIGO = "#4B0082"
CYAN = "#00DDFF"
VIBGYOR = [VIOLET, INDIGO, BLUE, GREEN, YELLOW, ORANGE, RED]

# Numpy Shorthand
array = np.array
asarray = np.asarray
arange = np.arange
linspace = np.linspace

# Commonly used Math functions
sqrt = np.sqrt
exp = np.exp
log = np.log
log10 = np.log10
log2 = np.log2

# Numpy Random Distributions
rand = np.random.rand
randn = np.random.randn
uniform = np.random.uniform
randint = np.random.randint
shuffle = np.random.shuffle

# Trigonometric functions
sin = np.sin
cos = np.cos
tan = np.tan

# Inverse Trigonometric functions
asin = np.arcsin
acos = np.arccos
atan = np.arctan
atan2 = np.arctan2

# Hyperbolic Trigonometric functions
sinh = np.sinh
cosh = np.cosh
tanh = np.tanh

# Inverse Hyperbolic Trigonometric functions
asinh = np.arcsinh
acosh = np.arccosh
atanh = np.arctanh
