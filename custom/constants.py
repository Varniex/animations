import numpy as np
from manimlib.constants import *

# Basic Shader Template
WGSL_TEMPLATE = """#INSERT mobject_uniforms.wgsl
#INSERT frame_uniforms.wgsl
#INSERT read_data.wgsl
#INSERT project_point.wgsl
#INSERT quad_corners.wgsl
#INSERT clip_test.wgsl

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) clip_distances: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    if (index >= VERTS_PER_QUAD) {
        out.position = vec4f(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    let point = read_vec3(quad_corner(index), DATA_OFFSET_point);
    let projection = project_point(point);
    out.position = projection.position;
    out.clip_distances = projection.clip_distances;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    clip_test(in.clip_distances);
    let uv: vec2f = in.position.xy / mob.iResolution;
    let color: vec3f = 0.5 + 0.5 * sin(mob.iTime + uv.xyx + vec3f(0.0, 2.0, 4.0));
    return vec4f(color, 1.0);
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
