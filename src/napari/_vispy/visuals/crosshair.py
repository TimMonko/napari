from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.gloo import IndexBuffer, VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

if TYPE_CHECKING:
    from vispy.visuals.visual import VisualView

_VERTEX_SHADER = """
// this shader has no vertices; instead, it uses indices to distinguish
// between which case we're in (i.e: which axis, which extremity, and which
// edge of the line). Each line is a quad made of 4 vertices: `axis` will be
// 0, 1 or 2, while `side` (along the line) and `across` (perpendicular to
// it) are either -1 or 1. `side` is used to output the segment extremities
// far past the screen so the lines look infinite, and `across` to give the
// line a width in screen pixels. The central gap is then discarded in the
// fragment shader.
// Lines are drawn as triangles because GL_LINES widths other than 1 are
// not supported on all platforms (e.g: macOS core profile, ANGLE).

attribute float a_idx;  // int attributes not working, so we use a float
attribute vec4 a_color;  // one color per axis, repeated for each vertex

varying vec2 v_center;
varying vec4 v_color;
varying float v_across;  // signed distance from the line center, in pixels

void main()
{
    v_color = a_color;

    int idx = int(a_idx);
    int axis = idx / 4;
    float side = (int(mod(float(idx), 2.0)) == 0) ? -1.0 : 1.0;
    float across = (int(mod(float(idx / 2), 2.0)) == 0) ? -1.0 : 1.0;

    vec3 direction =
        axis == 0 ? vec3(side, 0, 0) :
        axis == 1 ? vec3(0, side, 0) :
                    vec3(0, 0, side);

    vec4 center = $visual_to_render(vec4($center, 1));
    vec4 dir_clip = $visual_to_render(vec4(direction, 0));

    // direction of the axis on screen at the center, i.e. the derivative of
    // the perspective-divided position along the axis. Using only
    // dir_clip.xy would be wrong under perspective projection.
    vec2 axis_dir = dir_clip.xy * center.w - center.xy * dir_clip.w;

    if (length(axis_dir) < 1e-5 * abs(center.w))
    {
        // very small, so we're basically looking straight down this axis; we drop
        // it by putting it outside of the clip range (cannot discard in vertex)
        gl_Position = vec4(-2, -2, 0, 1);
        return;
    }

    // pass the center position in ndc coordinates to the fragment shader
    vec2 center_ndc = (center.xy / center.w);
    v_center = center_ndc;

    float extent = 5.0;  // should be enough to always go out of screen

    vec2 pos;
    if (dir_clip.w > 1e-6)
    {
        // under perspective, the end of the axis that moves away from the
        // camera does not go to infinity on screen: it converges to the
        // vanishing point, and nothing must be drawn past it
        pos = dir_clip.xy / dir_clip.w;
    }
    else
    {
        // position of this vertex (the extremity), far past the screen edge
        // in the direction of the axis, starting from the center
        pos = center_ndc + normalize(axis_dir) * extent;
    }

    // offset the vertex perpendicular to the line on screen. This must be
    // computed in pixels, because ndc are stretched by the canvas aspect.
    // The quad is padded by one physical pixel on each side so the edges
    // can be antialiased in the fragment shader.
    vec2 axis_px = normalize(axis_dir * $canvas_size);
    vec2 normal_px = vec2(-axis_px.y, axis_px.x);
    float half_width = $width / 2.0 + 1.0 / $pixel_ratio;
    v_across = across * half_width;
    // ndc span 2 units over the canvas size
    pos += normal_px * v_across * 2.0 / $canvas_size;

    gl_Position = vec4(pos, center.z / center.w, 1.0);
}
"""

_FRAGMENT_SHADER = """
varying vec2 v_center;
varying vec4 v_color;
varying float v_across;

void main() {
    // gl_FragCoord is in physical pixels; $canvas_size is logical pixels
    vec2 screen_pos = gl_FragCoord.xy / $pixel_ratio;
    vec2 center_px = (v_center * 0.5 + 0.5) * $canvas_size;
    float dist_from_center = length(screen_pos - center_px);

    // discard fragments inside the circular gap
    if (dist_from_center < $gap / 2.0) {
        discard;
    }

    // antialias the edges of the line over one physical pixel
    float edge = ($width / 2.0 - abs(v_across)) * $pixel_ratio + 0.5;
    float alpha = clamp(edge, 0.0, 1.0);
    if (alpha <= 0.0) {
        discard;
    }
    gl_FragColor = vec4(v_color.rgb, v_color.a * alpha);
}
"""


class CrosshairVisual(Visual):
    """Crosshair visual with a central gap.

    Displays an "infinite" 3D crosshair around a central point, with
    a circular gap around the center defined in screen pixels. Lines
    are drawn as screen-aligned quads, so their width (in screen pixels)
    is consistent across platforms.
    """

    def __init__(self) -> None:
        super().__init__(vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        # 4 vertices per axis, 2 triangles per axis
        self.shared_program['a_idx'] = VertexBuffer(
            np.arange(12, dtype=np.float32)
        )
        quad = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        self._index_buffer = IndexBuffer(
            np.concatenate([quad + 4 * axis for axis in range(3)])
        )
        self._draw_mode = 'triangles'
        self.position = np.array((0, 0, 0))
        self.color = np.array((1, 1, 1, 1))
        self.gap = 20
        self.width = 1

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = np.array(value, dtype=np.float32)
        self.shared_program.vert['center'] = self._position
        self.update()

    @property
    def color(self) -> np.ndarray:
        return self._color

    @color.setter
    def color(self, value: np.ndarray) -> None:
        """A single RGBA color, or one RGBA color per axis (x, y, z)."""
        self._color = np.array(value, dtype=np.float32)
        per_axis = np.broadcast_to(self._color, (3, 4))
        # 4 vertices per axis
        self.shared_program['a_color'] = VertexBuffer(
            np.repeat(per_axis, 4, axis=0).astype(np.float32)
        )
        self.update()

    @property
    def gap(self) -> float:
        return self._gap

    @gap.setter
    def gap(self, value: float) -> None:
        self._gap = float(value)
        self.shared_program.frag['gap'] = self._gap
        self.update()

    @property
    def width(self) -> float:
        """Width of the lines, in screen pixels."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = float(value)
        self.shared_program.vert['width'] = self._width
        self.shared_program.frag['width'] = self._width
        self.update()

    def _prepare_transforms(self, view: VisualView | None = None) -> None:
        if view is not None:
            view.view_program.vert['visual_to_render'] = (
                view.transforms.get_transform('visual', 'render')
            )

    def _prepare_draw(self, view: VisualView | None = None) -> None:
        """This method is called immediately before each draw.

        The *view* argument indicates which view is about to be drawn.
        """
        if view is not None:
            px_scale = self.transforms.pixel_scale
            for shader in (self.shared_program.vert, self.shared_program.frag):
                shader['canvas_size'] = view.canvas.size
                shader['pixel_ratio'] = px_scale


Crosshair = create_visual_node(CrosshairVisual)
