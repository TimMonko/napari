import contextlib
import gc
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

import numpy as np
from vispy.scene import VisualNode
from vispy.visuals.transforms import MatrixTransform

from napari._vispy.utils.gl import BLENDING_MODES, get_max_texture_sizes
from napari._vispy.utils.shared_resource_manager import (
    get_shared_resource_manager,
)
from napari.layers import Layer
from napari.utils.events import disconnect_events

_L = TypeVar('_L', bound=Layer)


class VispyBaseLayer(ABC, Generic[_L]):
    """Base object for individual layer views

    Meant to be subclassed.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer model.
    node : vispy.scene.VisualNode
        Central node with which to interact with the visual.

    Attributes
    ----------
    layer : napari.layers.Layer
        Layer model.
    node : vispy.scene.VisualNode
        Central node with which to interact with the visual.
    scale : sequence of float
        Scale factors for the layer visual in the scenecanvas.
    translate : sequence of float
        Translation values for the layer visual in the scenecanvas.
    MAX_TEXTURE_SIZE_2D : int
        Max texture size allowed by the vispy canvas during 2D rendering.
    MAX_TEXTURE_SIZE_3D : int
        Max texture size allowed by the vispy canvas during 2D rendering.


    Notes
    -----
    _master_transform : vispy.visuals.transforms.MatrixTransform
        Transform positioning the layer visual inside the scenecanvas.
    """

    layer: _L

    def __init__(self, layer: _L, node: VisualNode) -> None:
        super().__init__()
        self.events = None  # Some derived classes have events.

        self.layer = layer
        self._array_like = False
        self.node = node
        self.first_visible = False

        (
            self.MAX_TEXTURE_SIZE_2D,
            self.MAX_TEXTURE_SIZE_3D,
        ) = get_max_texture_sizes()

        # Register with shared resource manager
        self._resource_manager = get_shared_resource_manager()
        layer_type = self._get_layer_type()
        self._resource_manager.register_layer(layer_type, self)

        self.layer.events.refresh.connect(self._on_refresh_change)
        self.layer.events.set_data.connect(self._on_data_change)
        self.layer.events.visible.connect(self._on_visible_change)
        self.layer.events.opacity.connect(self._on_opacity_change)
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.scale.connect(self._on_matrix_change)
        self.layer.events.translate.connect(self._on_matrix_change)
        self.layer.events.rotate.connect(self._on_matrix_change)
        self.layer.events.shear.connect(self._on_matrix_change)
        self.layer.events.affine.connect(self._on_matrix_change)
        self.layer.experimental_clipping_planes.events.connect(
            self._on_experimental_clipping_planes_change
        )

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.MatrixTransform:
        Central node's firstmost transform.
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self.node.transform, MatrixTransform):
            self.node.transform = MatrixTransform()

        return self.node.transform

    @property
    def translate(self):
        """sequence of float: Translation values."""
        return self._master_transform.matrix[-1, :]

    @property
    def scale(self):
        """sequence of float: Scale factors."""
        matrix = self._master_transform.matrix[:-1, :-1]
        _, upper_tri = np.linalg.qr(matrix)
        return np.diag(upper_tri).copy()

    @property
    def order(self):
        """int: Order in which the visual is drawn in the scenegraph.

        Lower values are closer to the viewer.
        """
        return self.node.order

    @order.setter
    def order(self, order):
        self.node.order = order
        self._on_blending_change()

    @abstractmethod
    def _on_data_change(self):
        raise NotImplementedError

    def _on_refresh_change(self):
        self.node.update()

    def _on_visible_change(self):
        self.node.visible = self.layer.visible

    def _on_opacity_change(self):
        self.node.opacity = self.layer.opacity

    def _on_blending_change(self, event=None):
        blending = self.layer.blending
        blending_kwargs = cast(dict, BLENDING_MODES[blending]).copy()

        if self.first_visible:
            # if the first layer, then we should blend differently
            # the goal is to prevent pathological blending with canvas
            # for minimum, use the src color, ignore alpha & canvas
            if blending == 'minimum':
                src_color_blending = 'one'
                dst_color_blending = 'zero'
            # for additive, use the src alpha and blend to black
            elif blending == 'additive':
                src_color_blending = 'src_alpha'
                dst_color_blending = 'zero'
            # for all others, use translucent blending
            else:
                src_color_blending = 'src_alpha'
                dst_color_blending = 'one_minus_src_alpha'
            blending_kwargs = {
                'depth_test': blending_kwargs['depth_test'],
                'cull_face': False,
                'blend': True,
                'blend_func': (
                    src_color_blending,
                    dst_color_blending,
                    'one',
                    'one',
                ),
                'blend_equation': 'func_add',
            }

        self.node.set_gl_state(**blending_kwargs)
        self.node.update()

    def _on_matrix_change(self):
        dims_displayed = self.layer._slice_input.displayed
        # mypy: self.layer._transforms.simplified cannot be None
        transform = self.layer._transforms.simplified.set_slice(dims_displayed)
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translate = transform.translate[::-1]
        matrix = transform.linear_matrix[::-1, ::-1].T

        # The following accounts for the offset between samples at different
        # resolutions of 3D multi-scale array-like layers (e.g. images).
        # The 2D case is handled differently because that has more complex support
        # (multiple levels, partial field-of-view) that also currently interacts
        # with how pixels are centered (see further below).
        if (
            self._array_like
            and self.layer._slice_input.ndisplay == 3
            and self.layer.multiscale
            and hasattr(self.layer, 'downsample_factors')
        ):
            # The last downsample factor is used because we only ever show the
            # last/lowest multi-scale level for 3D.
            translate += (
                # displayed dimensions, order inverted to match VisPy, then
                # adjust by half a pixel per downscale level
                self.layer.downsample_factors[-1][dims_displayed][::-1] - 1
            ) / 2

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        affine_matrix[-1, : len(translate)] = translate

        child_offset = np.zeros(len(dims_displayed))

        if self._array_like and self.layer._slice_input.ndisplay == 2:
            # Perform pixel offset to shift origin from top left corner
            # of pixel to center of pixel.
            # Note this offset is only required for array like data in
            # 2D.
            offset_matrix = self.layer._data_to_world.set_slice(
                dims_displayed
            ).linear_matrix
            offset = -offset_matrix @ np.ones(offset_matrix.shape[1]) / 2
            # Convert NumPy axis ordering to VisPy axis ordering
            # and embed in full affine matrix
            affine_offset = np.eye(4)
            affine_offset[-1, : len(offset)] = offset[::-1]
            affine_matrix = affine_matrix @ affine_offset
            if self.layer.multiscale:
                # For performance reasons, when displaying multiscale images,
                # only the part of the data that is visible on the canvas is
                # sent as a texture to the GPU. This means that the texture
                # gets an additional transform, to position the texture
                # correctly offset from the origin of the full data. However,
                # child nodes, which include overlays such as bounding boxes,
                # should *not* receive this offset, so we undo it here:
                child_offset = (
                    np.ones(offset_matrix.shape[1]) / 2
                    - self.layer.corner_pixels[0][dims_displayed][::-1]
                )
            else:
                child_offset = np.full(offset_matrix.shape[1], 1 / 2)
        self._master_transform.matrix = affine_matrix

        child_matrix = np.eye(4)
        child_matrix[-1, : len(child_offset)] = child_offset
        for child in self.node.children:
            child.transform.matrix = child_matrix

    def _on_experimental_clipping_planes_change(self):
        if hasattr(self.node, 'clipping_planes') and hasattr(
            self.layer, 'experimental_clipping_planes'
        ):
            # invert axes because vispy uses xyz but napari zyx
            self.node.clipping_planes = (
                self.layer.experimental_clipping_planes.as_array()[..., ::-1]
            )

    def _on_camera_move(self, event=None):
        return

    def reset(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_matrix_change()
        self._on_experimental_clipping_planes_change()
        self._on_camera_move()

    def _on_poll(self, event=None):
        """Called when camera moves, before we are drawn.

        Optionally called for some period once the camera stops, so the
        visual can finish up what it was doing, such as loading data into
        VRAM or animating itself.
        """

    def close(self):
        """Vispy visual is closing."""
        try:
            # Unregister from shared resource manager first
            if hasattr(self, '_resource_manager'):
                layer_type = self._get_layer_type()
                self._resource_manager.unregister_layer(layer_type, self)

            # Ensure all pending GPU operations complete before cleanup
            if hasattr(self, 'node') and self.node is not None:
                # Properly cleanup vispy resources
                try:
                    # Force completion of any pending OpenGL operations for this node
                    if (
                        hasattr(self.node, 'canvas')
                        and self.node.canvas is not None
                    ):
                        canvas = self.node.canvas
                        if (
                            hasattr(canvas, 'context')
                            and canvas.context is not None
                        ):
                            with contextlib.suppress(OSError, AttributeError):
                                canvas.context.finish()
                                canvas.context.flush()

                    # Properly detach from scene graph first
                    if hasattr(self.node, 'parent'):
                        self.node.parent = None

                    # Clear any visual data that might hold OpenGL resources
                    if hasattr(self.node, 'set_data'):
                        with contextlib.suppress(AttributeError, OSError):
                            self.node.set_data(None)

                    # If this is a compound visual, clean up subvisuals
                    if hasattr(self.node, '_subvisuals'):
                        for subvisual in getattr(self.node, '_subvisuals', []):
                            if subvisual is not None:
                                with contextlib.suppress(
                                    AttributeError, OSError
                                ):
                                    if hasattr(subvisual, 'parent'):
                                        subvisual.parent = None
                                    if hasattr(subvisual, 'set_data'):
                                        subvisual.set_data(None)

                    # Reset transforms after detachment to avoid holding references
                    with contextlib.suppress(AttributeError, OSError):
                        if hasattr(self.node, 'transform'):
                            self.node.transform = MatrixTransform()

                except (OSError, AttributeError, RuntimeError):
                    # Continue cleanup even if some parts fail
                    pass

            # Disconnect events after visual cleanup
            disconnect_events(self.layer.events, self)

        except (OSError, AttributeError, RuntimeError):
            # Emergency cleanup - still try to disconnect events
            with contextlib.suppress(Exception):
                disconnect_events(self.layer.events, self)

        finally:
            # Force a small garbage collection to help release resources
            gc.collect()

    def _get_layer_type(self) -> str:
        """
        Get the layer type name for resource management.

        Returns
        -------
        str
            The layer type name
        """
        if hasattr(self.layer, '__class__'):
            class_name = self.layer.__class__.__name__.lower()
            # Extract the base layer type name
            if 'points' in class_name:
                return 'points'
            if 'labels' in class_name:
                return 'labels'
            if 'shapes' in class_name:
                return 'shapes'
            if 'image' in class_name:
                return 'image'
            if 'surface' in class_name:
                return 'surface'
            if 'tracks' in class_name:
                return 'tracks'
            if 'vectors' in class_name:
                return 'vectors'
        return 'unknown'
