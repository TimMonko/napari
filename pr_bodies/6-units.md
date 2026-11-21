# References and relevant issues
Depends on the Shapes, Points, Labels and status PRs: LINK THEM HERE. Follows #7889, which added unit-aware rendering.

# Description
Rendering and slicing convert from a layer's units to the units shared by the viewer, but `world_to_data` and `data_to_world` did not. With a layer in mm over an image in um, the layer was drawn at 1000 times the coordinates where it responded to the mouse: a Shapes vertex drawn at (500, 500) could not be grabbed, and a Points value there was `None`. This affected hit testing, the transform box and multiscale corner pixels.

This adds `Layer._data_to_viewer_scale`, as one way to relate data coordinates to the screen (layer scale, affine and units), and uses it in Shapes, Points, Labels and the status bar.

**Change to expected behavior, please check this API change:** when layers in a viewer use different units, the public `world_to_data` and `data_to_world` now use the viewer's units, so scripts get different numbers (`(0.2, 0.2)` becomes `(200, 200)` for mm in um). The result can therefore change when another layer is added or removed. Nothing changes if all layers share units, or for a layer outside a viewer. `Layer._data_to_world` stays in layer units, because `LayerList` extents rely on that.

Not tested: dragging the transform box (hover is tested) and 3D ray picking with mixed units.
