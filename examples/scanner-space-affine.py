"""
Scanner space and voxel space
=============================

This example takes apart the affine transform of an MRI volume with napari,
and views the volume side by side in grid mode: on its plain voxel grid, and
placed in the scanner's coordinates.

Medical images store an affine that maps voxel indices to scanner
coordinates, in millimetres. napari can take that matrix directly with
``affine=``, but it can also split it into the layer's ``scale``, ``rotate``
and ``translate``, which shows what each part does: the voxel size, the
orientation of the slices (tilted by about 9 degrees in this oblique
acquisition), and the position in the scanner.

The volume is a small test image shipped with nibabel.

.. tags:: visualization-advanced, layers
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.testing import data_path

import napari
from napari.utils.transforms import Affine

img = nib.load(Path(data_path) / 'example4d.nii.gz')
# the file is a short fMRI time series, keep the first volume
volume = np.asarray(img.dataobj[..., 0])

# the affine maps (i, j, k) voxel indices to RAS+ scanner coordinates:
# x increases to the right, y to the front and z to the top of the head
transform = Affine(affine_matrix=img.affine)
print(f'affine matrix:\n{img.affine}')
print(f'scale (voxel size): {transform.scale}')
print(f'translate (position of the first voxel): {transform.translate}')
# the rotation tilts the slices about x, and also flips x: the voxels are
# stored from right to left, but x increases to the right
print(f'rotate:\n{transform.rotate.round(3)}')

space_unit, _ = img.header.get_xyzt_units()  # 'mm'
layer_kwargs = {
    'axis_labels': ('x', 'y', 'z'),
    'units': (space_unit,) * 3,
    'blending': 'additive',
}

# only the voxel size: the volume sits at the origin, aligned with the axes
viewer = napari.Viewer()
voxel_layer = viewer.add_image(
    volume, scale=transform.scale, name='voxel space', **layer_kwargs
)
# the whole affine, from its parts. This is the same as affine=img.affine
viewer.add_image(
    volume,
    scale=transform.scale,
    rotate=transform.rotate,
    translate=transform.translate,
    name='scanner space',
    colormap='magma',
    **layer_kwargs,
)

viewer.scene.overlays.coordinate_grid.visible = True
viewer.canvas.overlays.axes.visible = True
viewer.canvas.overlays.scale_bar.visible = True

# the 3D camera looks along x by default, so the tilt (a rotation between
# y and z) faces it
viewer.dims.ndisplay = 3
# reorder the dimensions so that z is the first axis, y the second and x the third
viewer.dims.order = (2, 1, 0)
viewer.scene.camera.angles = (-22, 7, -44)
viewer.fit_to_view()

if __name__ == '__main__':
    napari.run()
