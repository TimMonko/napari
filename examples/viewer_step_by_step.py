"""
Build up a viewer step by step
==============================

Use one viewer across several steps, as you would in a Jupyter notebook:
open an image, segment it, add the segmentation on top, then look at the
result in 3D.

In the gallery, each step that changes the viewer is followed by a
screenshot of it, while steps that only compute something are not.

.. tags:: visualization-basic, layers
"""

# %%
# Open the image
# --------------
# Start with the nuclei channel of the ``cells3d`` dataset, a 3D stack of
# fluorescence images. The viewer shows one 2D slice at a time.

from scipy import ndimage as ndi
from skimage import data, filters

import napari

nuclei = data.cells3d()[:, 1]

viewer = napari.Viewer()
image_layer = viewer.add_image(nuclei, name='nuclei', colormap='magma')

# %%
# Segment the nuclei
# ------------------
# Smooth the image, threshold it and label each connected region. This step
# does not touch the viewer, so it has no screenshot.

smoothed = filters.gaussian(nuclei, sigma=2)
binary = smoothed > filters.threshold_otsu(smoothed)
labels, n_nuclei = ndi.label(binary)
print(f'Found {n_nuclei} nuclei')

# %%
# Add the segmentation
# --------------------
# Add the labels on top of the image in the same viewer.

labels_layer = viewer.add_labels(labels, name='segmentation', opacity=0.5)

# %%
# View in 3D
# ----------
# Switch the same viewer to 3D and turn the camera to see the volume.
# sphinx_gallery_thumbnail_number = 3

viewer.dims.ndisplay = 3
viewer.camera.angles = (10, -30, 60)

if __name__ == '__main__':
    napari.run()
