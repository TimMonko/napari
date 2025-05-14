"""
Calculate the bounding boxes of shapes
==========

Display shapes of various types in a shapes layer. Then, calculated the bounding
box of each shape and create a new layer. Then, check if the shape is a rotated rectangle
and print the result.

.. tags:: visualization-advanced, experimental
"""

import numpy as np
from skimage import data

import napari

# add the image
viewer = napari.view_image(data.camera(), name='photographer')

# create a list of polygons
polygons = [
    np.array([[11, 13], [111, 113], [22, 246]]),
    np.array(
        [
            [505, 60],
            [402, 71],
            [383, 42],
            [251, 95],
            [212, 59],
            [131, 137],
            [126, 187],
            [191, 204],
            [171, 248],
            [211, 260],
            [273, 243],
            [264, 225],
            [430, 173],
            [512, 160],
        ]
    ),
    np.array(
        [
            [310, 382],
            [229, 381],
            [209, 401],
            [221, 411],
            [258, 411],
            [300, 412],
            [306, 435],
            [268, 434],
            [265, 454],
            [298, 461],
            [307, 461],
            [307, 507],
            [349, 510],
            [352, 369],
            [330, 366],
        ]
    ),
]

# add polygons
layer = viewer.add_shapes(
    polygons,
    shape_type='polygon',
    edge_width=1,
    edge_color='coral',
    face_color='royalblue',
    name='shapes',
)

# change some attributes of the layer
layer.selected_data = set(range(layer.nshapes))
layer.current_edge_width = 5
layer.selected_data = set()

# add an ellipse to the layer
ellipse = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
layer.add(
    ellipse,
    shape_type='ellipse',
    edge_width=5,
    edge_color='coral',
    face_color='purple',
)

# create a rectangle
rectangle = np.array([[-100, 200], [300, 200], [300, 300], [-100, 300]])
# add the rectangle to the viewer
layer.add(
    rectangle,
    shape_type='rectangle',
    edge_width=5,
    edge_color='coral',
    face_color='purple',
)

# create a rotated rectangle
rotated_rectangle = np.array(
    [
        [-64.75069, 353.22742],
        [137.60655, 284.7292 ],
        [206.105  , 357.5849 ],
        [133.24908, 426.08325]
    ],
    dtype=np.float32
)
layer.add(
    rotated_rectangle,
    shape_type='rectangle',
    edge_width=5,
    edge_color='orange',
    face_color='green',
)


def inbounds_shape_shift(verts, image_shape):
    """
    Calculate the shift needed to move the entire shape within image bounds.

    Parameters
    ----------
    verts : (N, 2) np.ndarray
        Array of vertices (Y, X) for the shape.
    image_shape : tuple of int
        (height, width) of the image.

    Returns
    -------
    shift : np.ndarray, shape (2,)
        The [shift_y, shift_x] needed to bring the shape fully into bounds.
    """
    # Find the minimum and maximum Y and X coordinates of the shape
    min_y, min_x = verts.min(axis=0)
    max_y, max_x = verts.max(axis=0)

    # Initialize shift values for Y and X
    shift_y = 0
    shift_x = 0

    # If any part of the shape is above the top edge, shift down
    if min_y < 0:
        shift_y = -min_y
    # If any part of the shape is below the bottom edge, shift up
    elif max_y >= image_shape[0]:
        shift_y = image_shape[0] - 1 - max_y

    # If any part of the shape is left of the left edge, shift right
    if min_x < 0:
        shift_x = -min_x
    # If any part of the shape is right of the right edge, shift left
    elif max_x >= image_shape[1]:
        shift_x = image_shape[1] - 1 - max_x

    # Return the shift as a numpy array
    return np.array([shift_y, shift_x])

# get all the bounding boxes of the shapes
image_shape = viewer.layers['photographer'].data.shape
bboxes = []
for i, shape in enumerate(layer._data_view.shapes):
    bboxes.append(shape._bounding_box)

    if shape.name != 'rectangle':
        continue

    verts = shape.data
    bbox = shape._bounding_box
    # Get the 4 corners of the bounding box
    bbox_corners = np.array([
        [bbox[0,0], bbox[0,1]],
        [bbox[0,0], bbox[1,1]],
        [bbox[1,0], bbox[0,1]],
        [bbox[1,0], bbox[1,1]],
    ])
    # Check if each vertex matches (within tolerance) any bbox corner
    matches = [np.any(np.all(np.isclose(v, bbox_corners), axis=1)) for v in verts]

    if all(matches):
        print('axis-aligned rectangle')
    else:
        print('rotated rectangle')

    in_bounds = np.all(
        (verts[:, 0] >= 0) & (verts[:, 0] < image_shape[1]) &
        (verts[:, 1] >= 0) & (verts[:, 1] < image_shape[0])
    )

    if in_bounds:
        print('all vertices in bounds')
    if not in_bounds:
        print('some vertices out of bounds, sliding back into bounds')
        print('original vertices:', verts)
        shift = inbounds_shape_shift(verts, image_shape)
        layer._data_view.shift(i, shift)
        print('shifted vertices:', verts)





# add the bounding boxes to a new shapes layer
viewer.add_shapes(
    bboxes,
    shape_type='rectangle',
    edge_width=1,
    edge_color='yellow',
    face_color='transparent',
    name='bounding boxes',
)

if __name__ == '__main__':
    napari.run()
