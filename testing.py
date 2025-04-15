import napari
import numpy as np

viewer = napari.Viewer()
# 4D, CZYX data, made with a pattern from left to right
data = np.linspace(0, 1, 100).reshape(1, 1, 1, -1)  # Gradient along the last axis
data = np.tile(data, (2, 10, 50, 1))  # Repeat to match the desired shape

# data  = np.random.random((2, 10, 50, 100))
layer = viewer.add_image(data, name='4D image', channel_axis=0, colormap=['magenta', 'green'])

# print(viewer.dims.order)
# viewer.dims.order = (1,2,0)
# print(viewer.dims.order)
# print(viewer.dims.displayed)


print(f'2D\n{viewer._sliced_extent_world_augmented}')
fig = viewer.export_figure()
print(f'2D figure\n{fig.shape}')


# print(viewer.dims.displayed)

viewer.dims.ndisplay = 3
# viewer.camera.angles = (45,45,45)

print(f'3D\n{viewer._sliced_extent_world_augmented}')

fig = viewer.export_figure()
print(f'3D figure\n{fig.shape}')

viewer.dims.order = (2,1,0)
print(f'3D with order (2,1,0)\n{viewer._sliced_extent_world_augmented}')
fig = viewer.export_figure()
print(f'3D figure with order (2,1,0)\n{fig.shape}')

# viewer.dims.order = (1,2,0)
# print(f'3D with order (1,2,0)\n{viewer._sliced_extent_world_augmented}')
# fig = viewer.export_figure()
# print(f'3D figure with order (1,2,0)\n{fig.shape}')

viewer.dims.ndisplay = 2
viewer.add_image(fig, name='exported figure', rgb=True)


# it looks, again, like grid display is messed up
# however, extent calculations seem to be ok,
# so Wouter's observation is likely specific to order and then flipping the canvas or something in the qt methods

if __name__ == '__main__':
    napari.run()

