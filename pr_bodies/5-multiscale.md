# References and relevant issues

# Description
The half-voxel offset between levels of a 3D multiscale image used `layer.scale`, so with an affine scale of 0.01 a level 1 image was offset by 0.5 world units instead of 0.005, most of the way across a 0.64-wide volume. The offset now goes through the full data to world transform, which also covers rotation and conversion to viewer units.

**Change to expected behavior:** 3D multiscale images with a scale in `affine` move to the correct position, and so line up with the same data in 2D or with other layers. Images using only `scale` are unchanged.
