# References and relevant issues
Follows up on the integer coordinates from #9287.

# Description
The status bar only looked at `layer.scale` and `layer.translate` to choose integer coordinates and precision. An image with an affine scale of 0.01 and default units showed world position `(0.123, 1.956)` as `[0, 2]`. It now shows `[0.1230, 1.9560]`, the same as `scale`.

**Change to expected behavior:** layers with default units and a non-identity `affine`, `rotate` or `shear` now show decimal coordinates instead of integers. Layers with no transform still show integers.
