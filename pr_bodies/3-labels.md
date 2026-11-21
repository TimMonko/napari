# References and relevant issues

# Description
The brush used `layer.scale` in places where the cursor used the full transform, so they disagreed under an affine:

| Case | Before | After |
|---|---|---|
| Size-21 brush, affine scale `(1, 4)`, painted world area | 21 x 84 | 21 x 20, same as `scale` |
| Alt-drag 100 px, scale 0.01, 2 canvas px per data pixel, starting at size 10 | stays 10 | 60 |
| Brush cursor after doubling `scale` or `affine`, or changing displayed axes | not updated | updated |

**Change to expected behavior:**
- With an anisotropic affine, the same stroke now paints **different pixels**, matching the round cursor and what `scale` already did. Saved labels are unaffected until something is painted.
- Alt-drag resizing now follows the cursor, so it is faster than before on layers with a small scale, and slower on layers with a large scale.
