# References and relevant issues
Related to #6649, please see the question below.

# Description
Points were drawn from `layer.scale[-1]` only, while the hit area used the full transform, so what you see and what you can click disagreed:

| Case | Drawn before | Hit area before | Both after |
|---|---|---|---|
| `size=100`, `scale=0.5`, zoom 2 | 100 px | 100 px | 100 px |
| `size=100`, affine scale of 0.5, zoom 2 | 200 px | 100 px | 100 px |
| `size=10`, `scale=(1, 1, 100)`, 100x axis hidden, zoom 2 | at least 600 px | 1198 px | 20 px |

**Question for reviewers:** in #6649 it was suggested that Points may intentionally not follow an affine, since they mark locations. But they do follow `scale`, and the `size` docstring says data pixels. I went with making `affine` match `scale`, so the drawn size and hit area agree. If ignoring the affine is preferred, the hit area needs the opposite fix.

**Change to expected behavior, this will be visible:**
- Points on a layer with a scaling affine change drawn size. A size-10 point with a 0.008 affine used to be 10 world units wide, and is now 10 data pixels. Anyone who tuned `size` to compensate will need to update it.
- With different scales per axis, the drawn size now follows the last *displayed* axis, so it changes when `dims.order` changes which axes are shown.
