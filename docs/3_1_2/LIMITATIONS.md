# Limitations

- Returns at most one instance.
- Does not support all-instance exact set output.
- Does not support constrained-single, spatial, ordinal, or relation queries.
- Micro/tiny parts are weak.
- Multi-instance queries are structurally limited.
- Mask quality is limited by bbox prompt quality.
- Runtime P95 is about 220 ms in the sealed clone environment.
- Not production quality.
- No Route A empty fallback.
