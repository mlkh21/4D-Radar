# Test Area

This directory is the isolated area for local experiments, smoke tests, and one-off debugging helpers.

Rules:

- Put temporary verification scripts here instead of the repo root.
- Keep production code under `diffusion_consistency_radar/` and `NTU4DRadLM_pre_processing/`.
- Put quick experiment launchers under `test/mini-test/`.
- Treat generated artifacts under `test/` as disposable unless you explicitly want to keep them.

Current layout:

- `fix_test.py`: alignment sanity check for raw radar/LiDAR frames
- `plot_raw_overlay.py`: quick BEV overlay image generator
- `test_interp.py`: sparse occupancy interpolation sanity check
- `mini-test/`: isolated minimal train/infer workflow
