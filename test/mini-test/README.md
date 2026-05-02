# Mini Test

This folder is the isolated mini experiment area.

Use it when you want to validate preprocessing, training, and inference logic without mixing outputs into the main `Result/` tree.

Entry points:

- `bash test/mini-test/train_minimal.sh all`
- `bash test/mini-test/inference_minimal.sh ldm`
- `bash test/mini-test/run_minimal_experiment.sh`
- `bash test/mini-test/diagnose_minimal.sh`

Default behavior:

- mini checkpoints live under `test/mini-test/train_results_mini/`
- mini inference outputs live under `test/mini-test/inference_results_mini/`
- temporary linked mini dataset lives under `test/mini-test/.tmp_mini_train_dataset/`

Use the production launchers under `diffusion_consistency_radar/launch/` when you want formal training or formal evaluation outputs.
