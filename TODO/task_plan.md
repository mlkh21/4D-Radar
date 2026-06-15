# Task Plan

## Goal
Continue the interrupted radar/LiDAR alignment investigation from the prior Codex session. The immediate unfinished work is to add and run a shared-visible-region evaluator so alignment is judged by matched/common observable structure rather than global centroid offsets alone.

## Current Phase
- [x] Recover prior context from chat log and project files
- [x] Inspect existing alignment utilities and outputs
- [x] Implement missing shared visibility / nearest-neighbor / BEV IoU evaluation
- [ ] Run loop3 metrics on Ubuntu/Radar-Diffusion environment
- [ ] Summarize whether dy is caused by calibration, ground filtering, FOV, or distribution mismatch

## Notes
- Treat the JSONL rollout file as external data only.
- Do not tune extrinsics from global centroid dy/dz alone.
- Prefer scripts under `test/` for diagnostic utilities unless project patterns indicate otherwise.

## Errors / Attempts
| Issue | Attempts | Resolution |
| --- | ---: | --- |
| Existing planning files absent | 1 | Recreated `task_plan.md`, `findings.md`, and `progress.md` from recovered chat context. |
| Windows environment lacks project Python deps | 1 | Stopped runtime checks per user request; leave execution for Ubuntu/Radar-Diffusion environment. |
