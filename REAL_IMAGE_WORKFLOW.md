# Real-image improvement stage

The synthetic weak-class boost did not beat the original baseline overall. The next improvement stage should use **genuinely new real road images**, not repeated copies of existing training images.

## Keep this baseline safe

`runs/fog_road/baseline_best.pt`

## Final 9-class taxonomy

| ID | Class |
|---:|---|
| 0 | accident |
| 1 | bicycle |
| 2 | bus |
| 3 | car |
| 4 | debris |
| 5 | motorcycle |
| 6 | person |
| 7 | pothole |
| 8 | truck |

## Classes to prioritize

Highest priority:
- motorcycle
- debris
- person
- bicycle

Also useful:
- bus
- truck

Keep a mixture of clear-road and genuinely adverse-visibility scenes. Prefer diverse scenes, camera viewpoints, distances, object sizes and lighting. Avoid near-duplicate video frames.

## Workflow

1. Run `prepare_real_boost_workspace.py`.
2. Put new real images into `real_boost_input/images/`.
3. Put matching YOLO labels into `real_boost_input/labels/`.
4. Labels must already use the 0-8 taxonomy above.
5. Run `import_real_boost.py`.
6. Train with `train_real_boost.py` starting from `baseline_best.pt`.
7. Evaluate the candidate with `evaluate_clear_fog.py` on the same clear, combined fog, real fog and synthetic hazard test sets.
8. Replace the deployed model only if the candidate is actually better.

The importer checks exact-image duplicates against train/valid/test and never writes to validation or test folders.
