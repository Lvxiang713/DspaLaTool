---
id: modeling_guide
title: Guided modeling
keywords: guide start begin steps workflow how to use model new
---

# Guided modeling workflow

1. Prepare a UTF-8 CSV with one target column; remove identifiers and leakage.
2. Upload it and confirm the detected feature types; uncheck anything to
   exclude. The target is excluded from predictors automatically.
3. Keep the default Basic settings unless you have a reason to change them.
4. Open Advanced settings only for epochs, landmarks, seed, holdout or device.
   Tether weight and learning rate are recommended to stay at 1 and 0.01.
5. Start modeling and analysis. Watch the Total Loss curve; Stop training is
   available and preserves partial artifacts.
6. When complete, preview the analysis report and equation predictor, then
   use the standalone predictor for new rows without retraining.
