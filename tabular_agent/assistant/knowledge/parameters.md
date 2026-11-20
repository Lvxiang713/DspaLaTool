---
id: parameters
title: Training parameters
keywords: epochs landmarks tether weight seed holdout batch size learning rate device cpu cuda gpu memory parameter setting default
---

# Training parameters

Epochs: complete passes over the training data. Increase for longer training;
loss and epoch progress are always recorded.

Landmarks per class: number of learned reference landmarks per class.
Default 128. More landmarks increase capacity and computation time.

Tether weight: strength of the tether regularization relative to
classification loss. The recommended default in this version is 1.

Learning rate: initial optimizer step size. The recommended default in this
version is 0.01.

Random seed: controls reproducible splitting and initialization. Changing it
produces another reproducible run, not a model selection criterion.

Holdout fraction: share of uploaded rows reserved for evaluation and excluded
from training. It is observational and never selects the trained model.

Batch size: rows processed together per training batch. Gradients accumulate
across batches before each epoch update. Smaller batches reduce temporary
memory. Default 8.

Device: CPU always works. CUDA requires compatible hardware and a working
PyTorch CUDA installation. If GPU memory errors occur, choose CPU or reduce
batch size and landmarks, and record the change as a new run.
