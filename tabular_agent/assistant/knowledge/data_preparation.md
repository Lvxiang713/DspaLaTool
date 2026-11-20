---
id: data_preparation
title: Data preparation
keywords: csv data column identifier leakage prepare upload encode category categorical numeric schema target
---

# Data preparation

Provide a UTF-8 CSV with one target column. Remove row identifiers and any
column that would not be available when predicting a new case, because they
cause leakage. The frontend detects columns and infers each feature as numeric
or categorical; you can override the suggestion per column or uncheck a column
to exclude it.

All encoding and statistics are fitted on training rows only. Categorical
features become training-fitted indicator groups; numeric features are
standardized with source-training mean and standard deviation. Constant
training columns are recorded and dropped. Duplicate or empty
column names and inconsistent row widths are rejected instead of being
silently renamed.

An optional external CSV is used only for evaluation after training. It never
fits preprocessing, never trains the model and never selects parameters.
Use a new output directory for every run so failures keep their artifacts.
