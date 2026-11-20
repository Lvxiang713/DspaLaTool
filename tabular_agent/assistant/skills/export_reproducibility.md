---
id: export_reproducibility
title: Export and reproducibility
keywords: export reproduce reproducibility share predictor standalone html batch template hash provenance
---

# Export and reproducibility

Every run saves its configuration, encoder, equation, summary and checkpoint
with integrity hashes, so a colleague can verify that an exported predictor
matches its recorded training artifacts. predictor.html works standalone in a
browser: use Download CSV template, fill in new rows, upload the CSV and
download predictions. No server, Python or model checkpoint is needed at
prediction time.

To reproduce a run, keep the same CSV, feature types and settings, and record
the random seed. Deliberate changes belong in a new run so results stay
comparable.
