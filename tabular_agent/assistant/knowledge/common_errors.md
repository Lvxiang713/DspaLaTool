---
id: common_errors
title: Common errors
keywords: error fail exception stopped unknown detached memory cuda csv invalid duplicate column class rows retry overwrite
---

# Common errors

CSV rejected: check duplicate or empty column names, inconsistent row widths,
invalid numeric text or infinity. The error names the offending column.

Too few rows per class: every class needs enough rows for the training and
holdout partitions. Check rare labels before starting.

CUDA or memory failure: switch Device to CPU or reduce batch size and
landmarks. Keep the failed run directory as evidence; corrected attempts use
a new run directory. Runs are never automatically retried or overwritten.

Unknown run status: after a UI restart, an unfinished detached run is marked
unknown. It is never restarted automatically; its final status appears if the
original process finishes.

Stop training: the Stop button cancels only the process owned by this server.
Logs and partial artifacts are preserved and the run is marked stopped rather
than complete.
