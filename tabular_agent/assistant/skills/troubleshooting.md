---
id: troubleshooting
title: Troubleshooting run failures
keywords: troubleshoot diagnose why failed error broken stopped unknown stuck
---

# Troubleshooting run failures

1. Read the error text in the interface or the run's process log first; it
   names the failing stage and, for data errors, the exact column.
2. Classify the cause: input schema, data content, memory/device, or an
   interrupted process.
3. Fix the smallest thing. Correct the CSV or reduce batch size, landmarks or
   device load. Do not change several settings at once.
4. Start a new run in a new output directory. Never overwrite or silently
   retry a failed run; the failed artifacts stay as evidence.
5. If the run is marked unknown after a UI restart, do not start a duplicate.
   Wait for the original process to record its final status.
