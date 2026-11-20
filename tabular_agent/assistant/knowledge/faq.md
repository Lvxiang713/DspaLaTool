---
id: faq
title: Frequently asked questions
keywords: faq question output predictor batch template standalone privacy local rerun retrain reuse
---

# Frequently asked questions

Where are results saved? Each run creates a new folder under the chosen output
directory containing report.html, predictor.html, summary JSON, coefficient
tables and the model checkpoint.

Can I reuse a trained model? Use "Reanalyze a saved model" to extract or
compare new expressions from a completed run without retraining. Frozen
artifact hashes are checked before analysis.

Does the exported predictor need this application? No. predictor.html is
standalone: single-row and batch CSV prediction run entirely in the browser.

Does anything leave my computer? Training, analysis and prediction are local.
The optional external LLM is off by default; when you enable it for one
question, only the question and a small summary of the current configuration
or run state are sent, never your CSV rows.
