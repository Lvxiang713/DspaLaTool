# DSpaLaRefiner tool

A local interface for tabular classification, model-derived equations and prediction. Upload a CSV, select the label and predictors, train the native DSpaLaRefiner classifier, and export analyses and a standalone HTML predictor.

## Quick start

1. **Install from GitHub Releases (recommended).** Download the Windows x64 setup EXE from [Releases](https://github.com/Lvxiang713/DspaLaTool/releases) and run it. Choose an installation folder and open **DspaLaTool** from the Start menu. The installer includes Python and the CPU dependencies. To try the tool locally, upload `examples/manufacturing.csv` from the installation folder and select `quality` as the target.
2. **Run from source.** Download and extract the repository, then double-click `Start_DSpaLaRefiner.cmd`. The first launch downloads any needed Python runtime and CPU dependencies, then opens the tool in your browser. Use `examples/manufacturing.csv` from the extracted folder to try the same workflow. If the browser does not open, go to `http://127.0.0.1:8765/`. Keep the launcher window open while using the tool.

## Paper predictor

Download the [merged-hospital predictor](predictors/Diabetes_Equation_Predictor_From_SWMU%26NSMC.html) and open the HTML file in a browser. It runs offline and supports individual and batch inputs. See the [prediction instructions](predictors/README.md).

## Developer install

Install with `python -m pip install .` and run `dspalarefiner-ui --runs ../private_runs`. For GPU training, install PyTorch for your CUDA environment before installing the tool. To reuse a downloaded CPU PyTorch wheel during source setup, run `Start_DSpaLaRefiner.cmd --torch-wheel path-to-wheel.whl`.

## Workflow

1. Upload a training CSV and choose its label column and predictor columns. Automatic numeric/categorical detection can be overridden.
2. Configure training and select geometric additive or Taylor orders 1–3. Training progress and mean Total Loss are recorded.
3. Review five classification metrics, original-model/expression agreement, feature importance and optional SHAP. Multiple expressions share one trained model.
4. Export the report and standalone HTML. The latter supports complete single inputs, a CSV template/upload, batch prediction and feature/interaction drilldown.

Saved models can be analyzed again without retraining. Reanalysis writes a new output directory and checks the parent files. Keep the original training CSV available privately for reanalysis.

## Synthetic example

`examples/manufacturing.csv` contains synthetic manufacturing data.

```sh
dspalarefiner examples/manufacturing.csv --target quality --output ../private_runs/demo --epochs 10 --landmarks 16 --device cpu
dspalarefiner-predict ../private_runs/demo new_rows.csv --output ../private_runs/predictions.csv
```

Open the report and predictor in `../private_runs/demo/`.

## Export features

Use a finished modeling run and a CSV with the same input columns.

```sh
dspalarefiner-features ../private_runs/demo input.csv --output features.csv --combine
```

Use `--combine` to include the original encoded features alongside the landmark responses. Omit it to export only the landmark responses.

## Optional assistant

The interface includes a local knowledge guide. To use an external LLM, select DeepSeek, Qwen or a custom compatible service and provide connection details in the UI.

## Data access and reproduction

Keep data and run outputs outside the repository, for example `../private_data` and `../private_runs`. To request clinical data, contact the paper’s corresponding author affiliated with the hospital that holds the data. Each hospital reviews requests under its clinical data access and use requirements. To reproduce the study, use the approved data, original cohort splits and experiment configurations. Prepare one row per sample, use consistent feature units and codes across hospitals, and fit preprocessing using the training cohort.

The [standalone predictor](predictors/Diabetes_Equation_Predictor_From_SWMU%26NSMC.html) runs the interpretable equation trained on data from both hospitals. Other hospitals can use it directly for external validation on their own approved data.

## Repository layout

| Path | Contents |
| --- | --- |
| `tabular_agent/` | Data loading, training, equation extraction, feature analysis, HTML export and the local UI. |
| `tabular_agent/vendor/native_model/` | Native model layers and losses. |
| `tabular_agent/assistant/` | Assistant routing, provider integration, local knowledge, prompts and offline browser assets. |
| `predictors/` | Standalone merged-hospital predictor and usage instructions. |
| `examples/` | Synthetic CSV for trying the tool. |
