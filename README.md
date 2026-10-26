# anly735-lab03-Parm1486
## Reproducibility and provenance

The analysis uses a locally processed Civil Comments/Jigsaw dataset.
Its equivalence to the official WILDS release has not been
independently verified.

The custom development and evaluation populations, exclusions,
model settings, metrics, subgroup analyses, and diagnostics are
implemented in `analysis.py`.

The report reads completed results and calls `analysis.py` to
regenerate presentation figures. Rendering does not retrain the model.

### Run the analysis

From the repository folder:

```bash
python3 -u analysis.py \
  --data "/path/to/all_data_with_identities.csv" \
  --output "results" \
  --bootstrap 1000 \
  --seed 42
```

Replace the example data path with the location of your local CSV.

### Render the report

```bash
quarto render report.qmd --to docx
```

### Recorded metadata

`results/run_metadata.json` records:

- Successful completion status.
- Data and analysis-script SHA-256 checksums.
- Python and package versions.
- Random seed and bootstrap repetitions.
- Decision threshold and subgroup length boundary.

`results/model_metadata.json` records model settings and
training information.

### Consistency checks and limitations

The report checks the analysis-script checksum against the completed
run metadata. It does not reread and rehash the source CSV during
rendering.

If the source data or analysis code changes, rerun the analysis before
rendering the report.

An independent comparison with the official dataset release, a locked
software environment, and preprocessing-script checksums would
strengthen reproducibility.