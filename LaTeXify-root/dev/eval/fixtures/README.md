# Evaluation Fixtures

This directory contains reference LaTeX/PDF pairs used for regression testing
and run evaluation. Each fixture folder provides a `main.tex` reference file
alongside a rendered `reference.pdf`. The mapping between fixtures and the
pipeline's `build/main.tex` input is recorded in `manifest.json` via a SHA-256
fingerprint of the canonical TeX source.

To use a fixture as the pipeline input, copy its `main.tex` into `build/main.tex`
(or have your aggregation stage emit the same TeX). When the pipeline runs, the
resulting `build/main.tex` will hash to the value stored in the manifest, allowing
evaluation scripts to resolve the fixture automatically.
