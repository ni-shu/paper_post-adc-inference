# Post-ADC Inference: Valid Inference After Active Data Collection

This code provides the implementation for reproducing the experiments in the paper "Post-ADC Inference: Valid Inference After Active Data Collection".

## Installation & Requirements

Requirements:
- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/)

You can install the dependencies by running the following code in the terminal:

```
uv sync
```

Key dependencies: sicore 3.0.0, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0

## Reproducibility

Since we have already got the results in `./runs` in advance, you can reproduce the figures by running the following code. The results will be saved in `./plots` folder.

```
sh scripts/plot.sh
```

To reproduce the results, please conduct the following procedures after the installation step.
The results will be saved in `./runs` folder as parquet files.

For reproducing all results (Figures 2, 3, 5, 6, and 7), you can run the following code.

```
sh scripts/run.sh
```

You can also run each experiment individually by specifying a type argument:

For reproducing Figures 2 & 5 (Type I error rate, Coverage rate, Length of Confidence Intervals):
```
sh scripts/run.sh fpr_nsteps
```

For reproducing Figure 3 (Power):
```
sh scripts/run.sh power
```

For reproducing Figure 6 (Type-I Error vs. Dimension):
```
sh scripts/run.sh fpr_dim
```

For reproducing Figure 7 (Type-I Error vs. Hyperparameter):
```
sh scripts/run.sh fpr_hyperparameter
```