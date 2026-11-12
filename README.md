config.py: Windows paths, paper hyperparameters, running modes, structural statistics, command-line configurations.
data_processing.py: EXO fields, 06:00–24:00 timeframe, 15-minute aggregation, unified 7:1.5:1.5 chronological temporal split, max-value normalization, training profiling, and auditing.
preprocessing.py: Historical training DTW, same-time weighted KNN, training-period GMM, anomaly shrinkage, saving fitters/estimators.
graph_construction.py: Transfer-free directed route paths, latest vehicle speed prior to prediction, DBC, regional structural positive and negative neighbors.
model.py: Sparse physical EFT, GRU filter coefficients, differentiable ISTA-SER, semantic attention, dual-view, degree-aware mask, Chebyshev residuals, TCN-GLU, multi-step prediction.
losses.py: Station-level InfoNCE, correlation-based false-negative exclusion, regional prototype InfoNCE, gradient weighting formula.
sampling.py: Sliding windows without cross-night/cross-label splits, raw labels and masks, training-period Pearson matrix.
training.py: Training, validation, early stopping, saving weights, original-scale/unscaled metrics, precise logical batch gradient balancing, testing/prediction.
search.py: GA, DE, PSO, GWO, SA, random search, and budget-limited grid search; true validation fitness and evaluation counting.
experiments.py: 5-seed summaries, 8-component ablation study, sensor failure experiments, temperature/SER sensitivity analysis, efficiency measurements, optimizer comparisons.
main.py: Unified command-line entry point, supporting execution for a single region or ALL regions.
synthetic_data.py: Same-field synthetic input generator, strictly for testing purposes.
test_pipeline.py: DTW, frozen statistics, time boundaries, original metrics, region selection, search budgets, PyTorch forward/backward execution testing.
reporting.py: Alignment of true search trajectories by evaluation count/wall-clock time, summarizing cross-seed means and standard deviations.



We do not have permission to share the EXO data, but some similiar data will be available on request.

If you have any question, please send email to: xinyizhou0318@gmail.com
