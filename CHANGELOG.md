# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2021-07-28
### Added
- `TrainDataset(dataset)` with support for huggingface datasets.
- `augment_five_crop` function for use with `dataset.map(augment_five_crop, batched=True)`

## [0.1.2] - 2021-07-26
### Added
- Added metrics `EvalMetrics().evaluate(model, eval_dataset)` class to calculate PSNR and SSIM on a model and 
  evaluation dataset. Accepts `EvalDataset(dataset)` with huggingface datasets.

### Changed
- Replaced `EvalDataset(dataset)` to use huggingface datasets instead of H5 files.
- Fixed `EvalDataset(dataset)` to be robust to wrongly sized HR images (not equals to scaled LR image size).
