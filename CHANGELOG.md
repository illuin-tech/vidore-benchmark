# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [4.0.2] - 2024-10-17

### Deprecated

- Deprecate the `interpretability` module

### Build

- Fix and update conflicts for package dependencies

## [4.0.1] - 2024-10-07

- Rename ColPali model alias to match model name (use `--model-name vidore/colpali` instead of `--model-name vidore/colpali-v1.2` with the `vidore-benchmark evaluate-retriever` CLI)
- Use the ColPali model name to load ColPaliProcessor instead of the PaliGemma one

### Fixed

- Add missing `model.eval()` to all vision retrievers to make results deterministic

## [4.0.0] - 2024-09-21

### Added

- Add "Ruff" and "Test" CI pipelines
- Add upper bound for `colpali-engine` to prevent eventual breaking changes

### Changed

- Remove unused deps from `pyproject`
- Clean `pyproject`
- Bump `colpali-engine` to `v0.3.0` and adapt code for the new API
- Replace black with ruff linter
- Add better ColPali model loading

### Removed

- Remove duplicate code with `colpali-engine` (e.g. remove `ColPaliProcessor`, `ColPaliScorer`...)

### Fixed

- Change typing to support Python 3.9
- Fix the `generate_similarity_maps` CLI
- Various fixes

## [3.4.2] - 2024-09-09

- Fix typo when making `model_name` configurable in previous release
- Fix wrong image processing for ColPali

## [3.4.1] - 2024-09-09

### Changed

- Make `model_name` configurable in `ColPaliRetriever` as optional arg
- Tweak `EvalManager`
- Tweak `ColPaliProcessor`
- Improve tests

## [3.4.0] - 2024-08-29

### Changed

- Add the `colpali-engine` dependency
- Add `sentence-transformers` to compulsory dependencies
- Bump ColPali to v1.2 [[model card]](https://huggingface.co/vidore/colpali-v1.2)
- Remove the `baselines` dependency group

## [3.3.0] - 2024-08-22

### Added

- Add int8 quantization for embeddings (experimental)
- Add tests for int8 quantization
- Add dynamic versioning with `hatch-vcs`
- Add loggers in modules (with `loguru`)

### Changed

- Re-organize quantization modules

### Fixed

- Fix dtype conversion in `ColPaliScorer`
- Fix int8 overflow when computing ColBERT score w/ int8 embeddings
- Fix quantization tests

### Removed

- Remove unused `ColPaliWithPoolingRetriever`

## [3.2.0] - 2024-08-15

### Added

- Add experiments on token pooling
- Add `from_multiple_json` method in `EvalManager`

### Fixed

- Fix error when using token pooling with bfloat16 tensors
- Add missing L2 normalization in token pooling

## [3.1.0] - 2024-08-06

### Added

- Add support for token pooling in document embeddings

### Fixed

- Fix ruff settings

## [3.0.0] - 2024-07-30

### Added

- Add barplot for score per token in the interpretability script

### Changed

- [Breaking] All the CLI arguments now clearly show if the batch size is related to the query embedding computation, the doc embedding computation, or the scoring

## Fixed

- Fix `DummyRetriever`
- Fix a few broken retriever tests

## [2.2.1] - 2024-07-26

### Fixed

- Fix wrong dictionary key in the `evaluate_retriever` CLI script


## [2.2.0] - 2024-07-22

### Changed

- Currify vision retriever registry
- Rename `requirements.txt` to `requirements-dev.txt` to avoid confusion

### Fixes

- Fix wrong typing

## [2.1.0] - 2024-07-18

### Added

- Add support for BGE-M3 with ColBERT (multi-vector)

### Changed

- set default batch_score in `evaluate_retriever

### Fixed

- Add missing `torch.no_grad()`
- Fix wrong savepaths in CLI scripts
- Allow `image_util` paths for non-posix systems

## [2.0.0] - 2024-07-04

### Changed

- For all instances of VisionRetriever, the forward_queries and forward_documents methods now require a batch_size argument

## [1.0.1] - 2024-07-02

### Fixed

- Fix a bug for the retrieve_on_pdfs entrypoint script

## [1.0.0] - 2024-06-28

### Added

- Add evaluation scripts for the ViDoRe benchmark introduced in the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
- Add off-the-shelf retriever implementations from the ColPali paper: BGE-M3, BM25, Jina-CLIP, Nomic-Vision, SigLIP, and ColPali.
- Add code to generate similarity maps following the methodology introduced in the ColPali paper.
- Add entrypoint scripts: vidore-benchmark (evaluation scripts) and generate-similarity-maps (interpretability).

<!-- Links -->
[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html
