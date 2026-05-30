# Repository Completeness Check

## Root Files
- [x] README.md
- [x] LICENSE
- [x] LICENSE_DATA.md
- [x] CITATION.cff
- [x] .zenodo.json
- [x] CHANGELOG.md
- [x] REPRODUCIBILITY.md
- [x] .gitignore
- [x] .gitattributes
- [x] requirements.txt
- [x] environment.yml

## Code
- [x] src/train_single_hand.py
- [x] src/predict.py
- [x] src/evaluate.py
- [x] src/export_onnx_single_hand.py
- [x] src/export_scaler_params.py
- [x] src/export_split_files.py
- [x] src/utils/
- [x] src/utils_original/gesture_experiment/

## Data Documentation
- [x] data/README.md
- [x] data/dataset_card.md
- [x] data/DATA_PROVENANCE.md
- [x] data/asl_dataset/
- [x] data/indian_sign_language/
- [x] data/nus_hand_posture/
- [x] data/asl_large_dataset/
- [ ] Raw datasets are not redistributed; users must download them from original sources.

## Models
- [x] models/README.md
- [x] models/model_card.md
- [x] full_model_seed42.pth or documented external link
- [x] ONNX model or documented external link
- [x] scaler_full_model_seed42.pkl or documented missing
- [x] label_encoder_full_model_seed42.pkl or documented missing
- [x] models/checksums.txt

## Results
- [x] results/tables/
- [x] results/raw_csv/
- [x] results/figures/
- [x] results/manifests/
- [x] ALL_DATASETS CSV files copied if available

## Deployment
- [x] deployment/jetson/README.md
- [x] deployment/jetson/jetson_environment.md
- [x] deployment/jetson/tensorrt_build_commands.md
- [x] TensorRT benchmark scripts or placeholders
- [x] tegrastats folders

## Demo
- [x] demo/demo_readme.md
- [x] demo/run_demo.py
- [x] demo/sample_images/
- [x] demo video or Zenodo link placeholder

## Tests / CI
- [x] Lightweight pytest smoke tests
- [x] GitHub Actions workflow

## GitHub / Release
- [x] Large files tracked by Git LFS or documented for Zenodo
- [x] Empty folders contain .gitkeep
- [x] Zenodo release tag planned: v1.0.0
- [ ] Replace placeholder author, GitHub URL, release date, and DOI in CITATION.cff/.zenodo.json before public release.
- [ ] Add CONTRIBUTING.md, CODE_OF_CONDUCT.md, and issue/PR templates if the repository will accept external contributions.
