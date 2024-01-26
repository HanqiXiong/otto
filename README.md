# Otto Recommender System

## Introduction
This project aims to build and validate a recommender system using the Otto dataset. The process involves training a Word2Vec model, followed by model validation and prediction. The project is designed to run on Kaggle's GPU environment for optimal performance.

## Prerequisites
- Kaggle Account (for GPU access)
- At least 128GB of RAM for efficient processing
- Familiarity with Jupyter Notebooks

## Key Dependencies
- pandas
- numpy
- lightgbm
- cudf
- gensim

## Installation
Ensure that all the above libraries are installed in your Python environment. You can install these packages using pip:
```bash
pip install pandas numpy lightgbm cudf gensim
```

## Dataset Download and Preparation
The datasets required for this project are automatically downloaded and prepared when you run the initial setup notebook. Ensure that the following datasets are listed in your setup:

1. [Otto Chunk Data in Parquet Format](https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format)
2. [Otto Validation Dataset](https://www.kaggle.com/datasets/cdeotte/otto-validation)
3. [Otto Recommender System Competition Data](https://www.kaggle.com/competitions/otto-recommender-system/data)
4. [Otto Full Optimized Memory Footprint](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint)

These datasets will be automatically downloaded into an 'input' folder as part of the initial setup process.

## Running the Project
Follow these steps to run the project:

### Model Training
- **File**: `word2vec-train.ipynb`
- **Environment**: Recommended to run on Kaggle GPU for efficiency.

### Model Validation
Execute the following notebooks in order:
1. **Recall Program** (`code/recall_valid.ipynb`): Use Kaggle GPU for better performance.
2. **Feature Preparation** (`code/feature_prepare_valid.ipynb`)
3. **Ranking Model** (`code/rank_model_valid.ipynb`): Modify parameters `t` (can be `clicks`, `carts`, or `orders`). The default recall number is 50, which can be increased up to 250.

### Model Prediction
Execute the following notebooks in order:
1. **Recall Program** (`code/recall_test.ipynb`): It is recommended to use Kaggle GPU.
2. **Feature Preparation** (`code/feature_prepare_test.ipynb`)
3. **Ranking Model** (`code/rank_model_test.ipynb`): This will generate the submission file `submission.csv`.

## Additional Notes
- The total runtime for the code is approximately 48 hours.
- Monitor the GPU usage on Kaggle to ensure it is being utilized effectively.
- Adjust parameters and settings as needed based on your specific requirements.