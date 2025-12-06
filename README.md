# Multi-modal Machine Learning for Real Estate Price Prediction

ML model for predicting house prices using both tabular property features and property images. The project implements and compares three modeling approaches: Tabular-Only, Image-Only (CNN), and a Hybrid Multimodal Network. Detailed information about the models can be found in the Project Report (multimodal_housing_prediction.pdf)

## 🏠 Project Overview
This project predicts real estate prices in Southern California using the [SoCal dataset](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal). It demonstrates how combining visual data (house photos) with structured data (bedrooms, sqft, location) can improve predictive performance.

### Key Features
*   **Three Model Architectures**:
    *   **Tabular Model**: MLP for structured data.
    *   **Image Model**: ResNet18-based CNN for house images.
    *   **Hybrid Model**: Feature fusion network combining both data modalities.
*   **Advanced Visualization**: Tools to compare model performance, training history, and price range accuracy.
*   **Result Analysis**: Detailed metrics including R² and RMSE across different price bands.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/pdd23001/Multimodal-Housing-Price-Prediction.git
    cd Multimodal-Housing-Price-Prediction
    ```

2.  **Create a virtual environment**:
    
    **Option A: Using venv (Standard Python)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    **Option B: Using Conda (Anaconda/Miniconda required to be installed)**
    ```bash
    conda create -n house_price_pred python=3.13
    conda activate house_price_pred
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
## 📁 Project Structure

```text
.
├── main.py                  # Training entry point
├── visualize_results.py     # Plotting & comparison script
├── requirements.txt         # Dependencies
├── src/
│   ├── data_loader.py       # Dataset & DataLoader 
│   ├── models.py            # Model definitions
│   ├── preprocess.py        # Data processing logic
│   ├── train.py             # Training loop
│   └── utils.py             # Metrics & helpers
└── results/                 # Saved models/logs when you will run
└── visualizations/          # Saved plots when you will run
└── data/                    # Dataset
└── visualizations_parth/    # Saved plots from my side
└── results_parth/           # Saved models & logs from my side
```

## 📂 Dataset Setup

The project uses the **SoCal** dataset, which is already included in the `data/` folder. No manual download is required from your side.

Ensure the directory structure remains as follows:

```text
data/
├── socal2.csv            # Tabular data
└── socal2/
    └── socal_pics/       # Folder containing house images
        ├── 0.jpg
        ├── 1.jpg
        └── ...
```

## 🚀 Usage (Replication of Report Results)

**Note:** The results/plots used in the project report are preserved in the `results_parth/` and `visualizations_parth/` directories for reference. You can replicate these results by following the steps below.

### 1. Training
To replicate the study, train all three model variants. The results will be saved to a new `results/` directory.

**1. Train Multimodal Hybrid Model:**
```bash
python3 main.py --model_type hybrid --epochs 20 --batch_size 32 --lr 0.0001 --seed 42
```

**2. Train Unimodal Baseline Models:**
```bash
python3 main.py --model_type tabular --epochs 20 --batch_size 32 --lr 0.01 --seed 42
python3 main.py --model_type image --epochs 20 --batch_size 32 --lr 0.0001 --seed 42
```

### 2. Visualization & Comparison
After training all three models, you can generate the comparative plots (Loss, R², RMSE, Accuracy Bands).

**Visualize Single Run (run with the respective directory):**
Replace the timestamps below with the actual folder names generated in your `results/` directory:
```bash
python3 visualize_results.py --run_dir results/hybrid_YYYYMMDD_HHMMSS 
python3 visualize_results.py --run_dir results/tabular_YYYYMMDD_HHMMSS 
python3 visualize_results.py --run_dir results/image_YYYYMMDD_HHMMSS 
```
**Compare Models:**
Replace the timestamps below with the actual folder names generated in your `results/` directory. The plots will be saved in a new `visualizations/` directory:

Hybrid vs Tabular:
```bash
python3 visualize_results.py \
  --run_dir results/hybrid_YYYYMMDD_HHMMSS \
  --compare results/tabular_YYYYMMDD_HHMMSS 
```
All Three Models:
```bash
python3 visualize_results.py \
  --run_dir results/hybrid_YYYYMMDD_HHMMSS \
  --compare results/tabular_YYYYMMDD_HHMMSS \
   results/image_YYYYMMDD_HHMMSS
```

## Where to find results from report
**Metrics:**\n
The final metrics used in the project report (R², RMSE, Loss) can be found in the **`metrics.json`** files located within each model's subdirectory in **`results_parth/`** or within **`results/`** after you run the code yourself according to the instructions above. 

For example: `results/hybrid_YYYYMMDD_HHMMSS/metrics.json`

**Figure 4 from Report (Training Histories):**
The training history plots (Loss vs Epochs) shown in Figure 4 can be found in the **`visualizations_parth/`** directory:
*   **Fig 4a (Tabular)**: `visualizations_parth/tabular_history.png`
*   **Fig 4b (Image)**: `visualizations_parth/image_history.png`
*   **Fig 4c (Hybrid)**: `visualizations_parth/hybrid_history.png`

**If reproducing results:**
After you run the visualization component (from Usage), the new plots will be generated in your **`visualizations/`** folder:
*   **Fig 4a (Tabular)**: `visualizations/tabular_history.png`
*   **Fig 4b (Image)**: `visualizations/image_history.png`
*   **Fig 4c (Hybrid)**: `visualizations/hybrid_history.png`


**Figure 5 (Validation Loss Comparison):**
The validation loss comparison plot (Fig 5) can be found in:
*   **Report Version**: `visualizations_parth/hybrid_vs_tabular_vs_image_validation_loss.png`
*   **If reproducing results**: `visualizations/hybrid_vs_tabular_vs_image_validation_loss.png` 

**Figure 6 (Predicted vs. Actual Log-Prices):**
The prediction scatter plots shown in Figure 6 can be found in the **`visualizations_parth/`** directory:
*   **Fig 6a (Tabular)**: `visualizations_parth/tabular_predictions.png`
*   **Fig 6b (Image)**: `visualizations_parth/image_predictions.png`
*   **Fig 6c (Hybrid)**: `visualizations_parth/hybrid_predictions.png`

**If reproducing results:**
After you run the visualization component, the new plots will be generated in your **`visualizations/`** folder:
*   **Fig 6a (Tabular)**: `visualizations/tabular_predictions.png`
*   **Fig 6b (Image)**: `visualizations/image_predictions.png`
*   **Fig 6c (Hybrid)**: `visualizations/hybrid_predictions.png`

**Figure 7 (Accuracy Band Comparison):**
The accuracy band comparison plot (Fig 7) can be found in:
*   **Report Version**: `visualizations_parth/hybrid_vs_tabular_vs_image_accuracy_bands.png`
*   **If reproducing results**: `visualizations/hybrid_vs_tabular_vs_image_accuracy_bands.png` (after running Step 2 "Compare Models")

**Figure 8 (Residual Analysis):**
The residual analysis plots (Scatter plot & Histogram) shown in Figure 8 can be found in the **`visualizations_parth/`** directory:
*   **Fig 8a (Tabular)**: `visualizations_parth/tabular_residuals.png`
*   **Fig 8b (Image)**: `visualizations_parth/image_residuals.png`
*   **Fig 8c (Hybrid)**: `visualizations_parth/hybrid_residuals.png`

**If reproducing results:**
After you run the visualization component, the new plots will be generated in your **`visualizations/`** folder:
*   **Fig 8a (Tabular)**: `visualizations/tabular_residuals.png`
*   **Fig 8b (Image)**: `visualizations/image_residuals.png`
*   **Fig 8c (Hybrid)**: `visualizations/hybrid_residuals.png`

**Figure 9 (R² Comparison):**
The R² score comparison bar chart (Fig 9) can be found in:
*   **Report Version**: `visualizations_parth/hybrid_vs_tabular_vs_image_r2_comparison.png`
*   **If reproducing results**: `visualizations/hybrid_vs_tabular_vs_image_r2_comparison.png` (after running Step 2 "Compare Models")

**Figure 10 (RMSE Comparison):**
The RMSE (log) comparison bar chart (Fig 10) can be found in:
*   **Report Version**: `visualizations_parth/hybrid_vs_tabular_vs_image_rmse_comparison.png`
*   **If reproducing results**: `visualizations/hybrid_vs_tabular_vs_image_rmse_comparison.png` (after running Step 2 "Compare Models")

**Figure 11 (Per-bin R²):**
The price range R² comparison plot (Fig 11) can be found in:
*   **Report Version**: `visualizations_parth/hybrid_vs_tabular_price_range_r2.png`
*   **If reproducing results**: `visualizations/hybrid_vs_tabular_price_range_r2.png` 
    *   *Note*: To reproduce this exact plot (comparing only Hybrid vs Tabular), you must run the specific "Hybrid vs Tabular" comparison command listed in the **Usage** section.
