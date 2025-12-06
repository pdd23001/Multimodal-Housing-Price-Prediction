# src/preprocess.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def _make_price_bins(prices: np.ndarray) -> pd.Categorical:
    """
    Discretize prices into 3 bins (after filtering >2M):
      $0-500k, $500k-1M, $1M-2M

    Returns a pandas Categorical suitable for stratification.
    """
    return pd.cut(
        prices,
        bins=[0, 500_000, 1_000_000, 2_000_000],
        labels=["$0-500k", "$500k-1M", "$1M-2M"],
        include_lowest=True,
        right=True
    )


def load_and_preprocess_data(
    csv_path,
    test_size=0.2,
    random_state=42,
    image_folder="data/socal2/socal_pics",
    max_price=2_000_000,
    do_stratify=True,
    verbose=True
):
    """
    Loads the dataset, filters extreme outliers (> max_price), preprocesses tabular features,
    and performs an 80/20 stratified split (seed=42 by default).

    Returns:
        dict with:
            - 'train': (X_train_processed, y_train_log, train_df)
            - 'test' : (X_test_processed,  y_test_log,  test_df)
            - 'preprocessor'
            - 'feature_names'
            - 'log_transform' (True)
            - 'price_bins' (overall bin counts after filtering)
    """
    df = pd.read_csv(csv_path)

    # Construct image path column from image_id (if present)
    if "image_id" in df.columns:
        df["image_path"] = df["image_id"].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

    # Identify target column
    if "price" in df.columns:
        target_col = "price"
    elif "Price" in df.columns:
        target_col = "Price"
    else:
        # fallback
        target_col = df.columns[-1]
        if verbose:
            print("Warning: 'price'/'Price' not found. Using last column as target:", target_col)

    # Drop rows with missing target
    df = df.dropna(subset=[target_col]).copy()

    # Filter out extreme outliers (>= max_price)
    # You said remove 2M+ entirely -> keep prices <= 2M
    df = df[df[target_col] <= max_price].copy()

    # y on original scale and bins for stratification
    y_price = df[target_col].astype(float).values
    y_bins = _make_price_bins(y_price)

    # Drop any rows that didn't fall into a bin (should be none after filtering)
    valid_mask = ~pd.isna(y_bins)
    df = df.loc[valid_mask].copy()
    y_price = y_price[valid_mask]
    y_bins = y_bins[valid_mask]

    if verbose:
        overall_counts = pd.Series(y_bins).value_counts().reindex(["$0-500k", "$500k-1M", "$1M-2M"], fill_value=0)
        overall_pct = overall_counts / overall_counts.sum() * 100
        print("\nOverall price-bin distribution (after filtering <= $2M):")
        for k in overall_counts.index:
            print(f"{k:10s}  {overall_counts[k]:5d}  ({overall_pct[k]:5.2f}%)")

    # Features: drop non-predictive identifiers / text fields
    X = df.drop(columns=[target_col])

    exclude_cols = [
        "id", "image_path", "file_name", "url", "image_id",
        "street", "n_citi"  # keep 'citi' if present
    ]
    feature_cols = [c for c in X.columns if c not in exclude_cols]
    X_features = X[feature_cols]

    numeric_features = X_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_features.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )

    # Log-transform target for training
    y_log = np.log1p(y_price)

    # Stratified split on bins (recommended for your per-range evaluation)
    stratify_arg = y_bins if do_stratify else None

    X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
        X_features,
        y_log,
        y_bins,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    if verbose:
        def _print_split_counts(bins, label):
            c = pd.Series(bins).value_counts().reindex(["$0-500k", "$500k-1M", "$1M-2M"], fill_value=0)
            p = c / c.sum() * 100
            print(f"\n{label} price-bin counts:")
            for k in c.index:
                print(f"{k:10s}  {c[k]:5d}  ({p[k]:5.2f}%)")

        _print_split_counts(bins_train, "TRAIN")
        _print_split_counts(bins_test, "TEST")

    # Keep indices for mapping back to original df rows
    train_indices = X_train.index
    test_indices = X_test.index

    # Fit/transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = (
            numeric_features
            + list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features))
        )

    return {
        "train": (X_train_processed, np.asarray(y_train), df.loc[train_indices]),
        "test": (X_test_processed, np.asarray(y_test), df.loc[test_indices]),
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "log_transform": True,
    }
