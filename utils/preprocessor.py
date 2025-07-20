import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, cardinality_threshold=20, binary_as_categorical=True):
    """
    Preprocesses a tabular dataset for input into PATE-GAN.
    
    Returns:
        - processed_data (np.ndarray): The transformed data matrix
        - feature_names (List[str]): Column names after encoding
    """
    df = df.copy()
    
    # Column classification
    categorical_cols = []
    numerical_cols = []
    binary_cols = []
    datetime_cols = []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        num_unique = len(unique_vals)
        dtype = df[col].dtype

        if np.issubdtype(dtype, np.datetime64):
            datetime_cols.append(col)
        elif dtype == object or dtype.name == 'category':
            categorical_cols.append(col)
        elif np.issubdtype(dtype, np.number):
            if num_unique == 2:
                binary_cols.append(col)
                if binary_as_categorical:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            elif num_unique <= cardinality_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

    # Drop datetime columns (or optionally convert)
    df = df.drop(columns=datetime_cols)

    # Impute missing values
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    if numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Encode categoricals
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df[categorical_cols]) if categorical_cols else np.array([])

    # Normalize numericals
    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(df[numerical_cols]) if numerical_cols else np.array([])

    # Binary columns (if not included in cat/numeric)
    bin_data = df[binary_cols].values if binary_cols and not binary_as_categorical else np.array([])

    # Combine everything
    parts = [p for p in [cat_encoded, num_scaled, bin_data] if p.size > 0]
    processed_data = np.hstack(parts).astype(np.float32)

    # Get feature names
    feature_names = []
    if categorical_cols:
        feature_names += encoder.get_feature_names_out(categorical_cols).tolist()
    if numerical_cols:
        feature_names += numerical_cols
    if binary_cols and not binary_as_categorical:
        feature_names += binary_cols

    return processed_data, feature_names, {
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "binary_cols": binary_cols,
        "encoder": encoder if categorical_cols else None,
        "scaler": scaler if numerical_cols else None
    }

def reconstruct_data(synth_data, metadata):
    """
    Reconstructs original-format data from synthetic array using saved preprocessing metadata.

    Args:
        synth_data (np.ndarray): Generated synthetic data.
        metadata (dict): Preprocessing metadata returned by preprocess_data.

    Returns:
        pd.DataFrame: Human-readable reconstructed synthetic data.
    """
    cat_cols = metadata["categorical_cols"]
    num_cols = metadata["numerical_cols"]
    bin_cols = metadata["binary_cols"]
    encoder = metadata["encoder"]
    scaler = metadata["scaler"]

    result = pd.DataFrame()

    col_start = 0

    # Decode categorical
    if cat_cols:
        cat_width = len(encoder.get_feature_names_out(cat_cols))
        cat_data = synth_data[:, col_start:col_start + cat_width]
        cat_decoded = encoder.inverse_transform(cat_data)
        cat_df = pd.DataFrame(cat_decoded, columns=cat_cols)
        result = pd.concat([result, cat_df], axis=1)
        col_start += cat_width

    # Decode numerical
    if num_cols:
        num_width = len(num_cols)
        num_data = synth_data[:, col_start:col_start + num_width]
        num_denorm = scaler.inverse_transform(num_data)
        num_df = pd.DataFrame(num_denorm, columns=num_cols)
        result = pd.concat([result, num_df], axis=1)
        col_start += num_width

    # Binary columns
    if bin_cols and not metadata["categorical_cols"]:  # Treated separately
        bin_data = synth_data[:, col_start:col_start + len(bin_cols)]
        bin_data = np.round(bin_data).astype(int)
        bin_df = pd.DataFrame(bin_data, columns=bin_cols)
        result = pd.concat([result, bin_df], axis=1)

    return result
