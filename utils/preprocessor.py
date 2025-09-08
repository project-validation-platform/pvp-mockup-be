import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def classify_columns(df, cardinality_threshold=20, binary_as_categorical=True):
    categorical_cols, numerical_cols, binary_cols, datetime_cols = [], [], [], []
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
    return categorical_cols, numerical_cols, binary_cols, datetime_cols

def drop_datetime_columns(df, datetime_cols):
    return df.drop(columns=datetime_cols)

def impute_missing_values(df, categorical_cols, numerical_cols):
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    return df

def encode_categoricals(df, categorical_cols):
    if not categorical_cols:
        return np.empty((len(df), 0), dtype=np.float32), None
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    return encoded, encoder

def scale_numericals(df, numerical_cols):
    if not numerical_cols:
        return np.empty((len(df), 0), dtype=np.float32), None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numerical_cols])
    return scaled, scaler

def combine_parts(cat_encoded, num_scaled, bin_data, n_rows):
    parts = [p for p in [cat_encoded, num_scaled, bin_data] if p.size > 0]
    if parts:
        return np.hstack(parts).astype(np.float32)
    return np.empty((n_rows, 0), dtype=np.float32)

def preprocess_data(df, cardinality_threshold=20, binary_as_categorical=True):
    df = df.copy()

    cat_cols, num_cols, bin_cols, dt_cols = classify_columns(
        df, cardinality_threshold, binary_as_categorical
    )

    df = drop_datetime_columns(df, dt_cols)
    
    df = impute_missing_values(df, cat_cols, num_cols)

    cat_encoded, encoder = encode_categoricals(df, cat_cols)
    num_scaled, scaler = scale_numericals(df, num_cols)

    bin_data = df[bin_cols].values.astype(np.float32) if (bin_cols and not binary_as_categorical) else np.empty((len(df), 0), dtype=np.float32)

    processed = combine_parts(cat_encoded, num_scaled, bin_data, n_rows=len(df))

    feature_names = []
    if cat_cols:
        feature_names += OneHotEncoder(sparse_output=False, handle_unknown='ignore').get_feature_names_out
        feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    if num_cols:
        feature_names += num_cols
    if bin_cols and not binary_as_categorical:
        feature_names += bin_cols

    return processed, feature_names, {
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
        "binary_cols": bin_cols,
        "binary_as_categorical": binary_as_categorical,
        "encoder": encoder,
        "scaler": scaler
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
