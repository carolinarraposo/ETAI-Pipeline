import numpy as np
import pandas as pd
  
def flag_invalid_values(df, rules):
    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = pd.Series(False, index=df.index)

        if "min" in bounds:
            invalid = invalid | (numeric < bounds["min"])

        if "max" in bounds:
            invalid = invalid | (numeric > bounds["max"])

        df.loc[invalid, column] = np.nan

    return df