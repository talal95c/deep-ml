import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return per-column types, missingness, cardinality,
    statistics and anomaly flags."""

    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    columns = {}

    for name in df.columns:

        s = df[name]

        # --------------------
        # 1. Type de colonne
        # --------------------
        if is_bool_dtype(s):
            kind = "boolean"
        elif is_numeric_dtype(s):
            kind = "numeric"
        else:
            kind = "categorical"

        # --------------------
        # 2. Missing + uniques
        # --------------------
        missing_pct = round(float(s.isna().mean() * 100), 2)
        n_unique = int(s.nunique(dropna=True))
        non_missing = s.dropna()
        non_missing_count = len(non_missing)

        profile = {
            "kind": kind,
            "missing_pct": missing_pct,
            "n_unique": n_unique
        }

        # --------------------
        # 3. Colonnes numériques
        # --------------------
        n_outliers = 0

        if kind == "numeric":

            if non_missing_count == 0:

                profile["mean"] = None
                profile["std"] = None
                profile["min"] = None
                profile["median"] = None
                profile["max"] = None
                profile["n_outliers"] = 0

            else:

                profile["mean"] = round(float(non_missing.mean()), 4)

                if non_missing_count >= 2:
                    profile["std"] = round(
                        float(non_missing.std(ddof=1)), 4
                    )
                else:
                    profile["std"] = None

                profile["min"] = round(float(non_missing.min()), 4)
                profile["median"] = round(float(non_missing.median()), 4)
                profile["max"] = round(float(non_missing.max()), 4)

                # Quartiles
                q1 = non_missing.quantile(0.25)
                q3 = non_missing.quantile(0.75)

                # Interquartile range
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Compter les outliers
                n_outliers = int(
                    (
                        (non_missing < lower_bound)
                        | (non_missing > upper_bound)
                    ).sum()
                )

                profile["n_outliers"] = n_outliers

        # --------------------
        # 4. Boolean / categorical
        # --------------------
        else:

            if non_missing_count == 0:

                profile["top"] = None
                profile["top_freq"] = 0

            else:
                # sort=False permet de conserver
                # l'ordre d'apparition
                counts = non_missing.value_counts(sort=False)

                max_freq = int(counts.max())

                # En cas d'égalité, on veut
                # la première valeur rencontrée
                for value in non_missing:
                    if counts[value] == max_freq:
                        top = value
                        break

                # convertir numpy scalar -> Python scalar
                if hasattr(top, "item"):
                    top = top.item()

                profile["top"] = top
                profile["top_freq"] = max_freq

        # --------------------
        # 5. Flags
        # --------------------
        flags = []

        # ordre demandé par l'énoncé

        if n_unique <= 1:
            flags.append("constant")

        if missing_pct > 50:
            flags.append("high_missing")

        if (
            kind == "categorical"
            and non_missing_count >= 5
            and n_unique > 0.9 * non_missing_count
        ):
            flags.append("high_cardinality")

        if kind == "numeric" and n_outliers > 0:
            flags.append("outliers")

        profile["flags"] = flags

        # Ajouter le profil de cette colonne
        columns[name] = profile

    # --------------------
    # 6. Résultat final
    # --------------------
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "duplicate_rows": duplicate_rows,
        "columns": columns
    }
