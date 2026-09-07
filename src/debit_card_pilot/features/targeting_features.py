import pandas as pd


def build_transaction_features(
    base: pd.DataFrame,
    transactions: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """Construye features transaccionales previas a una fecha de referencia."""

    tx = (
        transactions[
            transactions["cliente_id"].isin(base["cliente_id"])
        ]
        .merge(
            base[["cliente_id", "reference_date"]],
            on="cliente_id",
            how="inner",
        )
        .copy()
    )

    tx["days_from_reference"] = (
        tx["fecha"] - tx["reference_date"]
    ).dt.days

    tx = tx[
        tx["days_from_reference"].between(-window_days, -1)
    ].copy()

    tx["amount_volume"] = (
        pd.to_numeric(tx["monto"], errors="coerce").abs()
    )

    features = (
        tx
        .groupby("cliente_id", as_index=False)
        .agg(
            tx_count=("transaccion_id", "count"),
            total_amount=("amount_volume", "sum"),
            active_days=("fecha", "nunique"),
            avg_amount=("amount_volume", "mean"),
            last_transaction_date=("fecha", "max"),
        )
    )

    features = base.merge(
        features,
        on="cliente_id",
        how="left",
    )

    features["recency_days"] = (
        features["reference_date"]
        - features["last_transaction_date"]
    ).dt.days

    numeric_cols = [
        "tx_count",
        "total_amount",
        "active_days",
        "avg_amount",
    ]

    features[numeric_cols] = (
        features[numeric_cols]
        .fillna(0)
    )

    features["recency_days"] = (
        features["recency_days"]
        .fillna(window_days + 1)
    )

    return features.drop(
        columns=["last_transaction_date"]
    )


def add_customer_features(
    features: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Incorpora atributos de cliente disponibles a la fecha de referencia."""

    customer_data = customers[
        [
            "cliente_id",
            "fecha_nacimiento",
            "ciudad",
            "canal_adquisicion",
            "fecha_registro",
        ]
    ].copy()

    customer_data["fecha_nacimiento"] = pd.to_datetime(
        customer_data["fecha_nacimiento"],
        errors="coerce",
    )

    result = features.merge(
        customer_data,
        on="cliente_id",
        how="left",
    )

    invalid_birth = (
        result["fecha_nacimiento"].lt(
            pd.Timestamp("1900-01-01")
        )
        | result["fecha_nacimiento"].gt(
            result["reference_date"]
        )
    )

    result.loc[
        invalid_birth,
        "fecha_nacimiento",
    ] = pd.NaT

    result["age"] = (
        (
            result["reference_date"]
            - result["fecha_nacimiento"]
        ).dt.days
        / 365.25
    )

    result["tenure_days"] = (
        result["reference_date"]
        - result["fecha_registro"]
    ).dt.days

    result["canal_adquisicion"] = (
        result["canal_adquisicion"]
        .fillna("desconocido")
    )

    return result.drop(
        columns=[
            "fecha_nacimiento",
            "fecha_registro",
        ]
    )