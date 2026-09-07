import pandas as pd


def classify_card_ownership(cards: pd.DataFrame) -> pd.DataFrame:
    """Clasifica clientes según tenencia de tarjeta física y virtual."""

    ownership = (
        cards
        .assign(
            has_physical=cards["tipo"].eq("fisica"),
            has_virtual=cards["tipo"].eq("virtual"),
        )
        .groupby("cliente_id", as_index=False)
        .agg(
            has_physical=("has_physical", "max"),
            has_virtual=("has_virtual", "max"),
        )
    )

    ownership["card_group"] = "other"

    ownership.loc[
        ownership["has_physical"] & ownership["has_virtual"],
        "card_group",
    ] = "physical_and_virtual"

    ownership.loc[
        ownership["has_physical"] & ~ownership["has_virtual"],
        "card_group",
    ] = "physical_only"

    ownership.loc[
        ~ownership["has_physical"] & ownership["has_virtual"],
        "card_group",
    ] = "virtual_only"

    return ownership


def build_modeling_populations(
    cards: pd.DataFrame,
    customers: pd.DataFrame,
    scoring_date: pd.Timestamp,
    activation_window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye poblaciones elegibles de entrenamiento y scoring."""

    ownership = classify_card_ownership(cards)
    valid_customer_ids = set(customers["cliente_id"])

    train_ids = set(
        ownership.loc[
            ownership["card_group"].eq("physical_and_virtual"),
            "cliente_id",
        ]
    ) & valid_customer_ids

    candidate_ids = set(
        ownership.loc[
            ownership["card_group"].eq("virtual_only"),
            "cliente_id",
        ]
    ) & valid_customer_ids

    # Población de entrenamiento
    physical_cards = cards[
        cards["tipo"].eq("fisica")
        & cards["cliente_id"].isin(train_ids)
    ].copy()

    physical_cards["activation_delay_days"] = (
        physical_cards["fecha_activacion"]
        - physical_cards["fecha_emision"]
    ).dt.days

    physical_cards["target"] = (
        physical_cards["activation_delay_days"]
        .between(0, activation_window_days)
        .fillna(False)
        .astype(int)
    )

    training_base = (
        physical_cards[
            physical_cards["fecha_emision"].notna()
        ]
        [["cliente_id", "fecha_emision", "target"]]
        .rename(columns={"fecha_emision": "reference_date"})
    )

    train_virtual_issue = (
        cards[
            cards["tipo"].eq("virtual")
            & cards["cliente_id"].isin(training_base["cliente_id"])
        ]
        .groupby("cliente_id", as_index=False)["fecha_emision"]
        .min()
        .rename(columns={"fecha_emision": "virtual_issue_date"})
    )

    training_base = (
        training_base
        .merge(train_virtual_issue, on="cliente_id", how="left")
        .merge(
            customers[["cliente_id", "fecha_registro"]],
            on="cliente_id",
            how="left",
        )
    )

    training_base = (
        training_base[
            training_base["virtual_issue_date"].notna()
            & (
                training_base["virtual_issue_date"]
                <= training_base["reference_date"]
            )
            & (
                training_base["fecha_registro"]
                <= training_base["reference_date"]
            )
        ]
        [["cliente_id", "reference_date", "target"]]
        .copy()
    )

    # Población candidata
    candidate_base = pd.DataFrame({
        "cliente_id": sorted(candidate_ids),
        "reference_date": scoring_date,
    })

    candidate_virtual_issue = (
        cards[
            cards["tipo"].eq("virtual")
            & cards["cliente_id"].isin(candidate_ids)
        ]
        .groupby("cliente_id", as_index=False)["fecha_emision"]
        .min()
        .rename(columns={"fecha_emision": "virtual_issue_date"})
    )

    candidate_base = (
        candidate_base
        .merge(
            customers[["cliente_id", "fecha_registro"]],
            on="cliente_id",
            how="left",
        )
        .merge(
            candidate_virtual_issue,
            on="cliente_id",
            how="left",
        )
    )

    candidate_base = (
        candidate_base[
            candidate_base["virtual_issue_date"].notna()
            & (
                candidate_base["virtual_issue_date"]
                <= candidate_base["reference_date"]
            )
            & (
                candidate_base["fecha_registro"]
                <= candidate_base["reference_date"]
            )
        ]
        [["cliente_id", "reference_date"]]
        .copy()
    )

    return training_base, candidate_base