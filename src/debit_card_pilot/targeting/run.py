from io import BytesIO

import boto3
import pandas as pd

from debit_card_pilot.config import (
    ACTIVATION_WINDOW_DAYS,
    CONTROL_INDEX_DATE,
    PRE_WINDOW_DAYS,
    S3_BUCKET,
    SILVER_PREFIX,
)
from debit_card_pilot.features.targeting_features import (
    add_customer_features,
    build_transaction_features,
)
from debit_card_pilot.targeting.population import (
    build_modeling_populations,
)
from debit_card_pilot.targeting.publish import (
    publish_model_to_s3,
    publish_ranking_to_s3,
)
from debit_card_pilot.targeting.score import score_candidates
from debit_card_pilot.targeting.train import train_final_model


AWS_PROFILE = "ds-technical-test"


def read_s3_parquet_folder(
    s3_client,
    folder: str,
) -> pd.DataFrame:
    """Lee y consolida archivos Parquet desde Silver."""

    prefix = f"{SILVER_PREFIX}/{folder}/"

    paginator = s3_client.get_paginator("list_objects_v2")
    parquet_files = []

    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=prefix,
    ):
        parquet_files.extend(
            obj["Key"]
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        )

    dataframes = []

    for key in parquet_files:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=key,
        )

        dataframes.append(
            pd.read_parquet(
                BytesIO(response["Body"].read())
            )
        )

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


def main() -> None:
    """Ejecuta el pipeline de targeting de punta a punta."""

    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client("s3")

    # Carga Silver
    customers = read_s3_parquet_folder(s3, "customers")
    cards = read_s3_parquet_folder(s3, "cards")
    transactions = read_s3_parquet_folder(s3, "transactions")

    # Poblaciones elegibles
    scoring_date = pd.Timestamp(CONTROL_INDEX_DATE)

    training_base, candidate_base = build_modeling_populations(
        cards=cards,
        customers=customers,
        scoring_date=scoring_date,
        activation_window_days=ACTIVATION_WINDOW_DAYS,
    )

    # Features
    train_features = build_transaction_features(
        training_base,
        transactions,
        PRE_WINDOW_DAYS,
    )

    candidate_features = build_transaction_features(
        candidate_base,
        transactions,
        PRE_WINDOW_DAYS,
    )

    train_features = add_customer_features(
        train_features,
        customers,
    )

    candidate_features = add_customer_features(
        candidate_features,
        customers,
    )

    # Entrenamiento y scoring
    pipeline = train_final_model(train_features)

    ranking = score_candidates(
        pipeline,
        candidate_features,
    )

    # Publicación
    parquet_uri, csv_uri = publish_ranking_to_s3(
        ranking,
        s3,
    )

    model_uri = publish_model_to_s3(
        pipeline,
        s3,
    )

    print(f"Training customers: {len(train_features):,}")
    print(f"Scored candidates: {len(ranking):,}")
    print(f"Parquet published: {parquet_uri}")
    print(f"CSV published: {csv_uri}")
    print(f"Model published: {model_uri}")

if __name__ == "__main__":
    main()