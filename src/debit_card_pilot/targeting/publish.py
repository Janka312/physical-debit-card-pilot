from io import BytesIO

import boto3
import joblib
import pandas as pd

from debit_card_pilot.config import (
    GOLD_PREFIX,
    MODEL_ARTIFACT_PREFIX,
    S3_BUCKET,
)


def publish_ranking_to_s3(
    ranking: pd.DataFrame,
    s3_client,
) -> str:
    """Publica el ranking final en la capa Gold."""

    buffer = BytesIO()

    ranking.to_parquet(
        buffer,
        index=False,
    )

    key = f"{GOLD_PREFIX}/customer_targeting/customer_targeting.parquet"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
    )

    return f"s3://{S3_BUCKET}/{key}"


def publish_model_to_s3(
    pipeline,
    s3_client,
) -> str:
    """Publica el pipeline entrenado como artefacto del modelo."""

    buffer = BytesIO()

    joblib.dump(
        pipeline,
        buffer,
    )

    buffer.seek(0)

    key = f"{MODEL_ARTIFACT_PREFIX}/targeting_pipeline.joblib"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
    )

    return f"s3://{S3_BUCKET}/{key}"