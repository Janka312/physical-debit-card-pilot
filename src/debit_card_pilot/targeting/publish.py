from io import BytesIO, StringIO

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
) -> tuple[str, str]:
    """Publica el ranking en Parquet y CSV dentro de Gold."""

    base_key = f"{GOLD_PREFIX}/customer_targeting"

    # Parquet para analítica
    parquet_buffer = BytesIO()

    ranking.to_parquet(
        parquet_buffer,
        index=False,
    )

    parquet_key = f"{base_key}/customer_targeting.parquet"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=parquet_key,
        Body=parquet_buffer.getvalue(),
    )

    # CSV para consumo ligero desde API
    csv_buffer = StringIO()

    ranking.to_csv(
        csv_buffer,
        index=False,
    )

    csv_key = f"{base_key}/customer_targeting.csv"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=csv_key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
    )

    return (
        f"s3://{S3_BUCKET}/{parquet_key}",
        f"s3://{S3_BUCKET}/{csv_key}",
    )


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