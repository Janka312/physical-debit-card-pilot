import csv
import io
import json

import boto3

BUCKET = "bg-ds-debit-card-pilot-bucket"
KEY = "gold/customer_targeting/customer_targeting.csv"

s3 = boto3.client("s3")


def lambda_handler(event, context):
    """Devuelve los clientes priorizados desde Gold."""

    params = event.get("queryStringParameters") or {}

    try:
        limit = int(params.get("limit", 100))
    except ValueError:
        limit = 100

    # Evita respuestas demasiado grandes
    limit = max(1, min(limit, 1000))

    response = s3.get_object(
        Bucket=BUCKET,
        Key=KEY,
    )

    body = response["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(body))

    results = []

    for row in reader:
        results.append(
            {
                "cliente_id": row["cliente_id"],
                "priority_score": float(row["priority_score"]),
                "priority_rank": int(row["priority_rank"]),
                "priority_decile": int(row["priority_decile"]),
            }
        )

        if len(results) >= limit:
            break

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(results),
    }