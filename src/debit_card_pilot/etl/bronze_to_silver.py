# %% Imports

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Column, DataFrame, functions as F
from pyspark.sql.types import StringType


# %% Configuración del Glue Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

# Evita diferencias de interpretación de timestamps entre entornos.
spark.conf.set("spark.sql.session.timeZone", "UTC")

job = Job(glue_context)
job.init(args["JOB_NAME"], args)


# %% Configuración del pipeline

DATABASE_NAME = "debit_card_pilot"

SILVER_BASE_PATH = (
    "s3://bg-ds-debit-card-pilot-bucket/silver"
)

BRONZE_TABLES = {
    "customers": "bronze_customers",
    "cards": "bronze_cards",
    "transactions": "bronze_transactions",
    "marketing_interactions": "bronze_marketing_interactions",
    "merchant_catalog": "bronze_merchant_catalog",
}

MISSING_TOKENS = ["", "nan", "null", "none"]


# %% Lectura desde Glue Data Catalog

def read_catalog_table(table_name: str) -> DataFrame:
    """Lee una tabla Bronze desde Glue Data Catalog."""

    dynamic_frame = glue_context.create_dynamic_frame.from_catalog(
        database=DATABASE_NAME,
        table_name=table_name,
    )

    return dynamic_frame.toDF()


datasets = {
    name: read_catalog_table(table_name)
    for name, table_name in BRONZE_TABLES.items()
}


# %% Validación inicial

for name, df in datasets.items():
    print(f"{name}: {df.count():,} rows")
    df.printSchema()


# %% Funciones de limpieza

def normalize_pseudo_missing(df: DataFrame) -> DataFrame:
    """Convierte pseudo-faltantes de columnas string a nulos reales."""

    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            value = F.trim(F.col(field.name))

            df = df.withColumn(
                field.name,
                F.when(
                    F.lower(value).isin(MISSING_TOKENS),
                    F.lit(None),
                ).otherwise(F.col(field.name)),
            )

    return df


def normalize_category(column: str) -> Column:
    """Estandariza espacios y capitalización."""

    return F.lower(
        F.trim(
            F.regexp_replace(
                F.col(column),
                r"\s+",
                " ",
            )
        )
    )


def parse_boolean(column: str) -> Column:
    """Homologa representaciones de verdadero y falso."""

    value = F.lower(
        F.trim(
            F.col(column).cast("string")
        )
    )

    return (
        F.when(
            value.isin("true", "1", "s", "si", "sí"),
            F.lit(True),
        )
        .when(
            value.isin("false", "0", "n", "no"),
            F.lit(False),
        )
        .otherwise(F.lit(None))
    )


def parse_mixed_timestamp(column: str) -> Column:
    """Convierte fechas convencionales y timestamps Unix.

    Usa try_to_timestamp en lugar de to_timestamp, y castea a double
    dentro de cada rama del When (no sobre toda la columna de forma
    incondicional), para evitar que un valor no interpretable
    provoque una excepción bajo modo ANSI; en su lugar, el resultado
    será null.
    """

    value = F.trim(
        F.col(column).cast("string")
    )

    epoch_milliseconds = F.when(
        value.rlike(r"^-?\d{12,13}$"),
        F.to_timestamp(
            F.from_unixtime(value.cast("double") / 1000)
        ),
    )

    epoch_seconds = F.when(
        value.rlike(r"^-?\d{10}$"),
        F.to_timestamp(
            F.from_unixtime(value.cast("double"))
        ),
    )

    return F.coalesce(
        epoch_milliseconds,
        epoch_seconds,
        F.try_to_timestamp(
            value,
            F.lit("yyyy-MM-dd HH:mm:ss"),
        ),
        F.try_to_timestamp(
            value,
            F.lit("yyyy-MM-dd"),
        ),
        F.try_to_timestamp(
            value,
            F.lit("dd/MM/yyyy HH:mm:ss"),
        ),
        F.try_to_timestamp(
            value,
            F.lit("dd/MM/yyyy"),
        ),
    )


def parse_amount(column: str) -> Column:
    """Normaliza formato monetario y convierte monto a decimal.

    Preserva el signo negativo (p. ej. transacciones de cashout);
    solo se eliminan símbolo de moneda, comas de miles y espacios.
    Usa try_cast para que un valor no convertible resulte en null
    en lugar de provocar una excepción bajo modo ANSI.
    """

    return F.expr(
        f"try_cast(regexp_replace(trim(`{column}`), '[\\\\$,\\\\s]', '') as decimal(18,2))"
    )


# %% Transformación de Customers

customers_silver = (
    normalize_pseudo_missing(datasets["customers"])
    .dropDuplicates()
)

for column in [
    "ciudad",
    "canal_adquisicion",
    "estado_cuenta",
]:
    customers_silver = customers_silver.withColumn(
        column,
        normalize_category(column),
    )

customers_silver = (
    customers_silver
    .withColumn(
        "fecha_nacimiento",
        F.to_date(
            parse_mixed_timestamp("fecha_nacimiento")
        ),
    )
    .withColumn(
        "fecha_registro",
        parse_mixed_timestamp("fecha_registro"),
    )
    .dropDuplicates()
)


# %% Transformación de Cards

cards_silver = (
    normalize_pseudo_missing(datasets["cards"])
    .dropDuplicates()
    .withColumn(
        "tipo",
        normalize_category("tipo"),
    )
    .withColumn(
        "fecha_emision",
        parse_mixed_timestamp("fecha_emision"),
    )
    .withColumn(
        "fecha_activacion",
        parse_mixed_timestamp("fecha_activacion"),
    )
    .dropDuplicates()
)


# %% Transformación de Transactions

transactions_silver = (
    normalize_pseudo_missing(datasets["transactions"])
    .dropDuplicates()
    .withColumn(
        "tipo_transaccion",
        normalize_category("tipo_transaccion"),
    )
    .withColumn(
        "es_devolucion",
        parse_boolean("es_devolucion"),
    )
    .withColumn(
        "fecha",
        parse_mixed_timestamp("fecha"),
    )
    .withColumn(
        "monto",
        parse_amount("monto"),
    )
    .dropDuplicates()
)


# %% Transformación de Marketing Interactions

marketing_silver = (
    normalize_pseudo_missing(
        datasets["marketing_interactions"]
    )
    .dropDuplicates()
)

for column in ["campana", "canal"]:
    marketing_silver = marketing_silver.withColumn(
        column,
        normalize_category(column),
    )

marketing_silver = (
    marketing_silver
    .withColumn(
        "respondio",
        parse_boolean("respondio"),
    )
    .withColumn(
        "fecha_contacto",
        parse_mixed_timestamp("fecha_contacto"),
    )
    .dropDuplicates()
)


# %% Transformación de Merchant Catalog

merchant_catalog_silver = (
    normalize_pseudo_missing(
        datasets["merchant_catalog"]
    )
    .dropDuplicates()
    .withColumn(
        "categoria",
        normalize_category("categoria"),
    )
    .dropDuplicates()
)


# %% Flags de calidad referencial

customer_lookup = (
    customers_silver
    .select("cliente_id")
    .distinct()
    .withColumn(
        "_customer_match",
        F.lit(True),
    )
)


def add_customer_orphan_flag(
    df: DataFrame,
) -> DataFrame:
    """Marca registros cuyo cliente no existe en Customers."""

    return (
        df
        .join(
            customer_lookup,
            on="cliente_id",
            how="left",
        )
        .withColumn(
            "is_orphan_customer",
            F.col("cliente_id").isNotNull()
            & F.col("_customer_match").isNull(),
        )
        .drop("_customer_match")
    )


cards_silver = add_customer_orphan_flag(
    cards_silver
)

transactions_silver = add_customer_orphan_flag(
    transactions_silver
)

marketing_silver = add_customer_orphan_flag(
    marketing_silver
)


# %% Flag de comercios no catalogados

merchant_lookup = (
    merchant_catalog_silver
    .select("comercio_codigo")
    .distinct()
    .withColumn(
        "_merchant_match",
        F.lit(True),
    )
)

transactions_silver = (
    transactions_silver
    .join(
        merchant_lookup,
        on="comercio_codigo",
        how="left",
    )
    .withColumn(
        "is_uncatalogued_merchant",
        F.col("comercio_codigo").isNotNull()
        & F.col("_merchant_match").isNull(),
    )
    .drop("_merchant_match")
)


# %% Datasets Silver

silver_datasets = {
    "customers": customers_silver,
    "cards": cards_silver,
    "transactions": transactions_silver,
    "marketing_interactions": marketing_silver,
    "merchant_catalog": merchant_catalog_silver,
}


# %% Validación post-transformación (Bronze vs. Silver)

for name, df in silver_datasets.items():
    bronze_rows = datasets[name].count()
    silver_rows = df.count()

    print(f"\n{name.upper()}")
    print(f"Bronze rows: {bronze_rows:,}")
    print(f"Silver rows: {silver_rows:,}")
    print(f"Rows removed: {bronze_rows - silver_rows:,}")

    df.printSchema()


# %% Escritura a Silver

for name, df in silver_datasets.items():
    output_path = f"{SILVER_BASE_PATH}/{name}/"

    (
        df
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    print(f"Silver written: {output_path}")


# %% Finalización del Glue Job

job.commit()