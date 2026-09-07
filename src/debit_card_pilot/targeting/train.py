from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "tx_count",
    "total_amount",
    "active_days",
    "avg_amount",
    "recency_days",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "ciudad",
    "canal_adquisicion",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_targeting_pipeline() -> Pipeline:
    """Construye el pipeline de preprocessing y regresión logística."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_final_model(train_features):
    """Entrena el modelo final utilizando toda la población etiquetada."""

    pipeline = build_targeting_pipeline()

    X = train_features[FEATURE_COLS]
    y = train_features["target"]

    pipeline.fit(X, y)

    return pipeline