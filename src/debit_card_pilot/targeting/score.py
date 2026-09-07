import pandas as pd

from debit_card_pilot.targeting.train import FEATURE_COLS


def score_candidates(
    pipeline,
    candidate_features: pd.DataFrame,
) -> pd.DataFrame:
    """Genera score, ranking y decil de prioridad para candidatos."""

    scores = pipeline.predict_proba(
        candidate_features[FEATURE_COLS]
    )[:, 1]

    ranking = candidate_features[
        ["cliente_id"]
    ].copy()

    ranking["priority_score"] = scores

    ranking["priority_rank"] = (
        ranking["priority_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    ranking["priority_decile"] = pd.qcut(
        ranking["priority_rank"],
        q=10,
        labels=list(range(10, 0, -1)),
    ).astype(int)

    return (
        ranking
        .sort_values("priority_rank")
        .reset_index(drop=True)
    )