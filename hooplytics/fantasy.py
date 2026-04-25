"""Fantasy scoring helpers."""
from __future__ import annotations

import pandas as pd

from .constants import FANTASY_WEIGHTS


def fantasy(df: pd.DataFrame) -> pd.Series:
    """Apply the fantasy scoring formula to a DataFrame containing pts/reb/ast/stl/blk/tov."""
    return sum(df[c] * w for c, w in FANTASY_WEIGHTS.items())  # type: ignore[return-value]
