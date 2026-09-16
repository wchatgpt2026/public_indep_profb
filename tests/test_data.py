import pandas as pd

from nflprob import data


class _FailingNFL:
    def __init__(self) -> None:
        self.calls = 0

    def load_pbp(self, season: int):
        self.calls += 1
        raise ConnectionError(f"transient failure for {season}")


def test_pbp_loader_falls_back_after_retries(monkeypatch):
    nfl = _FailingNFL()
    expected = pd.DataFrame({"game_id": ["2016_01_A_B"], "epa": [0.1]})

    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    monkeypatch.setattr(data, "_fallback_pbp", lambda season: expected.assign(season=season))

    result = data._load_pbp_season(nfl, 2016, attempts=3)

    assert nfl.calls == 3
    assert result.loc[0, "season"] == 2016
    assert result.loc[0, "game_id"] == "2016_01_A_B"
