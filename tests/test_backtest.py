import numpy as np
import pandas as pd

import nflprob.backtest as backtest


def test_walk_forward_uses_only_prior_seasons(monkeypatch):
    games = []
    for season in range(2019, 2023):
        for game in range(3):
            games.append(
                {
                    "game_id": f"{season}_{game}",
                    "season": season,
                    "week": game + 1,
                    "home_score": 24 + game,
                    "away_score": 20,
                }
            )
    frame = pd.DataFrame(games)
    fit_max_seasons = []

    class FakePredictor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, train):
            fit_max_seasons.append(int(train["season"].max()))
            return self

    def fake_score_holdout(model, test, max_distribution_games=None):
        del model, max_distribution_games
        home = test["home_score"].astype(float).to_numpy()
        away = test["away_score"].astype(float).to_numpy()
        margin = home - away
        total = home + away
        win = (home > away).astype(float)
        return pd.DataFrame(
            {
                "game_id": test["game_id"].to_numpy(),
                "season": test["season"].to_numpy(),
                "home_score": home,
                "away_score": away,
                "actual_margin": margin,
                "actual_total": total,
                "predicted_margin": margin + 0.5,
                "predicted_total": total + 1.0,
                "predicted_home_score": home + 0.75,
                "predicted_away_score": away + 0.25,
                "home_win": win,
                "home_win_probability": np.full(len(test), 0.70),
                "exact_score_probability": np.full(len(test), 0.05),
            }
        )

    monkeypatch.setattr(backtest, "NFLPredictor", FakePredictor)
    monkeypatch.setattr(backtest, "score_holdout", fake_score_holdout)

    report, predictions = backtest.walk_forward_backtest(
        frame,
        start_season=2020,
        end_season=2022,
        min_train_games=2,
    )

    assert fit_max_seasons == [2019, 2020, 2021]
    assert predictions["backtest_season"].tolist() == [2020] * 3 + [2021] * 3 + [2022] * 3
    assert report["summary"]["seasons_tested"] == 3.0
    assert report["summary"]["games"] == 9.0
