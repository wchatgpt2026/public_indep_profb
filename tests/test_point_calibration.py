import numpy as np

from nflprob.model import AffineTargetCalibrator, NFLPredictor


def test_affine_calibrator_recovers_linear_level_and_scale():
    predicted = np.linspace(-12.0, 12.0, 200)
    actual = 2.0 + 1.15 * predicted

    calibrator = AffineTargetCalibrator.fit(
        predicted,
        actual,
        prior_strength=0.0,
    )
    calibrated = calibrator.predict(predicted)

    assert abs(calibrator.scale - 1.15) < 1e-10
    assert abs(calibrator.offset - 2.0) < 1e-10
    assert np.max(np.abs(calibrated - actual)) < 1e-10


def test_calibration_strength_is_selected_on_later_oof_slice():
    predicted = np.linspace(-10.0, 10.0, 120)
    actual = 1.5 + 1.20 * predicted

    calibrator, prior_strength, validation_mae = NFLPredictor._select_affine_calibrator(
        predicted,
        actual,
    )

    raw_validation_mae = np.mean(np.abs(actual[90:] - predicted[90:]))
    calibrated_validation_mae = np.mean(
        np.abs(actual[90:] - calibrator.predict(predicted[90:]))
    )

    assert prior_strength < 1_000_000.0
    assert validation_mae < raw_validation_mae
    assert calibrated_validation_mae < raw_validation_mae
