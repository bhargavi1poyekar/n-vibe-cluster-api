import numpy as np
from app.models.data_models import RSSISignals


def normalization(x):
    return (x - -100) / (0 - -100)


def aggregate_rssi_signals(input_signals: list[RSSISignals]) -> RSSISignals:
    input_signals = [s.signal for s in input_signals]
    print(input_signals)
    mean_signal = np.mean(input_signals, axis=0, dtype=float)
    print(mean_signal)
    mean_rssi_signal = RSSISignals(signal=[m for m in mean_signal])
    return mean_rssi_signal
