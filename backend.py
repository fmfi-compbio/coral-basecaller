import tflite_runtime.interpreter as tflite
import platform
import numpy as np
from scipy.special import softmax
from fast_ctc_decode import beam_search
from datetime import datetime

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


class Coral:
    def __init__(self, model_file):
        self.interpreter = tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {})],
        )
        self.interpreter.allocate_tensors()
        self.last_start = datetime.now()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def input_shape(self):
        return self.interpreter.get_input_details()[0]["shape"]

    def call_raw(self, inp):
        self.interpreter.set_tensor(
            self.input_details["index"], inp
        )
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details["index"])

def _slice(raw_signal, start, end):
    pad_start = max(0, -start)
    pad_end = min(max(0, end - len(raw_signal)), end - start)
    return (
        np.pad(
            raw_signal[max(0, start) : min(end, len(raw_signal))],
            (pad_start, pad_end),
            constant_values=(0, 0),
        ),
        pad_start,
        pad_end,
    )


def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def rescale_signal(signal):
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad
    signal = signal.clip(-2.5, 2.5)
    return signal

def signal_to_chunks(raw_signal, read_id, s_len, pad):
    raw_signal = rescale_signal(raw_signal)
    pos = 0
    while pos < len(raw_signal):
        # assemble batch
        signal, pad_start, pad_end = _slice(
            raw_signal, pos - pad, pos - pad + s_len
        )
        crop = slice(max(pad, pad_start) // 3, -max(pad, pad_end) // 3)
        bound = read_id if pos == 0 else None
        pos += s_len - 2 * pad

        yield (bound, signal, crop)
