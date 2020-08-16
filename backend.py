import tflite_runtime.interpreter as tflite
import platform
import numpy as np
from scipy.special import softmax
from fast_ctc_decode import beam_search

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

    def input_shape(self):
        return self.interpreter.get_input_details()[0]["shape"]

    def call(self, inp):
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        input_quantization = input_details["quantization"]
        output_quantization = output_details["quantization"]

        inp_rescaled = inp / input_quantization[0] + input_quantization[1]
        self.interpreter.set_tensor(
            input_details["index"], inp_rescaled.astype(np.int8)
        )

        self.interpreter.invoke()
        out = self.interpreter.get_tensor(output_details["index"])
        out = (out - output_quantization[1]) * output_quantization[0]
        return out


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


class Caller:
    def __init__(self, model_file):
        self.coral = Coral(model_file)

    def basecall(self, raw_signal):
        b_len, s_len, _ = self.coral.input_shape()

        pad = 3 * 100
        pos = 0

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

        res = []
        while pos < len(raw_signal):
            # assemble batch
            data = []
            crop = []
            for b in range(b_len):
                signal, pad_start, pad_end = _slice(
                    raw_signal, pos - pad, pos - pad + s_len
                )
                crop.append(slice(max(pad, pad_start) // 3, -max(pad, pad_end) // 3))
                data.append(signal)
                pos += s_len - 2 * pad
            b_signal = np.stack(data)
            b_signal = b_signal.reshape((b_len, s_len, 1))
            b_out = self.coral.call(b_signal)

            b_out = np.split(b_out, b_len)
            for out, c in zip(b_out, crop):
                out = out.reshape((-1, 5))
                out = out[c]
                out = softmax(out, axis=1).astype(np.float32)

                ### TF has blank index 4 and we need 0
                out = np.flip(out, axis=1)

                alphabet = "NTGCA"
                seq, path = beam_search(
                    out, alphabet, beam_size=5, beam_cut_threshold=0.1
                )
                res.append(seq)

        bases = "".join(res)
        return bases, None  # "B" * len(bases)
