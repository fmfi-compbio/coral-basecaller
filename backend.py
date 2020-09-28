import tflite_runtime.interpreter as tflite
import platform
import numpy as np
from scipy.special import softmax
from fast_ctc_decode import beam_search
from datetime import datetime
import multiprocessing as mp

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

def caller_process(model, qin, qout):
    coral = Coral(model)
    coral.call_raw(np.zeros((4, 5004, 1), np.int8))
    while True:
        item = qin.get()
        if item is None:
            qout.put(None)
            break 
        signal = item[1]
        b_out = coral.call_raw(signal)
        qout.put((item[0], b_out, item[2]))

def finalizer_process(qin, qout, output_details):
    output_quantization = output_details["quantization"]
    cur_name = ""
    cur_out = []
    while True:
        item = qin.get()
        if item is None:
            break

        bounds, b_out, crop = item

        b_out = (b_out - output_quantization[1]) * output_quantization[0]
        b_out = np.split(b_out, b_out.shape[0])
        for bound, out, c in zip(bounds, b_out, crop):
            if bound is not None:
                if len(cur_out) > 0:
                    qout.put((cur_name, "".join(cur_out)))
                cur_out = []
                cur_name = bound
                
            out = out.reshape((-1, 5))
            out = out[c]
            out = softmax(out, axis=1).astype(np.float32)

            ### TF has blank index 4 and we need 0
            out = np.flip(out, axis=1)

            alphabet = "NTGCA"
            seq, path = beam_search(
                out, alphabet, beam_size=5, beam_cut_threshold=0.1
            )
            # TODO: correct write
            cur_out.append(seq)
    if len(cur_out) > 0:
        qout.put((cur_name, "".join(cur_out)))

def batch_process(qin, qout, input_details, pad):
    input_quantization = input_details["quantization"]
    b_len, s_len, _ = input_details["shape"]
    data, crop, bounds = [], [], []
    while True:
        item = qin.get()
        if item is None:
            break
        name, signal = item
        for b, s, c in signal_to_chunks(signal, name, s_len, pad):
            crop.append(c)
            data.append(s)
            bounds.append(b)
            if len(data) == b_len:
                b_signal = np.stack(data)
                b_signal = b_signal.reshape((b_len, s_len, 1))
                b_signal = b_signal / input_quantization[0] + input_quantization[1]
                b_signal = b_signal.astype(np.int8)
                qout.put((bounds, b_signal, crop))
                data, crop, bounds = [], [], []
    if len(data) > 0:
        while len(data) < b_len:
            crop.append(slice(0, 0))
            data.append(data[-1])
            bounds.append(None)
        b_signal = np.stack(data)
        b_signal = b_signal.reshape((b_len, s_len, 1))
        b_signal = b_signal / input_quantization[0] + input_quantization[1]
        b_signal = b_signal.astype(np.int8)
        qout.put((bounds, b_signal, crop))

    qout.put(None)


 


class Basecaller:
    def __init__(self, model_file, input_q, output_q, pad=15):
        self.input_details, self.output_details = self._get_params(model_file)
        b_len, s_len, _ = self.input_details["shape"]

        self.input_q = input_q
        call_q = mp.Queue(100)
        final_q = mp.Queue()

        self.batcher_proc = mp.Process(target=batch_process, args=(input_q, call_q, self.input_details, pad)) 
        self.caller_proc = mp.Process(target=caller_process, args=(model_file, call_q, final_q))
        self.final_proc = mp.Process(target=finalizer_process, args=(final_q, output_q, self.output_details))

        self.batcher_proc.start()
        self.caller_proc.start()
        self.final_proc.start()

    def _get_params(self, model):
        coral = Coral(model)
        return coral.interpreter.get_input_details()[0], coral.interpreter.get_output_details()[0]

    def terminate(self):
        self.input_q.put(None)
        self.final_proc.join()
        self.caller_proc.join()
        self.batcher_proc.join()

