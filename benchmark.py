#!/usr/bin/env python3
import argparse
import time
from datetime import datetime

import tflite_runtime.interpreter as tflite
import platform
import numpy

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


def make_interpreter(model_file):
    model_file, *device = model_file.split("@")
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(
                EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
            )
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", help="File path of .tflite file.")
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(input_details["shape"])
    start = time.perf_counter()

    inp = numpy.random.normal(size=input_details["shape"])
    input_quantization = input_details["quantization"]
    inp_rescaled = (inp / input_quantization[0] + input_quantization[1]).astype(numpy.int8)

    N = 10
    for i in range(N):
#        inp = numpy.zeros(input_details["shape"], dtype="int8")
        interpreter.set_tensor(input_details["index"], inp_rescaled)
        i_start = datetime.now()
        interpreter.invoke()
        i_time = (datetime.now() - i_start).total_seconds()
        print("inv time", i_time, inp.shape[0] * inp.shape[1] / i_time) 
        out = interpreter.get_tensor(output_details["index"])

    inference_time = time.perf_counter() - start

    print(inference_time, N * inp.shape[0] * inp.shape[1] / inference_time)
    # print(out)


if __name__ == "__main__":
    main()
