#!/usr/bin/env python

from ont_fast5_api.fast5_interface import get_fast5_file
import argparse
import os
import numpy as np
import datetime
import multiprocessing as mp
import sys
import gzip
import backend
from scipy.special import softmax
from fast_ctc_decode import beam_search
import time

def caller_process(model, qin, qout):
    coral = backend.Coral(model)
    coral.call_raw(np.zeros((4, 5004, 1), np.int8))
    last = datetime.datetime.now()
    while True:
        item = qin.get()
        if item is None:
            qout.put(None)
            break 
        signal = item[1]
        b_out = coral.call_raw(signal)
        qout.put((item[0], b_out, item[2]))

    pass

def finalizer_process(fn, qin, b_len, output_details):
    fo = open(fn, "w")
    got_output = False
    output_quantization = output_details["quantization"]
    while True:
        item = qin.get()
        if item is None:
            break

        bounds, b_out, crop = item

        b_out = (b_out - output_quantization[1]) * output_quantization[0]
        b_out = np.split(b_out, b_len)
        for bound, out, c in zip(bounds, b_out, crop):
            if bound is not None:
                if got_output:
                    fo.write("\n")
                fo.write(">%s\n" % bound)
                got_output = True
                
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
            fo.write(seq)
    if got_output:
        fo.write("\n")
    fo.close()


def write_output(read_id, basecall, quals, output_file, format):
    if len(basecall) == 0:
        return
    if format == "fasta":
        print(">%s" % read_id, file=fout)
        print(basecall, file=fout)
    else:  # fastq
        print("@%s" % read_id, file=fout)
        print(basecall, file=fout)
        print("+", file=fout)
        print(quals, file=fout)

def get_params(model):
    coral = backend.Coral(model)
    return coral.interpreter.get_input_details()[0], coral.interpreter.get_output_details()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast caller for ONT reads")

    parser.add_argument(
        "--directory", type=str, nargs="*", help="One or more directories with reads"
    )
    parser.add_argument("--reads", type=str, nargs="*", help="One or more read files")
    parser.add_argument(
        "--output", type=str, required=True, help="Output FASTA file name"
    )
    parser.add_argument("--model",)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size (defaults 5 for 48,56,64,80,96 and 20 for 256). Use 1 for greedy decoding.",
    )
    parser.add_argument(
        "--beam-cut-threshold",
        type=float,
        default=None,
        help="Threshold for creating beams (higher means faster beam search, but smaller accuracy). Values higher than 0.2 might lead to weird errors. Default 0.1 for 48,...,96 and 0.0001 for 256",
    )
    parser.add_argument("--output-format", choices=["fasta", "fastq"], default="fasta")
    parser.add_argument(
        "--gzip-output", action="store_true", help="Compress output with gzip"
    )


    args = parser.parse_args()

    input_details, output_details = get_params(args.model)
    input_quantization = input_details["quantization"]
    b_len, s_len, _ = input_details["shape"]
    pad = 15
 
    qcaller = mp.Queue(100)
    qfinalizer = mp.Queue()
    call_proc = mp.Process(target=caller_process, args=(args.model, qcaller, qfinalizer))
    final_proc = mp.Process(target=finalizer_process, args=(args.output, qfinalizer, b_len, output_details))
    call_proc.start()
    final_proc.start()

    files = args.reads if args.reads else []
    if args.directory:
        for directory_name in args.directory:
            files += [
                os.path.join(directory_name, fn) for fn in os.listdir(directory_name)
            ]

    if len(files) == 0:
        print("Zero input reads, nothing to do.")
        sys.exit()

    if args.beam_size is None:
        beam_size = 5
    else:
        beam_size = args.beam_size

    if args.beam_cut_threshold is None:
        beam_cut_threshold = 0.1
    else:
        beam_cut_threshold = args.beam_cut_threshold


    done = 0
    total_signals = 0
    total_batches = 0
    time.sleep(5)
    start_time = datetime.datetime.now()
    data, crop, bounds = [], [], []
    for fn in files:
        with get_fast5_file(fn, mode="r") as f5:
            for read in f5.get_reads():
                start = datetime.datetime.now()
                read_id = read.get_read_id()
                signal = read.get_raw_data()
                total_signals += len(signal)
                for b, s, c in backend.signal_to_chunks(signal, read_id, s_len, pad):
                    crop.append(c)
                    data.append(s)
                    bounds.append(b)
                    if len(data) == b_len:
                        b_signal = np.stack(data)
                        b_signal = b_signal.reshape((b_len, s_len, 1))
                        b_signal = b_signal / input_quantization[0] + input_quantization[1]
                        b_signal = b_signal.astype(np.int8)
                        total_batches += b_len * s_len
                        qcaller.put((bounds, b_signal, crop))
                        data, crop, bounds = [], [], []

                done += 1
                timing = (datetime.datetime.now() - start).total_seconds()
                print(
                    "done %d/%d" % (done, len(files)),
                    timing, len(signal),
                    len(signal) / timing,
                    file=sys.stderr,
                )


    if len(data) > 0:
        while len(data) < b_len:
            crop.append(slice(0, 0))
            data.append(data[-1])
            bounds.append(None)
        b_signal = np.stack(data)
        b_signal = b_signal.reshape((b_len, s_len, 1))
        b_signal = b_signal / input_quantization[0] + input_quantization[1]
        b_signal = b_signal.astype(np.int8)
        qcaller.put((bounds, b_signal, crop))
        total_batches += b_len * s_len
            



    a = datetime.datetime.now()
    qcaller.put(None)
    final_proc.join()
    qcaller.put(None)
    call_proc.join()
    fin_time = datetime.datetime.now()
    elapsed = (fin_time - start_time).total_seconds()
    print("finalizing", datetime.datetime.now() - a, elapsed, total_signals / elapsed, total_batches / elapsed, total_signals, total_batches)
