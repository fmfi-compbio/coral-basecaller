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
import queue
import multiprocessing

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
    # TODO: fastq
    #parser.add_argument("--output-format", choices=["fasta", "fastq"], default="fasta")
    parser.add_argument(
        "--gzip-output", action="store_true", help="Compress output with gzip"
    )
    parser.add_argument(
        "--mp-spawn", action="store_true", help="Use spawn method for multiprocess (should help on OSX)"
    )


    args = parser.parse_args()

    if args.mp_spawn:
        multiprocessing.set_start_method('spawn')

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

    if args.gzip_output:
        fout = gzip.open(args.output, "wt")
    else:
        fout = open(args.output, "w")

    qin = mp.Queue(100)
    qout = mp.Queue()

    caller = backend.Basecaller(args.model, qin, qout, beam_size=beam_size, beam_cut_threshold=beam_cut_threshold)

    done = 0
    total_signals = 0
    total_batches = 0
    time.sleep(5)
    start_time = datetime.datetime.now()
    data, crop, bounds = [], [], []
    sent = 0
    received = 0
    for fn in files:
        with get_fast5_file(fn, mode="r") as f5:
            for read in f5.get_reads():
                start = datetime.datetime.now()
                read_id = read.get_read_id()
                signal = read.get_raw_data()
                total_signals += len(signal)
                qin.put((read_id, signal))
                sent += 1
                done += 1
                timing = (datetime.datetime.now() - start).total_seconds()
                print(
                    "done %d/%d" % (done, len(files)),
                    timing, len(signal),
                    len(signal) / timing,
                    file=sys.stderr,
                )

                while sent - received > 20:
                    try:
                        name, seq = qout.get(False)
                        write_output(name, seq, None, fout, "fasta")
                        received += 1
                    except queue.Empty:
                        break

    qin.put(None)

    while sent - received > 0:
        name, seq = qout.get()
        write_output(name, seq, None, fout, "fasta")
        received += 1
       
    a = datetime.datetime.now()
    caller.terminate()
    fin_time = datetime.datetime.now()
    elapsed = (fin_time - start_time).total_seconds()
    print("finalizing", datetime.datetime.now() - a, elapsed, total_signals / elapsed, total_batches / elapsed, total_signals, total_batches)
    fout.close()
