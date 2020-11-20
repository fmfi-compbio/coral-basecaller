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

def listdir(directory_name):
    for fname in os.listdir(directory_name):
        fname = os.path.join(directory_name, fname)
        if os.path.isdir(fname):
            yield from listdir(fname)
        if fname.endswith(".fast5"):
            yield fname

class Watcher:
    def __init__(self, directories, stability_threshold=5, sleep=1):
        self.candidates = set()
        self.done = set()

        def gen():
            while True:
                for directory_name in directories:
                    new_cnt = 0
                    for f in listdir(directory_name):
                        if f not in self.done and f not in self.candidates:
                            new_cnt += 1
                            self.candidates.add(f)
                    if new_cnt:
                        print("Found %d new files since last time" % new_cnt)
                for f in list(self.candidates):
                    if os.stat(f).st_mtime + stability_threshold < time.time():
                        self.candidates.remove(f)
                        self.done.add(f)
                        yield f
                    else:
                        print("File modified recently, waiting to stabilize", f)
                yield None
                print("Watching for changed files...")
                time.sleep(sleep)
        self.gen = gen()

    def __iter__(self):
        return self.gen
    
    def __len__(self):
        return len(self.candidates) + len(self.done)

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
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--watch-sleep",
        default=10,
        type=int,
        help="Seconds to sleep between directory watches"
    )
    parser.add_argument(
        "--wait-since-last-write",
        default=5,
        type=int,
        help="Seconds to wait from last write before marking newly discovered file as safe to process"
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

    if not args.watch:
        files = args.reads if args.reads else []
        if args.directory:
            for directory_name in args.directory:
                files.extend(listdir(directory_name))
        if len(files) == 0 and not args.watch:
            print("Zero input reads, nothing to do.")
            sys.exit(0)
    else:
        if args.reads:
            print("--reads and --watch combination is unsupported")
            sys.exit(1)
        if not args.directory:
            print("--directory must be supplied when using --watch")
            sys.exit(1)
        files = Watcher(args.directory, args.wait_since_last_write, args.watch_sleep)

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

    qin = mp.Queue(30)
    qout = mp.Queue()

    caller = backend.Basecaller(args.model, qin, qout, beam_size=beam_size, beam_cut_threshold=beam_cut_threshold)

    total_signals = 0
    total_reads = 0
    total_bases = 0
    time.sleep(5)
    start_time = datetime.datetime.now()
    data, crop, bounds = [], [], []
    outstanding = 0

    def read_files(files):
        for fn in files:
            if fn is None:
                # in --watch but no new files to process
                yield None
                continue
            try:
                with get_fast5_file(fn, mode="r") as f5:
                    for read in f5.get_reads():
                        read_id = read.get_read_id()
                        signal = read.get_raw_data()
                        yield signal, dict(read_id=read_id, samples=len(signal))
            except Exception as e:
                print(e, file=sys.stderr)

    def drain_result_queue(outstanding_limit, block):
        global outstanding, total_bases, total_reads, total_signals
        while outstanding > outstanding_limit:
            try:
                metadata, seq = qout.get(block=block)
                total_bases += len(seq)
                total_reads += 1
                total_signals += metadata["samples"]
                write_output(metadata["read_id"], seq, None, fout, "fasta")
                outstanding -= 1
                print("done %d/%d" % (total_reads, len(files)), metadata["samples"], len(seq))
            except queue.Empty:
                break

    print("Starting...")
    try:
        for item in read_files(files):
            if item is None:
                # in --watch and no new files to process, finish outstanding
                drain_result_queue(0, True)
                continue
            read_id, signal = item

            qin.put((read_id, signal))
            outstanding += 1

            drain_result_queue(20, False)

        qin.put(None)
        drain_result_queue(0, True)
    except KeyboardInterrupt as error:
        print("Interrupted, cleaning up...")
        pass

    caller.terminate()

    fin_time = datetime.datetime.now()
    elapsed = (fin_time - start_time).total_seconds()
    print("--------------------")
    print("Basecalling finished")
    print("Elapsed time (sec)", elapsed)
    print("Reads processed", total_reads)
    print("Signals processed (M)", "%.4f" % (total_signals / 1e6))
    print("Bases called (M)", "%0.4f" % (total_bases / 1e6))
    print("Basecalling speed (Msamples/sec)", "%.2f" % (total_signals / 1e6 / elapsed))
    fout.close()
    sys.exit(0)