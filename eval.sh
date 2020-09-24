#!/bin/bash
time ./basecall.py --model $1  --directory test_data/ --output tmp.fasta

#bwa mem -x ont2d -t 2 barcode01.fasta tmp.fasta >tmp.sam && ./get_accuracy.py tmp.sam
echo $1
