#!/usr/bin/env python
import sys
import pysam
import numpy as np

sam = pysam.AlignmentFile(sys.argv[1])

stats = {}
istats = {}
dstats = {}
mapped, unmapped = 0, 0


for read in sam:
    if read.is_unmapped:
        stats[read.query_name] = 0
        unmapped += 1
        continue
    if read.is_supplementary:
        continue

    edit_dist = read.get_cigar_stats()[0][-1]
    query_len = read.query_alignment_length
    soft_clip = read.get_cigar_stats()[0][4]
#    print(read.get_cigar_stats(), query_len, edit_dist, edit_dist / query_len)
    inserts = read.get_cigar_stats()[0][1]
    deletes = read.get_cigar_stats()[0][2]
    stats[read.query_name] = (1 - edit_dist/ (query_len)) * 100
    istats[read.query_name] = (inserts/ (query_len)) * 100
    dstats[read.query_name] = (deletes/ (query_len)) * 100
    mapped += 1
#    break

values = np.array(list(stats.values()))

print("mapped", mapped)
print("unmapped", unmapped)
values = np.array(list(stats.values()))
print("id mean", np.mean(values))
print("           10   25   50   75   90")
print("identity", "%4.3f %4.3f %4.3f %4.3f %4.3f" % (np.percentile(values, 10), np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75), np.percentile(values, 90)))
values = np.array(list(istats.values()))
print("inserts ", "%4.3f %4.3f %4.3f %4.3f %4.3f" % (np.percentile(values, 10), np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75), np.percentile(values, 90)))
values = np.array(list(dstats.values()))
print("deletes ", "%4.3f %4.3f %4.3f %4.3f %4.3f" % (np.percentile(values, 10), np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75), np.percentile(values, 90)))
