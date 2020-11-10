# ONT basecaller running on Coral edge TPU

## Setup and installation

First you need [Coral TPU accelerator](https://coral.ai/products/accelerator/) plugged into your USB port.

Then install [edge TPU runtime](https://coral.ai/docs/accelerator/get-started#1-install-the-edge-tpu-runtime).
We strongly recommend using maximum frequency version and plugging the device into USB3 port.

Install requirements via `pip install -r requirements.txt`

Install [tflite runtime](https://www.tensorflow.org/lite/guide/python).

## Running

`./basecall.py --model networks/paper_both_init3_f128_k21_r5_edgetpu.tflite  --directory your_directory/ --output output.fasta`

You can use different tflite files you get different speed/accuracy tradeoff (lower depthwise kernel `k` means higher speed and smaller accuracy).
Use `*_edgetpu.tflite` files for accelerated Coral inference. 

In case you don't have Coral device and want to test our accuracy, you can use original files.
Note, however, that Tensorflow Lite is *not* optimized for x86_64 inference and it will be terribly slow (Google has it on a roadmap https://www.tensorflow.org/lite/guide/roadmap).
If you want to go this road, remove `  experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {})],` line in `backend.py`.

Finally, you can also use non-edgetpu files for visualisation of the architecture (use https://netron.app/ for that). 

## Reproducing results

If you wish to reproduce results from our paper (https://arxiv.org/abs/2011.04312), head to https://github.com/fmfi-compbio/coral-benchmark and to https://github.com/fmfi-compbio/coral-training/
