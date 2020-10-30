# ONT basecaller running on Coral edge TPU

## Setup and installation

First you need [Coral TPU accelerator](https://coral.ai/products/accelerator/) plugged into your USB port.

Then install [edge TPU runtime](https://coral.ai/docs/accelerator/get-started#1-install-the-edge-tpu-runtime). We strongly recommend using maximum frequency version.

Install requirements via `pip install -r requirements.txt`

Install [tflite runtime](https://www.tensorflow.org/lite/guide/python).

## Running

`./basecall.py --model networks/paper_both_init3_f128_k21_r5_edgetpu.tflite  --directory your_directory/ --output output.fasta`

You can use different tflite files you get different speed/accuracy tradeoff (lower k means higher speed and smaller accuracy).
Also use `*_edgetpu.tflite` files. The other ones are just for visualisation (you can use netron for that).
