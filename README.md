<h1 align="center">GPU-accelerated Guided Source Separation</h1>

**Guided source separation** is a type of blind source separation (blind = no training required)
in which the mask estimation is guided by a diarizer output. The original method was proposed
for the CHiME-5 challenge in [this paper](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf) by Boeddeker et al.

It is a kind of target-speaker extraction method. The inputs to the model are:

1. A multi-channel recording, e.g., from an array microphone, of a long, unsegmented,
multi-talker session (possibly with overlapping speech)
2. An RTTM file containing speaker segment boundaries

The system produces enhanced audio for each of the segments in the RTTM, removing the background
speech and noise and "extracting" only the target speaker in the segment.

This repository contains a GPU implementation of this method in Python, along with CLI binaries
to run the enhancement from shell. We also provide several example "recipes" for using the
method.

## Features

The core components of the tool are borrowed from [ `pb_chime5` ](https://github.com/fgnt/pb_chime5), but GPU support is added by porting most of the work to [CuPy](https://github.com/cupy/cupy).

* All the main components of the pipeline --- STFT computation, WPE, mask estimation with CACGMM, and beamforming ---
are ported to CuPy to use GPUs. For CACGMM, we batch all frequency indices instead of iterating over them.
* We have implemented batch processing of segments (see [this issue](https://github.com/desh2608/gss/issues/12) for details)
to maximize GPU memory usage and provide additional speed-up.
* The GSS implementation (see `gss/core`) has been stripped of CHiME-6 dataset-specific peculiarities
(such as array naming conventions etc.)
* We use Lhotse for simplified data loading, speaker activity generation, and RTTM representation. We provide
examples in the `recipes` directory for how to use the `gss` module for several datasets.
* The inference can be done on multi-node GPU environment. This makes it several times faster than the
original CPU implementation.
* We provide both Python modules and CLI for using the enhancement functions, which can be
easily included in recipes from Kaldi, Icefall, ESPNet, etc.

As an example, applying GSS on a LibriCSS OV20 session (~10min) took ~160s on a single RTX2080 GPU (with 12G memory).
See the `test.pstats` for the profiling output.

## Installation

### Preparing to install

Create a new Conda environment:

```bash
conda create -n gss python=3.8
```

Install CuPy as follows (see https://docs.cupy.dev/en/stable/install.html for the appropriate version
for your CUDA). Note that the following installs a pre-release version.

```bash
pip install cupy-cuda102 --pre -f https://pip.cupy.dev/pre
```

### Install (basic)

```bash
pip install git+http://github.com/desh2608/gss
```

### Install (advanced)

```bash
git clone https://github.com/desh2608/gss.git & cd gss
pip install -e '.[dev]'
pre-commit install # installs pre-commit hooks with style checks
```

## Usage

### Enhancing a single recording

For the simple case of target-speaker extraction given a multi-channel recording and an
RTTM file denoting speaker segments, run the following:

```bash
export CUDA_VISIBLE_DEVICES=0
gss enhance recording \
  /path/to/sessionA.wav /path/to/rttm exp/enhanced_segs \
  --recording-id sessionA --min-segment-length 0.1 --max-segment-length 10.0 \
  --max-batch-duration 20.0 --num-buckets 2 -o exp/segments.jsonl.gz
```

### Enhancing a corpus

See the `recipes` directory for usage examples. The main stages are as follows:

1. Prepare Lhotse manifests. See [this list](https://lhotse.readthedocs.io/en/latest/corpus.html#standard-data-preparation-recipes) of corpora currently supported in Lhotse.
You can also apply GSS on your own dataset by preparing it as Lhotse manifests.

2. If you are using an RTTM file to get segments (e.g. in CHiME-6 Track 2), convert the RTTMs
to Lhotse-style supervision manifest.

3. Create recording-level cut sets by combining the recording with its supervisions. These
will be used to get speaker activities.

4. Trim the recording-level cut set into segment-level cuts. These are the segments that will
actually be enhanced.

5. (Optional) Split the segments into as many parts as the number of GPU jobs you want to run. In the
recipes, we submit the jobs through `qsub` , similar to Kaldi or ESPNet recipes. You can
use the parallelization in those toolkits to additionally use a different scheduler such as
SLURM.

6. Run the enhancement on GPUs. The following options can be provided:

* `--num-channels`: Number of channels to use for enhancement. By default, all channels are used.

* `--bss-iteration`: Number of iterations of the CACGMM inference.

* `--context-duration`: Context (in seconds) to include on both sides of the segment.

* `--min-segment-length`: Any segment shorter than this value will be removed. This is
particularly useful when using segments from a diarizer output since they often contain
very small segments which are not relevant for ASR. A recommended setting is 0.1s.

* `--max-segment-length`: Segments longer than this value will be chunked up. This is
to prevent OOM errors since the segment STFTs are loaded onto the GPU. We use a setting
of 15s in most cases.

* `--max-batch-duration`: Segments from the same speaker will be batched together to increase
GPU efficiency. We used 20s batches for enhancement on GPUs with 12G memory. For GPUs with
larger memory, this value can be increased.

* `--max-batch-cuts`: This sets an upper limit on the maximum number of cuts in a batch. To
simulate segment-wise enhancement, set this to 1.

* `--num-workers`: Number of workers to use for data-loading (default = 1). Use more if you
increase the `max-batch-duration` .

* `--num-buckets`: Number of buckets to use for sampling. Batches are drawn from the same
bucket (see Lhotse's [ `DynamicBucketingSampler` ](https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/sampling/dynamic_bucketing.py) for details).

* `--enhanced-manifest/-o`: Path to manifest file to write the enhanced cut manifest. This
is useful for cases when the supervisions need to be propagated to the enhanced segments,
for downstream ASR tasks, for example.

* `--profiler-output`: Optional path to output stats file for profiling, which can be visualized
using Snakeviz.

* `--force-overwrite`: Flag to force enhanced audio files to be overwritten.

## FAQ

**What happens if I set the `--max-batch-duration` too large?**

The enhancement would still work, but you will see several warnings of the sort:
"Out of memory error while processing the batch. Trying again with <num-chunks> chunks.`
Internally, we have a fallback option to chunk up batches into increasingly smaller
parts in case OOM error is encountered (see `gss.core.enhancer.py`). However, this
would slow down processing, so we recommend reducing the batch size if you see this
warning very frequently.

**I am seeing "out of memory error" a lot. What should I do?**

Try reducing `--max-batch-duration`. If you are enhancing a large number of very small
segments, try providing `--max-batch-cuts` with some small value (e.g., 2 or 3). This
is because batching together a large number of small segments requires memory
overhead which can cause OOMs.

**How to understand the format of output file names?**

The enhanced wav files are named as *recoid-spkid-start_end.wav*, i.e., 1 wav file is
generated for each segment in the RTTM. The "start" and "end" are padded to 6 digits,
for example: 21.18 seconds is encoded as `002118` . This convention should be fine if
your audio duration is under ~2.75 h (9999s), otherwise, you should change the
padding in `gss/core/enhancer.py` .

** How should I generate RTTMs required for enhancement?**

For examples of how to generate RTTMs for guiding the separation, please refer to my
[diarizer](https://github.com/desh2608/diarizer) toolkit.

**How can I experiment with additional GSS parameters?**

We have only made the most important parameters available in the
top-level CLI. To play with other parameters, check out the `gss.enhancer.get_enhancer()` function.

## Contributing

Contributions for core improvements or new recipes are welcome. Please run the following
before creating a pull request.

```bash
pre-commit install
pre-commit run # Running linter checks
```

## Citations

Please refer to the [original](https://github.com/fgnt/pb_chime5) repository for papers
related to GSS. If you used this code for your work, consider citing the repo by clicking on
**Cite this repository**.

You can also cite our CHiME-6 paper which used diarization-based GSS for multi-talker speech reognition.

```
@article{Arora2020TheJM,
  title={The JHU Multi-Microphone Multi-Speaker ASR System for the CHiME-6 Challenge},
  author={Ashish Arora and Desh Raj and Aswin Shanmugam Subramanian and Ke Li and Bar Ben-Yair and Matthew Maciejewski and Piotr Żelasko and Paola Garc{\'i}a and Shinji Watanabe and Sanjeev Khudanpur},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.07898}
}
```
