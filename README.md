<h1 align="center">GPU-accelerated Guided Source Separation</h1>

**Guided source separation** is a type of blind source separation (blind = no training required)
in which the mask estimation is guided by a diarizer output. The original method was proposed
for the CHiME-5 challenge in [this paper](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf) by Boeddeker et al.

This repository contains a GPU implementation of this method in Python, along with CLI binaries
to run the enhancement from shell. We also provide several example "recipes" for using the
method.

## Features

The core components of the tool are borrowed from [ `pb_chime5` ](https://github.com/fgnt/pb_chime5), but GPU support is added by porting most of the work to [CuPy](https://github.com/cupy/cupy).

* The main components of the pipeline --- WPE, mask estimation with CACGMM, and beamforming --- are now
built into their own packages: [WPE](https://github.com/desh2608/wpe), [CACGMM](https://github.com/desh2608/cacgmm), and [beamformer](https://github.com/desh2608/beamformer). The code is also
directly included into this package for ease of installation.
* The GSS implementation (see `gss/core`) has been stripped of CHiME-6 dataset-specific peculiarities
(such as array naming conventions etc.)
* We use Lhotse for simplified data loading, speaker activity generation, and RTTM representation. We provide
examples in the `recipes` directory for how to use the `gss` module for several datasets. We
are currently aiming to support LibriCSS, AMI, and AliMeeting.
* The inference can be done on multi-node GPU environment. This makes it several times faster than the
original CPU implementation.
* We provide both Python modules and CLI for using the enhancement functions, which can be
easily included in recipes from Kaldi, Icefall, ESPNet, etc.

## Installation

### Preparing to install

Create a new Conda environment:

```bash
conda create -n gss python=3.8
```

Install CuPy as follows (see https://docs.cupy.dev/en/stable/install.html for the appropriate version
for your CUDA).

```bash
pip install cupy-cuda102
```

### Install (basic)

```bash
> pip install git+http://github.com/desh2608/gss
```

### Install (advanced)

```bash
> git clone https://github.com/desh2608/gss.git & cd gss
> pip install -e '.[dev]'
> pre-commit install # installs pre-commit hooks with style checks
```

## Usage

See the `recipes` directory for usage examples. The main stages are as follows:

1. Prepare Lhotse manifests. See [this list](https://lhotse.readthedocs.io/en/latest/corpus.html#standard-data-preparation-recipes) of corpora currently supported in Lhotse.
You can also apply GSS on your own dataset by preparing it as Lhotse manifests.

2. If you are using an RTTM file to get segments (e.g. in CHiME-6 Track 2), convert the RTTMs
to Lhotse-style supervision manifest.

3. Create recording-level cut sets by combining the recording with its supervisions. These
will be used to get speaker activities.

4. Trim the recording-level cut set into segment-level cuts. These are the segments that will
actually be enhanced.

5. Split the segments into as many parts as the number of GPU jobs you want to run. In the
recipes, we submit the jobs through `qsub` , similar to Kaldi or ESPNet recipes. You can
use the parallelization in those toolkits to additionally use a different scheduler such as
SLURM.

6. Run the enhancement on GPUs. The following options can be provided:

* `--num-channels`: Number of channels to use for enhancement. By default, all channels are used.

* `--bss-iteration`: Number of iterations of the CACGMM inference.

* `--min-segment-length`: Any segment shorter than this value will be removed. This is
particularly useful when using segments from a diarizer output since they often contain
very small segments which are not relevant for ASR. A recommended setting is 0.2s.

* `--max-segment-length`: Segments longer than this value will be chunked up. This is
to prevent OOM errors since the segment STFTs are loaded onto the GPU. We use a setting
of 15s in most cases.

Internally, we also have a fallback option to chunk up segments into increasingly smaller
parts in case OOM error is encountered (see `gss.core.enhancer.py` ).

The enhanced wav files will be written to `$EXP_DIR/enhanced` . The wav files are named
as *recoid-spkid-start_end.wav*, i.e., 1 wav file is generated for each segment in the RTTM.
The "start" and "end" are padded to 6 digits, for example: 21.18 seconds is encoded as
`002118` . This convention should be fine if your audio duration is under ~2.75 h (9999s),
otherwise, you should change the padding in `gss/core/enhancer.py` .

For examples of how to generate RTTMs for guiding the separation, please refer to my
[diarizer](https://github.com/desh2608/diarizer) toolkit.

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
