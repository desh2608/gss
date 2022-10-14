# Guided Source Separation with GPU

**NOTE:** Originally this repository was supposed to be a simplified version of [pb_chime5](https://github.com/fgnt/pb_chime5/tree/master/pb_chime5) toolkit from Paderborn University, but after several modifications, 
it is almost its own codebase, although we still keep it as a fork to honor the original purpose.

**Guided source separation** is a type of blind source separation (blind = no training required)
in which the mask estimation is guided by a diarizer output. The original method was proposed
for the CHiME-5 challenge in [this paper](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf).

## Features

We have borrowed the main components of the tool from `pb_chime5` , but added GPU support by 
porting most of the work to [CuPy](https://github.com/cupy/cupy).

* The main components of the pipeline --- WPE, mask estimation with CACGMM, and beamforming --- are now
built into their own packages: [WPE](https://github.com/desh2608/wpe), [CACGMM](https://github.com/desh2608/cacgmm), and [beamformer](https://github.com/desh2608/beamformer). The code is also
directly included into this package for ease of installation.
* The GSS implementation (see `gss/core`) has been stripped of CHiME-6 dataset-specific peculiarities
(such as array naming conventions etc.)
* We use Lhotse for simplified data loading, speaker activity generation, and RTTM representation. We provide
examples in the `scripts` directory for how to use the `gss` module for several datasets. We
are currently aiming to support LibriCSS, AMI, and AliMeeting.
* The inference can be done on multi-node GPU environment. This makes it several times faster than the 
original CPU implementation.

## Installation

```bash
> conda create -n gss python=3.7
> git clone https://github.com/desh2608/gss.git & cd gss
> pip install -e .
> pre-commit install # installs pre-commit hooks with style checks
```

One of the main requirements for this package is CuPy, which can be installed as:

```bash
> pip install cupy-cuda102
```

Please replace the above with the appropriate CUDA toolkit version that you have.

## Usage

Enhancing any dataset using this tool consists of 2 parts - "prepare" and "enhance" (see
the `run_ami.sh` script for an example).

In the "prepare" stage, we use Lhotse to create a manifest for the data describing the
cuts (which are the individual segments to be enhanced). At this step, you can also pass
RTTM files to define the segments. We also optionally split the cut set into multiple parts. 
Each of these parts will be enhanced on 1 GPU.

In the enhancement stage (see enhance.py), each cut (segment) is processed 1 at a time on
a GPU. After the processing is complete (it may take a while if the RTTM file has a lot of segments), 
the enhanced wav files will be written to `EXP_DIR` . The wav files are named
as *recoid-spkid-start_end.wav*, i.e., 1 wav file is generated for each segment in the RTTM.
The "start" and "end" are padded to 6 digits, for example: 21.18 seconds is encoded as
`002118` . This convention should be fine if your audio duration is under ~2.75 h (9999s), 
otherwise, you should change the padding in `gss/core/enhancer.py` .

For examples of how to generate RTTMs for guiding the separation, please refer to my
[diarizer](https://github.com/desh2608/diarizer) toolkit.

### How to prepare a new dataset for enhancement

If you want to perform enhancement on a new dataset, the important part is to create a
new `prepare_data.py` script in the `scripts` directory, similar to the existing ones.
These data preparation scripts heavily rely on Lhotse manifest preparation. You can find
a the list of existing Lhotse recipes [here](https://lhotse.readthedocs.io/en/latest/corpus.html#standard-data-preparation-recipes).

Additionally, we recommend the scripts to contain the following arguments:

* `--min-segment-length`: Any segment shorter than this value will be removed. This is 
particularly useful when using segments from a diarizer output since they often contain
very small segments which are not relevant for ASR. A recommended setting is 0.2s.

* `--max-segment-length`: Segments longer than this value will be chunked up. This is 
to prevent OOM errors since the segment STFTs are loaded onto the GPU. We use a setting
of 15s in most cases.

Internally, we also have a fallback option to chunk up segments into increasingly smaller
parts in case OOM error is encountered (see `gss.core.enhancer.py` ).

## Contributing

Contributions for core improvements or new recipes are welcome. Please run the following
before creating a pull request.

```bash
> pre-commit run # Running linter checks
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
