# GSS: Guided Source Separation

This repository contains a refactored and simplified version of the [pb_chime5](https://github.com/fgnt/pb_chime5/tree/master/pb_chime5)
toolkit from Paderborn University.

**Guided source separation** is a type of blind source separation (blind = no training required)
in which the mask estimation is guided by a diarizer output. The original method was proposed
for the CHiME-5 challenge in [this paper](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf).

## Features

The core parts of this package (i.e., WPE, mask estimation, and beamforming) are taken directly
from pb_chime5 (which uses [pb_bss](https://github.com/fgnt/pb_bss/tree/96fd72cb5934fb3ec21a707cc54ac6263782a71a)), and therefore the expected result should be the same as using that toolkit. Our 
major contribution is in streamlining the data pipeline using [Lhotse](https://github.com/lhotse-speech/lhotse)
and providing example scripts for applying the enhancement on several datasets other than CHiME-5.

* The GSS implementation (see `gss/core`) has been stripped of CHiME-6 dataset-specific peculiarities
(such as array naming conventions etc.)
* We use Lhotse for simplified data loading, speaker activity generation, and RTTM representation. We provide
examples in the `scripts` directory for how to use the `gss` module for several datasets. We
are currently aiming to support LibriCSS, AMI, and AISHELL-4.
* For distributed processing,  `mpi` has been replaced with [plz](https://github.com/pzelasko/plz) (based on [Dask-Jobqueue](https://jobqueue.dask.org/en/latest/)). To use on your own cluster, please fork `plz` and add your 
cluster environment (currently it supports CLSP and COE clusters at JHU).

## Installation

We recommend installing in a new Conda environment, since the `pb_bss` branch that this
tool relies on requires very specific NumPy and Scikit-learn versions.

```bash
> conda create -n gss python=3.7 # please use Python 3.7 only
> git clone https://github.com/desh2608/gss.git & cd gss
```

Install Cython, numpy and sklearn:

```bash
> pip install Cython
> pip install numpy==1.20.3
> pip install scikit-learn==0.19.2
```

Add `pb_bss` as a submodule, and install:

```bash
> git submodule init
> git submodule update
> pip install -U -e pb_bss/
```

Finally, install the `gss` package:

```bash
> pip install -e .
```

## Usage

End-to-end runnable scripts are provided in the `scripts` directory for some common
datasets. The enhancement can be done with/without your own diarization output. If you
do not provide your own RTTM file, we will use the ground truth annotations to generate
a "gold" RTTM file which is then used for the mask estimation.

**NOTE:** We currently use `plz` to perform distributed processing, which supports the 
clusters at JHU. To run it on your own cluster, you can clone the Github repo, and add
specifications for your own cluster, similar to [this PR](https://github.com/pzelasko/plz/pull/1/files), 
and then include the argument `grid="your-grid"` in the `plz.map()` call in `gss/executor.py` .

### Without using your own RTTM

```bash
> python scripts/run_libricss.py -j 20 /export/data/LibriCSS exp/libricss
```

### Using your own RTTM

```bash
> python scripts/run_libricss.py -j 20 -r data/rttm_path /export/data/LibriCSS exp/libricss
```

Here, the `rttm_path` can be either a single RTTM file, or a directory containing RTTM files.

After the processing is complete (it may take a while if the RTTM file has a lot of segments), 
the enhanced wav files will be written to `exp/libricss/enhanced` . The wav files are named
as *recoid-spkid-start_end.wav*, i.e., 1 wav file is generated for each segment in the RTTM.
The "start" and "end" are padded to 6 digits, for example: 21.18 seconds is encoded as
`002118` . This convention should be fine if your audio duration is under ~2.75 h (9999s), 
otherwise, you should change the padding in `gss/core/enhancer.py` .

If the RTTM contains too many small segments, we recommend additionally passing the option
`--m 0.2` , which filters out segments shorter than 0.2s.

For examples of how to generate RTTMs for guiding the separation, please refer to my
[diarizer](https://github.com/desh2608/diarizer) toolkit.

## Citations

Please refer to the [original](https://github.com/fgnt/pb_chime5) repository for paper citations.
If you found this simplified package useful, consider mentioning it as a footnote!

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
