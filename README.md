# GSS: Guided Source Separation

This repository contains a refactored and simplified version of the [pb_chime5](https://github.com/fgnt/pb_chime5/tree/master/pb_chime5)
toolkit from Paderborn University.

**Guided source separation** is a type of blind source separation (blind = no training required)
in which the mask estimation is guided by a diarizer output. The original method was proposed
for the CHiME-5 challenge in [this paper](http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_boeddecker.pdf).

## Features

The core part of this tool is exactly the same as **pb_chime5**. The difference lies merely in
making it easier to use for datasets other than CHiME-5. Here are the salient differences:

* The GSS implementation (see `gss/core`) has been stripped of CHiME-6 dataset-specific peculiarities
(such as array naming conventions etc.)
* We use [Lhotse](https://github.com/lhotse-speech/lhotse) for simplified data loading, and provide
examples in the `scripts` directory for how to use the `gss` module for several datasets. We
are currently aiming to support LibriCSS, AMI, and AISHELL-4.
* `mpi` has been replaced with [plz](https://github.com/pzelasko/plz) (based on [Dask-Jobqueue](https://jobqueue.dask.org/en/latest/)) 
for multi-node processing. To use on your own cluster, please fork `plz` and add your 
cluster environment (currently it supports CLSP and COE clusters at JHU).

The core implementation still relies on the [pb_bss](https://github.com/fgnt/pb_bss/tree/96fd72cb5934fb3ec21a707cc54ac6263782a71a) 
and [paderbox](https://github.com/fgnt/paderbox) toolkits from Paderborn.

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

### Without using your own RTTM

```bash
> python scripts/run_libricss.py -j 20 /export/data/LibriCSS exp/libricss
```

### Using your own RTTM

The RTTM files must be placed in a directory, with each session having its own RTTM.

```bash
> python scripts/run_libricss.py -j 20 -r data/rttm_dir /export/data/LibriCSS exp/libricss
```

After the processing is complete (it may take a while if the RTTM file has a lot of segments), 
the enhanced wav files will be written to `exp/libricss/enhanced` . The wav files are named
as *recoid-spkid-start_end.wav*, i.e., 1 wav file is generated for each segment in the RTTM.

**NOTE:** If your RTTM contains too many segments, we suggest removing extremely short segments
which are likely to be non-words or false alarms. In the future, we may build this option
into the package itself.

For examples of how to generate RTTMs for guiding the separation, please refer to my
[diarizer](https://github.com/desh2608/diarizer) toolkit.

## Citations

Please refer to the [original](https://github.com/fgnt/pb_chime5) repository for paper citations.
If you found this simplified package useful, consider mentioning it as a footnote!
