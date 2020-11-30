# VSR (Video Super Resolution )
Easy use of good super resolution algorithms, now only EDVR from BasicSR can be used, No mater how big the input is, it can process, so I'm going to release it now even it's still not compelete. 

Colab Demo: [Notebooks](https://colab.research.google.com/drive/13rb0AmNTcAguy48JO4ObfOPHsqQLQYgS?usp=sharing)

### Table of Contents
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Inferencing](#easy-inferencing)


### Citation


### Requirements and Dependencies
- NVIDIA GPU + CUDA (We test with: V100, P100, P4, T4, K80)
- Ubuntu (We test with Ubuntu = 18.04.5 LTS)
- FFmpeg
- [CUDA](https://developer.nvidia.com/cuda-downloads) (We test with CUDA = 10.1)
- cuDNN (Optional)
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)

### Installation
1. Install VSR
    ```bash
    git clone https://github.com/iBobbyTS/VSR.git
    cd VSR
    python build.py
    ```
1. Download models
	```bash
	cd BasicSR
	python scripts/download_pretrained_models.py EDVR
	cd ..
	```
### Easy inferencing
	```
	cd VSR
	python run.py -i input.mp4
	```

Check Other arguements in [run.py](https://github.com/iBobbyTS/VSR/blob/main/run.py). 

### Contact
[iBobby](mailto:iBobbyTS@gmail.com)

### License
See [MIT License](https://github.com/iBobby/VSR/blob/main/LICENSE)
