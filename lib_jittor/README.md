# 2024-TPAMI-CamoFormer (CamoFormer-Jittor Implementation)

## Introduction

The repo provides inference code of **CamoFormer (TPAMI-2024)** with [Jittor deep-learning framework](https://github.com/Jittor/jittor).

> **Jittor** is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA, C++.

## Usage

CamoFormer is also implemented in the Jittor toolbox which can be found in `lib_jittor`.
+ Create environment by `python3.7 -m pip install jittor` on Linux. 
A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments. (More details refer to this [installation tutorial](https://github.com/Jittor/jittor#install))

+ First, run `sudo sysctl vm.overcommit_memory=1` to set the memory allocation policy.

+ Second, switch to the project root by `cd CamoFormer/lib_jittor`

+ For testing, run `python test.py`. 

> Note that the Jittor model is just converted from the original PyTorch model via 'snapshot/convert_pkl.py', and thus, the trained weights of PyTorch model can be used to the inference of Jittor model.

## Checkpoints

The download link ([Pytorch](https://drive.google.com/drive/folders/1XTVMbFWmKtp3lWSlQ7XznmHHHjq5-xkp) / [Jittor](https://drive.google.com/file/d/18izUmDF2-wG_-hita0IApIIbxLT-L52j/view?usp=drive_link)) of prediction results on four testing dataset, including CHAMELEON, CAMO, COD10K, NC4K.


## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{yin2024camoformer,
      title={Camoformer: Masked separable attention for camouflaged object detection},
      author={Yin, Bowen and Zhang, Xuying and Fan, Deng-Ping and Jiao, Shaohui and Cheng, Ming-Ming and Van Gool, Luc and Hou, Qibin},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2024},
      publisher={IEEE}
    }
    
    @article{hu2020jittor,
      title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
      author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
      journal={Information Sciences},
      volume={63},
      number={222103},
      pages={1--21},
      year={2020}
    }

## Acknowlegement
Thanks [Ge-Peng Ji](https://gewelsji.github.io/) providing a friendly [jittor-codebase](https://github.com/GewelsJI/SINet-V2/tree/main/jittor_lib) for Camoflaged Object Detection. And our code is built based on it. 
