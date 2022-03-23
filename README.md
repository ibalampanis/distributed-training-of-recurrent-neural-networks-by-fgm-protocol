
# Distributed Training of Recurrent Neural Networks by FGM protocol #

### Abstract

<div align="justify">
Artificial Neural Networks are appealing because they learn by example and are strongly supportedby statistical and optimization theories. The usage of recurrent neural networks as identifiers andpredictors in non linear dynamic systems has increased significantly. They can present a wide range of dynamics, due to feedback and are also flexible nonlinear maps. Based on this, there is a need for distributed training on these networks, because of the enormous datasets. One of the most known protocols for distributed training is the Geometric Monitoring protocol. Our conviction is that this is a very expensive protocol regarding the communication of nodes. Recently, the Functional Geometric Protocol has tested training on Convolutional Neural Networks and has had encouraging results. The goal of this work is to test and compare these two protocols on Recurrent Neural Networks.
</div>

For more information about this work, please check [here](http://bit.ly/3q1fvYM) the thesis.

The presentation of this work can be found [here](https://bit.ly/3ox4heq).

### Code execution instructions
To resolve all project dependencies, firstly, you must have installed anaconda in your system. 

So, in the root of the folder, open a terminal and type:

```bash
cd conda_env/
conda env create -f environment.yml
```

After that, in the bash/ folder, to generate cmake files and compile the custom library, type:

```bash
cd ../bash
chmod a+x *.bash
bash setup-project.bash
```

To compile a target (eg. FGMLearn), back to the root of the folder, type:

```bash
cmake --build cmake-build-debug --target FGMLearn
```

and to run this target, type:

```bash
./cmake-build-debug/bin/FGMLearn absolute/path/to/input/gm_sample.json
```
