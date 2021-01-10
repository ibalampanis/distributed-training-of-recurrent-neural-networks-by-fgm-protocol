# Distributed Training of Recurrent Neural Networks by FGM protocol #

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

For more information about this work, please check the thesis, [here](http://bit.ly/3q1fvYM).

The presentation of this work can be found [here](https://bit.ly/3ox4heq).
