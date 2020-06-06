# Distributed Training of Recurrent Neural Networks by FGM protocol #

To resolve all project dependencies, firstly, you must have installed anaconda in your system. 

So, in the root of the folder, open a terminal and type:

```bash
cd conda/
conda env create -f environment.yml
```

After that, in the bash/ folder, to generate cmake files and compile the custom library, type:

```bash
cd ../bash
bash setup-project.bash
```

To compile a target (eg. GMLearn), back to the root of the folder, type:

```bash
cmake --build cmake-build-debug --target GMLearn
```

and to run this target, type:

```bash
./cmake-build-debug/bin/GMLearn absolute/path/to/input/gm_sample.json
```

For more information about this work, check [here](latex/proposal/proposal.pdf) the thesis proposal.
