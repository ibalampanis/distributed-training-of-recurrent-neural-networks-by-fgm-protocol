# Distributed Training of Recurrent Neural Networks by FGM protocol

To resolve all project dependencies, in the root of the folder, type in a Linux terminal,
```
$ cd bash
$ sudo bash setup-project-dependencies.bash
```
Initially, to run the targets, the 'dml' lib must be built. To build it, in the root of the folder, in a Linux terminal type,
```
$ cmake --build cmake-build-debug --target dml
```
After that, to compile a target (eg. GMNetEEG) type,
```
$ cmake --build cmake-build-debug --target GMNetEEG
```
and to run this type,
```
$ ./cmake-build-debug/bin/GMNetEEG absolute/path/to/simulation/gm_eeg_1.json
```


For more information about this work, check [here](tex/proposal/proposal.pdf) the thesis proposal.
