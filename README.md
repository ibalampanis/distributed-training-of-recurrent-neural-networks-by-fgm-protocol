# Distributed Training of Recurrent Neural Networks by FGM protocol

To resolve all project dependencies, in the root of the folder, open a terminal and type:
```
$ cd bash/
$ sudo chmod a+x *.bash
$ sudo bash setup-project-dependencies.bash
```
After that, in the same folder, to generate cmake files and compile the custom library, type:
```
$ bash setup-project.bash
```
To compile a target (eg. GMNetEEG), back to the root of the folder, type:
```
$ cmake --build cmake-build-debug --target GMNetEEG
```
and to run this target, type:
```
$ ./cmake-build-debug/bin/GMNetEEG absolute/path/to/simulation/gm_eeg.json
```


For more information about this work, check [here](tex/proposal/proposal.pdf) the thesis proposal.
