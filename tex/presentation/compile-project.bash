#!/bin/bash

# shellcheck disable=SC2164
cd /home/ibalampanis/CLionProjects/thesis/presentation/

/bin/bash /home/ibalampanis/CLionProjects/thesis/presentation/clean-latex-files.bash
rm -f presentation.pdf

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/thesis/presentation presentation.tex
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/thesis/presentation presentation.tex


/bin/bash /home/ibalampanis/CLionProjects/thesis/presentation/clean-latex-files.bash

