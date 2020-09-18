#!/bin/bash

# shellcheck disable=SC2164
cd /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/presentation/

/bin/bash /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/bash/cleanup_latex_pres.bash
rm -f main.pdf

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/presentation main.tex
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/presentation main.tex


/bin/bash /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/bash/cleanup_latex_pres.bash

