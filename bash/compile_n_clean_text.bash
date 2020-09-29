#!/bin/bash

# shellcheck disable=SC2164
cd /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/text

/bin/bash /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/bash/cleanup_latex_text.bash
rm -f thesis.pdf

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/text/ thesis.tex

biber --input-directory=/home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/text thesis

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/latex/text thesis.tex

/bin/bash /home/ibalampanis/CLionProjects/distributed-training-of-recurrent-neural-networks-by-fgm-protocol/bash/cleanup_latex_text.bash

