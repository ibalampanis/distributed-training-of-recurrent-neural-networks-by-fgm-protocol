#!/bin/bash

cd /home/ibalampanis/CLionProjects/thesis/text

/bin/bash /home/ibalampanis/CLionProjects/thesis/text/clean-latex-files.bash
rm -f thesis.pdf

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/thesis/text/ thesis.tex

biber --input-directory=/home/ibalampanis/CLionProjects/thesis/text thesis

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=/home/ibalampanis/CLionProjects/thesis/text thesis.tex

/bin/bash /home/ibalampanis/CLionProjects/thesis/text/clean-latex-files.bash

