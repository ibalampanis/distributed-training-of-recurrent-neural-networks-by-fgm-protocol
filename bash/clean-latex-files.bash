#!/bin/bash

cd ../latex/proposal
rm -f proposal.aux proposal.bbl proposal.bcf proposal.blg
rm -f proposal.log proposal.out proposal.run.xml proposal.synctex.gz

cd ../thesis
rm -f thesis.aux thesis.bbl thesis.bcf thesis.blg thesis.lof thesis.toc
rm -f thesis.log thesis.out thesis.run.xml thesis.synctex.gz config.aux

cd ../../
rm -rf out/