#!/bin/bash

cd /home/ibalampanis/CLionProjects/thesis/presentation

# shellcheck disable=SC2164
rm -f presentation.aux presentation.bbl presentation.bcf presentation.blg presentation.lof presentation.toc presentation.aux presentation.log
rm -f presentation.log presentation.out presentation.run.xml presentation.synctex.gz config.aux presentation.nav
rm -f presentation.out presentation.snm presentation.synctex.gz presentation.toc

rm -rf ../out/