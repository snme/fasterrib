#!/bin/bash

set -eo pipefail

DIR="$(cd $(dirname ${0}); pwd)"

cd $DIR/../data/stanford

i=0
for f in ./compressed_dicoms/*.tgz; 
do 
    ((i=i+1));
    echo "$1: $f";
    tar -xzf "$f" -C ./dicoms/; 
done
