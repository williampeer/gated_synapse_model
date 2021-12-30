#!/usr/bin/env bash

mkdir archive
cp -r figures archive/
cp -r Logs archive/
cp -r saved archive/

zip -r archive.zip archive/

./delete_dirs.sh
./create_dirs.sh
