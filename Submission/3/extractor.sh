#!/bin/bash

cd "/home/vmd/Downloads/CIS-6530_F25-main/Submission"

unzip "$1.zip"

cd "/home/vmd/Downloads/CIS-6530_F25-main/Submission/$1/executable"

for f in *.zip; do
	unzip -P 'infected' -o "$f"
done

for f in *.zip; do
	rm "$f"
done

for f in *; do
	mv "$f" "$1_$f.bin"
done

#for f in *; do
#	mv "$f" "/home/vmd/Documents/submission_3/samples/$f"
#done