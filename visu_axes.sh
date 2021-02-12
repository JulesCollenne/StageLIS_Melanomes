#! /bin/bash

source ./venv/bin/activate
cd src/
i=0
for filename in /home/adrien/ISIC_2019/NON_SEGMENTEES/TRAIN/NEV/*; do
	python example_visualization.py ${filename##*/}
	((i++))
	if (($i >= 10)); then
		break
	fi
done
