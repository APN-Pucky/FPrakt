#!/bin/bash
WD=$(dirname $0)/..
for filename in $WD/raw/*.csv; do
	awk 'BEGIN{FS=OFS=";"} NF{print $1,$2,-$1*667000000,$2*667000000;}' "$filename" > "$WD/data/$(basename $filename)"
done