#!/bin/bash
WD=$(dirname $0)/..
for filename in $WD/raw/*.peaks; do
	awk 'FNR> 2 {OFS = "+-"; print $3,$6/2.3548;}' "$filename" > "$WD/data/$(basename $filename)"
done