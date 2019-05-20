#!/bin/bash
cd "$(dirname "$0")"
cd ../data
awk '{if ($1 >404) print NR}' Zeitkalibrierung_cut.Spe > awk.out
