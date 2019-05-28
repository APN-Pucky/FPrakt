cd "$(dirname "$0")"
cd ../data
for filename in *.txt; do
  base=${filename%r.txt}
  awk '{sum =94+NR; printf "%4d %s\n",sum,$0}' "$filename" > "${base}r-cut.txt"
done
