for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 16 15 16 17 18 19 20 30 40 50 60 64

do

for nr in 0.0 0.2

python dd_fragmentation.py --k $k --active_log --noise_rate $nr --num_samples 1000 --resolution 50 --range_l 0.1 --range_r 0.1

done
done