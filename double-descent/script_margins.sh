## plot_method train gives scores for mislabeled and correctly labeled points too

for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 16 15 16 17 18 19 20 30 40 50 60 64

do

for nr in 0.0 0.2

python dd_margin.py --k $k --active_log --noise_rate $nr --num_iters 10 --num_samples 5000 --plot_method train

done
done