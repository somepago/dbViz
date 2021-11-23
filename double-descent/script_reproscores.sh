## Here ${k},1,1,${nr} represents k-values, model init seed, data intialization seed, noise rate
## For the following code to run properly, we need to run script_train.sh file with --set_seed set to 1 and 25

for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 16 15 16 17 18 19 20 30 40 50 60 64

do

for nr in 0.0 0.2

sbatch ~/scavenger.sh calculate_iou.py --paths ${k},1,1,${nr}\;${k},25,1,${nr} --num_samples 1000 --resolution 50 --k $k --plot_method test --range_l 0.1 --range_r 0.1

done
done