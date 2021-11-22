for k in 5 11 #50 60 64 20 30 40 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 
do

for nr in 0.2 #0.0

do

# sbatch ~/scavenger.sh calculate_iou.py --paths ${k},1,1,${nr}\;${k},1,50,${nr} --num_samples 1000 --resolution 50 --k $k --plot_method train
sbatch ~/scavenger.sh calculate_iou.py --paths ${k},1,1,${nr}\;${k},25,1,${nr} --num_samples 1000 --resolution 50 --k $k --plot_method test --range_l 0.1 --range_r 0.1

done
done

# for k in 1 2 4 8 11 12 18 #3 5 6 7 9 11 12 13 14 16 18 19 #1 2 4 8 10 15 20 64

# do

# sbatch ~/scavenger.sh main.py --k $k --active_log --epochs 4000 --noise_rate 0.0 --set_data_seed 50
# sbatch ~/scavenger.sh main.py --k $k --active_log --epochs 4000 --noise_rate 0.2 --set_data_seed 50


# done



