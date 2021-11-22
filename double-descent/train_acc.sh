for k in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 30 40 50 60 64

do

for nr in 0.0 0.2
do

python dd_fragmentation.py --k $k --active_log --noise_rate $nr --num_samples 1000 --resolution 50 --range_l 0.1 --range_r 0.1

done
done

# # # # python dd_fragmentation.py --k 1 --noise_rate 0.0 --num_samples 10 --active_log


# # for k in 20 64 4 8 10 15 

# # do

# # sbatch ~/high.sh calculate_iou.py --paths ${k},1,0.2\;${k},25,0.2\;${k},1,0.0 --num_samples 1000 #--plot_method test

# # done

# # for k in 2 3 5 6 7 9 #11 12 13 14 16 18 19 #1 4 8 10 15 20 64 #

# # do

# # # sbatch ~/scavenger.sh main.py --k $k --active_log --epochs 4000 --noise_rate 0.2 --set_seed 25



# # done


# for k in 64 1 10 4 20  

# do

# for nr in 0.2 0.0

# do

# # python db_plot.py --k $k --imgs 12,35780,2050 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/temp --plot_method train_ids --noise_rate $nr

# python db_plot.py --k $k --imgs 672,766,781 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/3random2/ --plot_method train_ids --noise_rate $nr
# #
# # python db_plot.py --k $k --imgs 75,32,304 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/3corr/1 --plot_method train_ids --noise_rate $nr 

# # python db_plot.py --k $k --imgs 75,32,2204 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/${nr}/2corr_mislabofclass/1/ --plot_method train_ids --noise_rate $nr

# # python db_plot.py --k $k --imgs 599,714,2050 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/${nr}/2corr_mislaboutsider/1/ --plot_method train_ids --noise_rate $nr

# done
# done

# # #25754,468,3791

# #old 3 rnd - 12,35780,8791


# python db_plot.py --k 64 --imgs 75,32,2328 --range_l 0.5 --range_r 0.5 --plot_path ./temp --noise_rate 0.2 --plot_method train_ids


# for k in 64 10

# do
# for cls in 1 #1 2 3 4 5 6 7 8 9
# do
# for nr in 0.2 #0.0

# do

# for tm in 11 12 15 16 18 29 38 45 76 89

# do 
# python db_plot.py --k $k --plot_train_class $cls --imgs ${tm},34,12 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/2c1m_3/${cls} --plot_method train_1inc --noise_rate $nr 

# # python db_plot.py --k $k --plot_train_class $cls --imgs 4,76,455 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/2c1m_2/${cls} --plot_method train_1inc --noise_rate $nr 
# done
# done
# done
# done

# 48077  7462] [107