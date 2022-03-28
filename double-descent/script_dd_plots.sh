# To create Fig. 7
for k in 1 10 64 4 7 20
do
for nr in 0.2 0.0
do

python db_plot.py --k $k --imgs 75,32,304 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/3corr/1 --plot_method train_ids --noise_rate $nr 
python db_plot.py --k $k --imgs 672,766,781 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/3randomc/ --plot_method train_ids --noise_rate $nr

done
done


# To create Fig 8
for k in 10 64
do
for cls in 1
do
for nr in 0.2 0.0
do
for tm in 11 12 15 16 18 29 38 45 76 89
do 
python db_plot.py --k $k --plot_train_class $cls --imgs ${tm},34,12 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/2c1m/${cls} --plot_method train_1inc --noise_rate $nr 
done
done
done
done



# To create Fig 14
for nr in 0.0 0.2
do

python db_plot.py --k 10 --imgs 989,417,93 --plot_train_class 0 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/fig_fragm/ --plot_method train_ids --noise_rate $nr 
python db_plot.py --k 10 --imgs 41,1024,5707 --plot_train_class 2 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/fig_fragm/ --plot_method train_ids --noise_rate $nr 
python db_plot.py --k 10 --imgs 994,412,52 --plot_train_class 7 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_paper/fig_fragm/ --plot_method train_ids --noise_rate $nr 

done



###### APPENDIX PLOTS

# Fig 19 and 20

for k in 30 1 10 64 4 7 20
do
for nr in 0.2 0.0
do
python db_plot.py --k $k --imgs 9,17,26 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_appendix/3same/ --plot_method train_ids --noise_rate $nr
python db_plot.py --k $k --imgs 122,24,45 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_appendix/3random2/ --plot_method 3random --noise_rate $nr
python db_plot.py --k $k --imgs 8923,28,49 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_appendix/3random3/ --plot_method 3random --noise_rate $nr

python db_plot.py --k $k --imgs 122,24,45 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_appendix/2corr1inc/ --plot_method train_1inc --noise_rate $nr --plot_train_class 2
python db_plot.py --k $k --imgs 122,24,45 --range_l 0.5 --range_r 0.5 --plot_path ./imgs/train_appendix/2corr1inc2/ --plot_method train_1inc --noise_rate $nr --plot_train_class 9

done
done