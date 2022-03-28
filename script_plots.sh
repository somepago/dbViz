## To create Fig. 1 and Fig. 3

declare -a ModelTypes=('resnet' 'fcnet' 'vit' 'densenet' 'vgg' 'MLPMixer4')

for mod in "${ModelTypes[@]}";
do

for seed in  0 1 2
do

python main.py --net $mod --load_net <path_to_saved_models>.pth --plot_path ./images/paper/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

done
done


for wid in 10 20 30
do
for seed in 0 1 2
do
python main.py --net WideResNet --widen_factor $wid --load_net <path_to_saved_models>.pth --plot_path ./images/paper/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5
done
done





### Plot uniform noise and random shuffled images (i.e. Figs. 2, 11, 12)

declare -a ModelTypes=('resnet' 'fcnet' 'vit' 'densenet' 'vgg' 'MLPMixer4')
for mod in "${ModelTypes[@]}";
do
for seed in 0 1 2 3 4 5 6 7 8 9
do
python main.py --net $mod --noise_type uniform_random --plot_path ./images/paper/uniform_noise/ --load_net <path_to_saved_models>.pth --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5
python main.py --net $mod --noise_type random_shuffle --plot_path ./images/paper/random_shuffle/ --load_net <path_to_saved_models>.pth --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5


done
done





### Mixup plots (Fig. 13)

python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/mixup --imgs 3,84,18 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/mixup --imgs 19,22,72 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/mixup --imgs 46,43,67 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/mixup --imgs 62,27,3 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/mixup --imgs 111,71,115 --set_seed 0 --range_l 0.5 --range_r 0.5

python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/nomixup --imgs 3,84,18 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/nomixup --imgs 19,22,72 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/nomixup --imgs 46,43,67 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/nomixup --imgs 62,27,3 --set_seed 0 --range_l 0.5 --range_r 0.5
python main.py --net resnet --load_net <path_to_saved_models>.pth --plot_path ./images/paper/nomixup --imgs 111,71,115 --set_seed 0 --range_l 0.5 --range_r 0.5
