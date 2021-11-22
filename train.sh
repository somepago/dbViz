# # declare -a ModelTypes=( 'ResNet' 'VGG' 'GoogLeNet' 'DenseNet' 'MobileNet' 'LeNet' 'FCNet' 'ViT4' 'ViT_pt' 'MLPMixer4')
# declare -a ModelTypes=('fcnet' 'vit' 'resnet' ) #'densenet' 'vgg' 'vit' 'resnet' )

# for mod in "${ModelTypes[@]}";

# do

# for seed in 1 2 3 4 5 6 7 8 9 10 
# do

# python main.py --net $mod --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/gowthami_models/${mod}_seed1.pth --plot_path ./images/temp/ --imgs 30,72,-100 --set_seed $seed --range_l 0.5 --range_r 0.5


# # sbatch ~/scavenger.sh main.py --net $mod --set_seed $seed --save_net ${mod}_${seed}cifar10 --imgs 500,5000,1600 --resolution 500 --active_log --epochs 200  --lr 0.001 --bs 64 # --plot_animation --bs 64

# # sbatch ~/scavenger.sh main.py --net $mod --load_net ./saved_models/naive/${mod}_${seed}cifar10.pth --plot_path images/best/traindata/5665_899_9000/${mod}_${seed}cifar10 --imgs 5665,899,9000,-100

# # python main.py --net $mod --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/gowthami_models/${mod}_seed${seed}.pth --plot_path ./images/paper/ --imgs 3,7,6 --set_seed $seed --range_l 0.5 --range_r 0.5
# # python main.py --net $mod --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/gowthami_models/${mod}_seed${seed}.pth --plot_path ./images/paper/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

# # python main.py --net $mod --load_net ./saved_models/naive/1/${mod}_1cifar10.pth --plot_path ./images/paper/ --imgs 17,40,1 --set_seed 1 --range_l 0.5 --range_r 0.5
# # python main.py --net $mod --load_net ./saved_models/naive/1/${mod}_1cifar10.pth --plot_path ./images/paper/ --imgs 13,62,35 --set_seed 1 --range_l 0.5 --range_r 0.5

# #215,889,6228 - 0 4 4
# #245,899,826 - 3 5 7
# #500,5000,1600 - 4 7 8
# done
# done


# sbatch ~/scavenger.sh main.py --net MLPMixer_pt --set_seed 5 --save_net MLPMixer_pt_5cifar10 --imgs 500,5000,1600 --resolution 500 --active_log --epochs 200 --lr 0.001






# for a in [0-9]*.png; do
#     mv $a `printf %03d.%s ${a%.*} ${a##*.}`
# done

# python temp.py --net ResNet --load_net ./saved_models/naive/1/ResNet_1cifar10.pth --plot_path ./images/temp/ --imgs 199,1297,6995 --set_seed 1 --range_l 0.5 --range_r 0.5

# for wid in 20 30

# do

# for seed in 0 1 2 
# do

# python main.py --net WideResNet --widen_factor $wid --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/widenet_models_new/wideresnet_${wid}/wideresnet_${wid}_seed${seed}.pth --plot_path ./images/paper/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

# done
# done


# for seed in 0 1 2 3 4

# do
# sbatch ~/high.sh main.py --net MLPMixer4 --set_seed $seed --save_net ${mod}_${seed}cifar10 --imgs 500,5000,1600 --resolution 500 --active_log --epochs 100 --lr 0.01

# done




# for seed in 0 #2 3
# do

# declare -a ModelTypes=( 'resnet' 'fcnet' 'vit' 'vgg' 'densenet')

# for mod in "${ModelTypes[@]}";

# do


# sbatch ~/high.sh fragmentation.py --net $mod --epochs 1000 --active_log --resolution 50 --set_seed $seed --range_l 0.1 --range_r 0.1 --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/gowthami_models/${mod}_seed${seed}.pth

# done
# done


# python fragmentation.py --net MLPMixer4 --epochs 1000 --active_log --resolution 50 --set_seed 0 --range_l 0.1 --range_r 0.1 --load_net ./saved_models/mlpmixer/_0cifar10.pth

# for wid in 10 #20 30

# do

# for seed in 1 2 3 4 5 6 7 8 9
# do

# # sbatch ~/high.sh fragmentation.py --net WideResNet --widen_factor $wid --epochs 1000 --active_log --resolution 50 --set_seed $seed --range_l 0.1 --range_r 0.1 --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/widenet_models_new/wideresnet_${wid}/wideresnet_${wid}_seed${seed}.pth
# python main.py --net WideResNet --widen_factor $wid --noise_type random_shuffle --plot_path ./images/paper/shuffled/ --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/widenet_models_new/wideresnet_${wid}/wideresnet_${wid}_seed1.pth --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5

# done
# done

# declare -a ModelTypes=( 'ViT_pt' ) #'ResNet' 'VGG' 'ViT4' ) #'densenet' 'vgg' 'vit' 'resnet' )

# for mod in "${ModelTypes[@]}";

# do
# declare -a OMTypes=('sgd' 'adam' 'sgdsam')
# for opti in "${OMTypes[@]}";
# do
# for seed in 1 
# do
# python main.py --net $mod --load_net /cmlscratch/pchiang/decision_bound/saved_models/naive/1/${mod}_${opti}.pth --plot_path ./images/paper/optimizer/${opti}/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

# # python main.py --net $mod --load_net /cmlscratch/pchiang/decision_bound/saved_models/naive/1/${mod}_${opti}_seed${seed}.pth --plot_path ./images/paper/optimizer/${opti}/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

# done
# done
# done

declare -a ModelTypes=('resnet') # 'VGG') #'densenet' 'vgg' 'vit' 'resnet' )
# declare -a ModelTypes=('MLPMixer4' ) #'resnet' 'fcnet' 'vit' 'densenet' 'vgg')
for mod in "${ModelTypes[@]}";

do

for seed in 0 #2 3 4 5 6 7 8 9 
do

# python main.py --net $mod --noise_type uniform_random --plot_path ./images/paper/uniform_noise/ --load_net /cmlscratch/gowthami/deci_bounds/decision_bound/saved_models/mlpmixer/_1cifar10.pth --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5
# python main.py --net $mod --noise_type random_shuffle --plot_path ./images/paper/shuffled/ --load_net /cmlscratch/gowthami/deci_bounds/decision_bound/saved_models/mlpmixer/_1cifar10.pth --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5
# python main.py --net $mod --noise_type two_random_shuffle --plot_path ./images/paper/two_shuffled/ --load_net /cmlscratch/gowthami/deci_bounds/decision_bound/saved_models/mlpmixer/_1cifar10.pth --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5


# python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 46,43,67 --set_seed $seed --range_l 0.5 --range_r 0.5
# python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 111,71,115 --set_seed $seed --range_l 0.5 --range_r 0.5

# python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/mixup/${mod}_cifar10.pth --plot_path ./images/paper/mixup --imgs 46,43,67 --set_seed $seed --range_l 0.5 --range_r 0.5
# python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/mixup/${mod}_cifar10.pth --plot_path ./images/paper/mixup --imgs 111,71,115 --set_seed $seed --range_l 0.5 --range_r 0.5



python main.py --net $mod --noise_type random_shuffle  --load_net /cmlscratch/lfowl/decision_bound/saved_models/paper_runs/gowthami_models/${mod}_seed1.pth --plot_path ./images/paper/shuffled/ --imgs 10,20,30 --set_seed $seed --range_l 0.5 --range_r 0.5

# # python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

# # python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 29,51,1 --set_seed $seed --range_l 0.5 --range_r 0.5
# # python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 62,27,3 --set_seed $seed --range_l 0.5 --range_r 0.5
# # python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 19,22,72 --set_seed $seed --range_l 0.5 --range_r 0.5
# # python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 116,14,12 --set_seed $seed --range_l 0.5 --range_r 0.5
# # python main.py --net $mod --load_net /cmlscratch/bansal01/summer_2021/decision_bound/saved_models/naive/1/${mod}_naive_1_cifar10.pth --plot_path ./images/paper/nomixup --imgs 3,84,18 --set_seed $seed --range_l 0.5 --range_r 0.5

# # python main.py --net $mod --load_net /cmlscratch/pchiang/decision_bound/saved_models/naive/1/${mod}_${opti}_seed${seed}.pth --plot_path ./images/paper/optimizer/${opti}/ --imgs 30,72,42 --set_seed $seed --range_l 0.5 --range_r 0.5

done
done