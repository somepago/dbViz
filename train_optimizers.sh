# training models for optimizer experiments
python main.py --net ResNet --set_seed 2 --save_net ResNet_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net ResNet --set_seed 2 --save_net ResNet_sgd_lr0.05 --epochs 200 --bs 64 --lr 0.05
python main.py --net ResNet --set_seed 2 --save_net ResNet_sgdsam_lr0.05 --epochs 200 --bs 64 --lr 0.05 --sam_radius 0.01
python main.py --net ViT4 --set_seed 2 --save_net ViT4_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net ViT4 --set_seed 2 --save_net ViT4_sgd_lr0.02 --epochs 200 --bs 64 --lr 0.02
python main.py --net ViT4 --set_seed 2 --save_net ViT4_sgdsam_lr0.02 --epochs 200 --bs 64 --lr 0.02 --sam_radius 0.01
python main.py --net MLPMixer4 --set_seed 2 --save_net MLPMixer4_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net MLPMixer4 --set_seed 2 --save_net MLPMixer4_sgd_lr0.02 --epochs 200 --bs 64 --lr 0.02
python main.py --net MLPMixer4 --set_seed 2 --save_net MLPMixer4_sgdsam_lr0.02 --epochs 200 --bs 64 --lr 0.02 --sam_radius 0.01
python main.py --net VGG --set_seed 2 --save_net VGG_adam_lr0.002 --epochs 200 --bs 64 --lr 0.002
python main.py --net VGG --set_seed 2 --save_net VGG_sgd_lr0.005 --epochs 200 --bs 64 --lr 0.005
python main.py --net VGG --set_seed 2 --save_net VGG_sgdsam_lr0.005 --epochs 200 --bs 64 --lr 0.005 --sam_radius 0.01
python main.py --net ResNet --set_seed 3 --save_net ResNet_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net ResNet --set_seed 3 --save_net ResNet_sgd_lr0.05 --epochs 200 --bs 64 --lr 0.05
python main.py --net ResNet --set_seed 3 --save_net ResNet_sgdsam_lr0.05 --epochs 200 --bs 64 --lr 0.05 --sam_radius 0.01
python main.py --net ViT4 --set_seed 3 --save_net ViT4_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net ViT4 --set_seed 3 --save_net ViT4_sgd_lr0.02 --epochs 200 --bs 64 --lr 0.02
python main.py --net ViT4 --set_seed 3 --save_net ViT4_sgdsam_lr0.02 --epochs 200 --bs 64 --lr 0.02 --sam_radius 0.01
python main.py --net MLPMixer4 --set_seed 3 --save_net MLPMixer4_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net MLPMixer4 --set_seed 3 --save_net MLPMixer4_sgd_lr0.02 --epochs 200 --bs 64 --lr 0.02
python main.py --net MLPMixer4 --set_seed 3 --save_net MLPMixer4_sgdsam_lr0.02 --epochs 200 --bs 64 --lr 0.02 --sam_radius 0.01
python main.py --net VGG --set_seed 3 --save_net VGG_adam_lr0.002 --epochs 200 --bs 64 --lr 0.002
python main.py --net VGG --set_seed 3 --save_net VGG_sgd_lr0.005 --epochs 200 --bs 64 --lr 0.005
python main.py --net VGG --set_seed 3 --save_net VGG_sgdsam_lr0.005 --epochs 200 --bs 64 --lr 0.005 --sam_radius 0.01
python main.py --net ResNet --set_seed 4 --save_net ResNet_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net ResNet --set_seed 4 --save_net ResNet_sgd_lr0.05 --epochs 200 --bs 64 --lr 0.05
python main.py --net ResNet --set_seed 4 --save_net ResNet_sgdsam_lr0.05 --epochs 200 --bs 64 --lr 0.05 --sam_radius 0.01
python main.py --net ViT4 --set_seed 4 --save_net ViT4_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net ViT4 --set_seed 4 --save_net ViT4_sgd_lr0.02 --epochs 200 --bs 64 --lr 0.02
python main.py --net ViT4 --set_seed 4 --save_net ViT4_sgdsam_lr0.02 --epochs 200 --bs 64 --lr 0.02 --sam_radius 0.01
python main.py --net MLPMixer4 --set_seed 4 --save_net MLPMixer4_adam_lr0.001 --epochs 200 --bs 64 --lr 0.001
python main.py --net MLPMixer4 --set_seed 4 --save_net MLPMixer4_sgd_lr0.02 --epochs 200 --bs 64 --lr 0.02
python main.py --net MLPMixer4 --set_seed 4 --save_net MLPMixer4_sgdsam_lr0.02 --epochs 200 --bs 64 --lr 0.02 --sam_radius 0.01
python main.py --net VGG --set_seed 4 --save_net VGG_adam_lr0.002 --epochs 200 --bs 64 --lr 0.002
python main.py --net VGG --set_seed 4 --save_net VGG_sgd_lr0.005 --epochs 200 --bs 64 --lr 0.005
python main.py --net VGG --set_seed 4 --save_net VGG_sgdsam_lr0.005 --epochs 200 --bs 64 --lr 0.005 --sam_radius 0.01
