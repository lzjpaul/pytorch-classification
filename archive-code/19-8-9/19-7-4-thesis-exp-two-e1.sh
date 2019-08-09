# each block 24 hours
# ncre1
/hdd1/zhaojing/anaconda3-cuda-10/bin/python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 0.0 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 1 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-9.log
/hdd1/zhaojing/anaconda3-cuda-10/bin/python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 1 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-10.log
/hdd1/zhaojing/anaconda3-cuda-10/bin/python gm_prior_cifar_tune_resnet_cifar_a_0_alpha_1_b_1.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/regresnet-110 --gmnum 4 --gmuptfreq 100 --paramuptfreq 50 --gpu-id 1 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-11.log
/hdd1/zhaojing/anaconda3-cuda-10/bin/python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 0.0 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 1 | tee -a 19-7-4-results/19-7-4-thesis-exp-two-12.log
# ncrf1