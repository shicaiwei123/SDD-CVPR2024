





#warmup20 1122
#
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 78.13
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 77.94
#
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res50_vgg8.yaml --gpu 0 --M [1,2,4]  75.79
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res50_vgg8.yaml --gpu 0 --M [1,2,4]  75.50
#
#
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res32x4_mv2.yaml --gpu 0 --M [1,2,4]  70.06
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res32x4_mv2.yaml --gpu 0 --M [1,2,4]  70.18









#SD_KD

#warmup 1 1122

#python train_origin.py --cfg configs/cifar100/sdd_kd/res50_shuv1.yaml --gpu 0 --M [1,2,4]   76.84
#python train_origin.py --cfg configs/cifar100/sdd_kd/res50_shuv1.yaml --gpu 0 --M [1,2,4]  76.68
#
#
#python train_origin.py --cfg configs/cifar100/sdd_kd/res50_vgg8.yaml --gpu 0 --M [1,2,4]  74.36
#python train_origin.py --cfg configs/cifar100/sdd_kd/res50_vgg8.yaml --gpu 0 --M [1,2,4]  75.04
#
#
#python train_origin.py --cfg configs/cifar100/sdd_kd/res32x4_mv2.yaml --gpu 1 --M [1,2,4] 69.01
#python train_origin.py --cfg configs/cifar100/sdd_kd/res32x4_mv2.yaml --gpu 1 --M [1,2,4]  68.84 68.55







#SDD_NKD
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 76.77
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 76.91
#
#
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res50_vgg8.yaml --gpu 0 --M [1,2]  75.20
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res50_vgg8.yaml --gpu 0 --M [1,2]  74.58
#
#
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res32x4_mv2.yaml --gpu 0 --M [1,2,4] 69.56
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res32x4_mv2.yaml --gpu 0 --M [1,2,4] 69.68