





#sdd_dkd
#warmup20 1122
#
#
#python train_origin.py --cfg configs/cifar100/sdd_dkd/vgg13_vgg8.yaml --gpu 3 --M [1,2]  74.31
#python train_origin.py --cfg configs/cifar100/sdd_dkd/vgg13_vgg8.yaml --gpu 3 --M [1,2]  74.48
#
#python train_origin.py --cfg configs/cifar100/sdd_dkd/vgg13_mv2.yaml --gpu 3 --M [1,2,4] 70.61
#python train_origin.py --cfg configs/cifar100/sdd_dkd/vgg13_mv2.yaml --gpu 3 --M [1,2,4] 70.02
#
#
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res50_mv2.yaml --gpu 3 --M [1,2,4]  71.61
#python train_origin.py --cfg configs/cifar100/sdd_dkd/res50_mv2.yaml --gpu 3 --M [1,2,4]  71.41






#gl_kd
#warmup 1
#
#python train_origin.py --cfg configs/cifar100/sdd_kd/vgg13_vgg8.yaml --gpu 3 --M [1,2]   73.54
#python train_origin.py --cfg configs/cifar100/sdd_kd/vgg13_vgg8.yaml --gpu 3 --M [1,2]  73.40
#
#python train_origin.py --cfg configs/cifar100/sdd_kd/vgg13_mv2.yaml --gpu 3 --M [1,2,4]  68.87
#python train_origin.py --cfg configs/cifar100/sdd_kd/vgg13_mv2.yaml --gpu 3 --M [1,2,4]   68.48
#
#
#python train_origin.py --cfg configs/cifar100/sdd_kd/res50_mv2.yaml --gpu 1 --M [1,2,4]  69.51
#python train_origin.py --cfg configs/cifar100/sdd_kd/res50_mv2.yaml --gpu 1 --M [1,2,4]  69.54





#gl_nkd


#warmup 20
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_vgg8.yaml --gpu 3 --M [1,2] 74.00
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_vgg8.yaml --gpu 3 --M [1,2] 73.77
#
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_mv2.yaml --gpu 3 --M [1,2,4] 69.50
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_mv2.yaml --gpu 3 --M [1,2,4] 68.52
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_mv2.yaml --gpu 0 --M [1,2] 69.79
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_mv2.yaml --gpu 0 --M [1,2] 69.66
#
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_mv2.yaml --gpu 1 --M [1,2] 69.11
#python train_origin.py --cfg configs/cifar100/sdd_nkd/vgg13_mv2.yaml --gpu 1 --M [1,2] 69.22
#
#
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res50_mv2.yaml --gpu 3 --M [1,2,4] 70.37
#python train_origin.py --cfg configs/cifar100/sdd_nkd/res50_mv2.yaml --gpu 3 --M [1,2,4] 70.40

