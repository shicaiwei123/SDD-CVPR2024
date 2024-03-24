python train_origin.py --cfg configs/cub200/sdd_kd/vgg13_vgg8.yaml --gpu 2 --M [1] 63.2
#python train_origin.py --cfg configs/cub200/sdd_kd/vgg13_vgg8.yaml --gpu 2 --M [1,2,4] 68.83

python train_origin.py --cfg configs/cub200/sdd_dkd/vgg13_vgg8.yaml --gpu 2 --M [1] 66.3
#python train_origin.py --cfg configs/cub200/sdd_dkd/vgg13_vgg8.yaml --gpu 2 --M [1,2,4] 67.93

python train_origin.py --cfg configs/cub200/sdd_nkd/vgg13_vgg8.yaml --gpu 2 --M [1] 67.5
python train_origin.py --cfg configs/cub200/sdd_nkd/vgg13_vgg8.yaml --gpu 2 --M [1,2,4] 69.3