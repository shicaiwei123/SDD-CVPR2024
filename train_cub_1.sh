

python train_origin.py --cfg configs/cub200/sdd_kd/res50_shuv1.yaml --gpu 0 --M [1] 56.81
python train_origin.py --cfg configs/cub200/sdd_kd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 60.14

python train_origin.py --cfg configs/cub200/sdd_kd/res32x4_shuv1.yaml --gpu 0 --M [1] 62.15
python train_origin.py --cfg configs/cub200/sdd_kd/res32x4_shuv1.yaml --gpu 0 --M [1,2,4] 65.65


python train_origin.py --cfg configs/cub200/sdd_dkd/res50_shuv1.yaml --gpu 0 --M [1] 59.33
python train_origin.py --cfg configs/cub200/sdd_dkd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 60.42

python train_origin.py --cfg configs/cub200/sdd_dkd/res32x4_shuv1.yaml --gpu 0 --M [1]
#python train_origin.py --cfg configs/cub200/sdd_dkd/res32x4_shuv1.yaml --gpu 0 --M [1,2,4] 60.20


#python train_origin.py --cfg configs/cub200/sdd_nkd/res50_shuv1.yaml --gpu 0 --M [1]  60.32
#python train_origin.py --cfg configs/cub200/sdd_nkd/res50_shuv1.yaml --gpu 0 --M [1,2,4] 61.30

python train_origin.py --cfg configs/cub200/sdd_nkd/res32x4_shuv1.yaml --gpu 0 --M [1]
python train_origin.py --cfg configs/cub200/sdd_nkd/res32x4_shuv1.yaml --gpu 0 --M [1,2,4]
