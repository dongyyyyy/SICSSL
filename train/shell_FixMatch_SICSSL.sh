# lr=0.0006
lr=0.002
conweight=0.1
gpu=0
seed=0

dataset='cifar10_con'
datasetName='cifar-10'
optim='adam'
Tcon=0.07


flag=0

threshold=0.95
temperature=1.0

batchsize=64
mu=1

imbRatio=100
numMaxL=1000
numMaxU=4000

python ./train/train_FixMatch_SICSSL.py --dataset ${dataset} --num-max ${numMaxL} --num-max-u ${numMaxU} --arch wideresnet --T ${temperature} --threshold ${threshold} --batch-size ${batchsize} --mu ${mu} --lr ${lr} --seed ${seed} --imb-ratio-label ${imbRatio} --imb-ratio-unlabel ${imbRatio} --ema-u 0.99 --gpu-id ${gpu} --optim ${optim} --flag-reverse-LT ${flag} --T-con ${Tcon} --con-weights ${conweight} 
