envname=Pendulum-v1
savefolder=Pendulum-v0
device=0
seed=0
for algo in MaTD3
do
	CUDA_VISIBLE_DEVICES=${device} python run_cont.py --getfig True --envname=${envname} --algo=${algo} --suffix=${seed} --off-render --seed=${seed} --savefolder=${savefolder}
done