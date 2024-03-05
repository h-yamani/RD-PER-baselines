# MfRL_Disc with MaCL
rt='MaPER' # rt (replay_type) ='PER' if the original prioritized experience replay is wanted.
game='ms_pacman'
seed='0'
CUDA_VISIBLE_DEVICES=0 python main.py --enable-cudnn --target-update 2000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --replay-type=${rt} --game=${game} --seed=${seed}

# MfRL_Disc without MaCL
 rt='PER'
 game='ms_pacman'
 seed='0'
 CUDA_VISIBLE_DEVICES=0 python main.py --enable-cudnn --target-update 2000 --memory-capacity 1000000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --replay-type=${rt} --game=${game} --seed=${seed}
