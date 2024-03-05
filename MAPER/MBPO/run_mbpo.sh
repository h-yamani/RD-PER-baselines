# Example: MBPO with MaPER on Hopper-v2
replay_type='MaPER'
env_name=Hopper-v2
num_epoch=300
CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --env_name=${env_name} --replay_type=${replay_type} --suffix=0 --num_epoch=${num_epoch} --pred_hidden_size=400 --num_train_repeat=20 --rollout_min_length=1 --rollout_max_length=25 --rollout_min_epoch=20 --rollout_max_epoch=300

# Example: MBPO without MaPER on Hopper-v2
replay_type='RANDOM'
env_name=Hopper-v2
num_epoch=300
CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --env_name=${env_name} --replay_type=${replay_type} --suffix=0 --num_epoch=${num_epoch} --pred_hidden_size=400 --num_train_repeat=20 --rollout_min_length=1 --rollout_max_length=25 --rollout_min_epoch=20 --rollout_max_epoch=300
