td3_exp_hyper_params = {
    "ACTOR_SIZE": [400, 300],
    "CRITIC_SIZE": [400, 300],
    "GAMMA": 0.98,
    "TAU": 0.005,
    "BUFFER_SIZE": 1000000,
    "BATCH_SIZE": 32,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-3,
    "WEIGHT_DECAY": 0.0,
    "POLICY_UPDATE_FREQ": 2,
    "EXPLORATION_NOISE": 0.1,
    "TARGET_POLICY_NOISE": 0.2,
    "TARGET_POLICY_NOISE_CLIP": 0.5,
    "INITIAL_RANDOM_ACTION": 1000,
    "TOTAL_STEPS": 1000000,
    "PER_ALPHA": 0.7,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "MULTIPLE_LEARN": 64,
    "TRAIN_FREQ": 64,
}
td3_test_hyper_params = {
    "ACTOR_SIZE": [400, 300],
    "CRITIC_SIZE": [400, 300],
    "GAMMA": 0.98,
    "TAU": 0.005,
    "BUFFER_SIZE": 1000000,
    "BATCH_SIZE": 32,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-3,
    "WEIGHT_DECAY": 0.0,
    "POLICY_UPDATE_FREQ": 2,
    "EXPLORATION_NOISE": 0.1,
    "TARGET_POLICY_NOISE": 0.2,
    "TARGET_POLICY_NOISE_CLIP": 0.5,
    "INITIAL_RANDOM_ACTION": 100,
    "TOTAL_STEPS": 200,
    "PER_ALPHA": 0.7,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "MULTIPLE_LEARN": 64,
    "TRAIN_FREQ": 64,
}
sac_exp_hyper_params = {
    "ACTOR_SIZE": [400, 300],
    "CRITIC_SIZE": [400, 300],
    "GAMMA": 0.98,
    "TAU": 0.02,
    "LR_ACTOR": 7.3e-4,
    "LR_QF1": 7.3e-4,
    "LR_QF2": 7.3e-4,
    "LR_ENTROPY": 7.3e-4,
    "BUFFER_SIZE": 1000000,
    "BATCH_SIZE": 32,
    "AUTO_ENTROPY_TUNING": True,
    "INITIAL_RANDOM_ACTION": 1000,
    "TOTAL_STEPS": 1000000,
    "MULTIPLE_LEARN": 64,
    "TRAIN_FREQ": 64,
    "PER_ALPHA": 0.7,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "WEIGHT_DECAY": 0.0
}
sac_test_hyper_params = {
    "ACTOR_SIZE": [400, 300],
    "CRITIC_SIZE": [400, 300],
    "GAMMA": 0.98,
    "TAU": 0.02,
    "LR_ACTOR": 7.3e-4,
    "LR_QF1": 7.3e-4,
    "LR_QF2": 7.3e-4,
    "LR_ENTROPY": 7.3e-4,
    "BUFFER_SIZE": 1000000,
    "BATCH_SIZE": 32,
    "AUTO_ENTROPY_TUNING": True,
    "INITIAL_RANDOM_ACTION": 256,
    "TOTAL_STEPS": 512,
    "MULTIPLE_LEARN": 64,
    "TRAIN_FREQ": 64,
    "PER_ALPHA": 0.7,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "WEIGHT_DECAY": 0.0
}
