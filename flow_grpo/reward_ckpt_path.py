import os

CKPT_PATH = os.environ.get(
    "REWARD_CKPTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reward_ckpts"),
)
