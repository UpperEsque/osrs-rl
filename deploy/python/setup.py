from setuptools import setup, find_packages

setup(
    name="osrs_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "requests",
    ],
    extras_require={
        "train": [
            "stable-baselines3",
            "tensorboard",
            "torch",
        ],
        "wandb": [
            "wandb",
        ],
    },
)
