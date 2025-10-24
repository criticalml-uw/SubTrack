from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="subtrackpp-torch",
    version="1.0",
    description="SubTrack++: Gradient Subspace Tracking for Scalable LLM Training",
    packages=["low_rank_torch"],
    install_requires=required,
)
