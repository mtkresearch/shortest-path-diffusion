from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion", "nonunif_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch>=1.11",
                    "torchvision", "tqdm", "scipy", "pyyaml", "tensorboard"],
)
