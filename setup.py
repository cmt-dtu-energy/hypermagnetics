from setuptools import find_packages, setup

setup(
    name="hypermagnetics",
    version="0.2.0",
    description="Scalable magnetic source-to-field inference with hypernetworks",
    author="Berian James",
    author_email="1518788+berianjames@users.noreply.github.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "equinox>=0.11.4",
        "fmm2dpy",
        "h5py",
        "ipython",
        "ipykernel",
        "jax[cuda12]==0.6.1",
        "jaxtyping>=0.2.28",
        "matplotlib",
        "ml-dtypes>=0.2.0",
        "numpy==1.25.2",
        "optax>=0.2.2",
        "pytest",
        "prettytable",
        "scienceplots",
        "wandb",
    ],
    python_requires=">=3.9,<3.12",
)
