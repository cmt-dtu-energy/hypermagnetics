from setuptools import find_packages, setup

setup(
    name="hypermagnetics",
    version="0.1.0",
    description="Scalable magnetic source-to-field inference with hypernetworks",
    author="Berian James",
    author_email="1518788+berianjames@users.noreply.github.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "optax>=0.2.2",
        "equinox>=0.11.4",
        "wandb",
        "jaxtyping>=0.2.28",
        # "jax" is intentionally left out here
    ],
    extras_require={
        "dev": [
            "pytest",
            "matplotlib",
            "ipython",
            "ipykernel",
            "scienceplots",
            # "pyvista", # Install via conda
        ],
        "metal": ["jax-metal==0.0.6"],
    },
    python_requires=">=3.9",
)
