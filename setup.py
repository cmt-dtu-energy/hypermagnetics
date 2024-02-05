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
        "optax",
        "equinox>=0.11.2",
        "wandb",
        "jaxtyping>=0.2.23",
        # "jax" is intentionally left out here
    ],
    extras_require={
        "dev": [
            "pytest",
            "matplotlib",
            "ipython",
            "ipykernel",
            "scienceplots",
            # "mayavi", # Install via conda
            # "PyQt5",
        ],
        "metal": ["jax-metal==0.0.5"],
    },
    python_requires=">=3.9",
)
