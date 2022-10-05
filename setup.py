"""Installation script for beaf"""

from setuptools import setup

requirements = [
    "numpy",
    "scipy",
    "spikeinterface",
    "h5py",
    "psutil",
]

# Usage: pip install -e .[dev]
extra_requirements = {
    "dev": [
        "ipykernel",
        "ipython",
        "jupyterlab",
    ]
}

setup(
    author="Jean de Montigny",
    description="BioCam Electrophysiology Analysis Framework",
    extras_require=extra_requirements,
    install_requires=requirements,
    license="",
    name="beaf",
    packages=["beaf"],
    url="https://github.com/JeandeMontigny/beaf.git",
    version="0.1",
)
