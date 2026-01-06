from setuptools import setup, find_packages

with open("../requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="consistency-models-radar",
    version="0.1.0",
    description="4D Radar Diffusion Models using Consistency Models and EDM",
    packages=find_packages(),
    py_modules=["cm", "evaluations"],
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
