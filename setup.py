import setuptools

import version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uncertainty-wizard",
    version=version.RELEASE,
    author="Michael Weiss",
    author_email="code@mweiss.ch",
    description="Quick access to uncertainty and confidence of Keras networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/testingautomated-usi/uncertainty_wizard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
