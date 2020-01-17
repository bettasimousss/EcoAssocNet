import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecoassocnet", 
    version="0.0.1",
    author="Sara SI-MOUSSI",
    author_email="sara.simoussi@gmail.com",
    description="A package for inferring biotic associations from species co-distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SoccoCMOS/EcoAssocNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)