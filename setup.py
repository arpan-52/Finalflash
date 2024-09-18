from setuptools import setup, find_packages

setup(
    name="finalflash",  # The package name for PyPI
    version="0.1",
    author="Arpan Pal",
    author_email="arpan522000@gmail.com",
    description="A tool for uGMRT primary beam correction",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "astropy"
    ],
    entry_points={
        'console_scripts': [
            'finalflash=finalflash.beam_corrector:main',
        ],
    },
    url="https://github.com/arpan-52",  # Replace with your repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
