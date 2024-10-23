from setuptools import setup, find_packages

setup(
    name="finalflash",
    version="0.3.3",
    author="Arpan Pal",
    author_email="arpan522000@gmail.com",
    description="A tool for uGMRT primary beam correction",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/arpan-52/Finalflash",  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "astropy",
    ],
    entry_points={
        'console_scripts': [
            'finalflash=finalflash.beam_corrector:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
