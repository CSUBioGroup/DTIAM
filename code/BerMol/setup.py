import os
from setuptools import setup, find_packages

def get_version():
    directory = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(directory, 'bermol', '__init__.py')
    with open(init_file) as f:
        for line in f:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")

with open('requirements.txt', 'r') as reqs:
    requirements = reqs.read().split()

setup(
    name='bermol',
    version=get_version(),
    description="Bidirectional Encoder Representations for Molecular",
    author="Zhangli Lu",
    author_email='luzhangli.csu@gmail.com',
    url='https://github.com/ZhangliLu/BerMol',
    packages=find_packages(),
    license=None,
    keywords=['Molecular', 'BerMol', 'Deep Learning', 'Pytorch'],
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        'Topic :: Scientific/Engineering :: Bioinformatics'
    ],
)
