from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    legal = f.read()

setup(
    name='postproc',
    version='0.1.2',
    description='My PhD post-processing tools',
    long_description=readme,
    author='Bernat Font Garcia',
    author_email='b.fontgarcia@soton.ac.uk',
    keywords=['post-processing, cfd, fortran, machine learning, python'],
    url='https://github.com/j-massey/postproc',
    license=legal,
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['numpy'],
    install_requires=['numpy', 'matplotlib', 'torch']
)
