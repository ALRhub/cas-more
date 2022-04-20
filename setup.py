from setuptools import setup

setup(
    name='cas-more',
    version='0.1',
    packages=['cas-more'],
    url='',
    license='MIT',
    author='Maximilian Hüttenrauch',
    author_email='max.huettenrauch@gmail.com',
    description='Implementation of CAS-MORE',
    install_requires=['numpy',
                      'scipy',
                      'nlopt',
                      'attrdict',
                      'cma']
)
