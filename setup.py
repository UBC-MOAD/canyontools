from setuptools import setup


setup(
    name='CanyonTools',
    version='1.0',
    description='Useful functions to analyze canyon experiments',
    author='Karina Ramos Musalem',
    author_email='kramosmu@eos.ubc.ca',
    install_requires=[
        "matplotlib",
        "scipy",
        "netCDF4",
    ],
    packages=['canyon_tools'],
)

 
