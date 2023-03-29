from setuptools import setup
  
setup(
    name='exotic_jedi',
    version='0.1',
    description='Exoplanet Timeseries Characterisation - JWST Extraction and Diagnostic Investigator',
    author='Lilli Alderson',
    author_email='lili.alderson@bristol.ac.uk',
    packages=['exotic_jedi'],
    install_requires=[
        'numpy',
        'jwst',
        'scipy',
        'astropy',
        'matplotlib',
        'batman-package',
        'emcee',
        'corner',
        'arviz',
        'tqdm',
        'xarray',
        'pandas',
        'exotic_ld'

    ],
)