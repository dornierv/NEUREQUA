from setuptools import setup, find_packages

setup(
    name='neurequa',

    version='0.1.1.19',    

    description='Python package to monitor micro-recording quality in humans',
    url='https://github.com/dornierv/NEUREQUA',
    author='Vincent Dornier',
    author_email='vincent.dornier@cnrs.fr',
    license='MIT',
    packages=find_packages(),
    install_requires=['neo',
                      'numpy',  
                      'scipy',
                      'matplotlib',
                      'seaborn',
                      'mne',
                      'pandas',
                      'openpyxl'                   
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'MIT',  
        'Programming Language :: Python :: 3.12',
    ],
)
