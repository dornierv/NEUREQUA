from setuptools import setup

setup(
    name='neurequa',
    version='0.1.1.5',    
    description='Python package to monitor micro-recording quality in humans',
    url='https://github.com/dornierv/NEUREQUA',
    author='Vincent Dornier',
    author_email='vincent.dornier@cnrs.fr',
    license='MIT',
    packages=['neurequa'],
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
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.12',
    ],
)
