from setuptools import setup

setup(
    name='NeuReQua',
    version='0.1.0',    
    description='Python package to monitor micro-recording quality in humans',
    url='https://github.com/dornierv/NEUREQUA',
    author='Vincent Dornier',
    author_email='vincent.dornier@cnrs.fr',
    license='MIT',
    packages=['NEUREQUA'],
    install_requires=['neo',
                      'numpy',  
                      'scipy',
                      'matplotlib',
                      'seaborn',
                      'mne',
                      'pandas',
                      'dhn_med_py==1.1.3'                   
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3.12',
    ],
)
