from setuptools import setup

setup(
    name='Genre Detect',
    version='0.0.1',
    author='Jakob Rossi',
    url='https://github.com/jmrossi98/genre_detect',
    python_requires="~=3.9",
    install_requires=[
        'tensorflow==2.9',
        'keras==2.9',
        'tensorflow-datasets==4.0.0',
        'tensorflow-estimator>=2.6.0',
        'scikit-learn>=1.0.2',
        'protobuf>=3.20.0',
        'librosa==0.8.0',
        'matplotlib>=3.7.1',
        'numpy>=1.19.2',
        'ffmpeg==1.4'
    ]
)
