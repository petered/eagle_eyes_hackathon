from setuptools import setup, find_packages

setup(
    name='eagle_eyes_hackathon',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',  # assuming numpy is a dependency
        'opencv-python',  # assuming opencv-python is a dependency
        'artemis-ml @ git+https://github.com/petered/artemis.git@develop',
        # 'pandastable @ git+https://github.com/petered/pandastable.git',
        'dataclass-serialization @ git+https://github.com/petered/python-dataclasses-serialization.git@add_setup',
        'exif'
    ],
)
