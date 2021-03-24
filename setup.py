from setuptools import setup, find_packages

setup(
    name='bottleneck-transformer-flax',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Bottleneck Transformer - Flax',
    author='Niccolò Zanichelli',
    author_email='niccolo.zanichelli@gmail.com',
    url='https://github.com/NZ99/bottleneck-transformer-flax',
    keywords=[
        'artificial intelligence', 'deep learning', 'transformer',
        'image classification'
    ],
    install_requires=['jax', 'jaxlib', 'flax', 'einops>=0.3'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)