import setuptools

with open('README.md') as f:
    readme = f.read()

setuptools.setup(
        name='baycomp_plotting',
        version='1.1.1',
        description='This package provides some extra functionality for plotting baycomp\'s posteriors.',
        long_description=readme,
        long_description_content_type='text/markdown',
        author='Mario Juez-Gil',
        author_email='mariojg@ubu.es',
        url='https://github.com/mjuez/baycomp_plotting',
        download_url='https://github.com/mjuez/baycomp_plotting/archive/v1_1_1.tar.gz',
        license='GPLv3',
        install_requires=[
            'matplotlib==3.3.2',
            'numpy==1.19.1',
            'iteround==1.0.2',
            'scipy==1.5.3'
        ],
        packages=setuptools.find_packages(),
        include_package_data=True,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'
        ]
)
