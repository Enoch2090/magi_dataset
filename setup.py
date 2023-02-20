#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

from importlib.metadata import entry_points
import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'magi_dataset'
DESCRIPTION = 'Convenient access to massive corpus of GitHub repositories'
URL = 'https://github.com/Enoch2090/magi_dataset'
EMAIL = 'ycgu2090@gmail.com'
AUTHOR = 'Enoch2090'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = '1.0.6'

REQUIRED = [
    'numpy>=1.15.4',
    'pandas>=1.2.0',
    'spacy',
    'scipy',
    'beautifulsoup4',
    'deep_translator',
    'hn',
    'langdetect',
    'lxml',
    'Markdown',
    'networkx',
    'PyGithub',
    'python_hn',
    'requests',
    'setuptools',
    'tqdm'
]

EXTRA_REQUIRE = {
    'elasticsearch': [
        'elasticsearch'
    ]
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    extras_require=EXTRA_REQUIRE,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    package_dir={'': './'},
    include_package_data=True,
    package_data={
            'magi_dataset.data':['patterns.txt'],
            '': ['./requirements.txt']
    },
    license='GPLv3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
    entry_points = {
        'console_scripts': [
            'magi_dataset = magi_dataset:entry',
        ]
    }
)