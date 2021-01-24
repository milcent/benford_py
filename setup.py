''' Setup Pyt for benford's module'''
from os import path
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README-pypi.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='benford_py',
      version='0.3.3',
      description='A library for testing data sets with Bendford\'s Law',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/milcent/benford_py',
      download_url='https://github.com/milcent/benford_py/archive/v0.3.3.tar.gz',
      author='Marcel Milcent',
      author_email='marcelmilcent@gmail.com',
      license='GPLv3.0',
      packages=['benford'],
      install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
      ],
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Intended Audience :: Other Audience',
          'Topic :: Office/Business :: Financial :: Accounting',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],)
