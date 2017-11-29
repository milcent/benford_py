from setuptools import setup

setup(name='bendford_py',
      version='0.1.0',
      description='A library for testing Bendford\'s Law data sets',
	  long_description=readme(),
      url='https://github.com/milcent/benford_py',
      author='Marcel Milcent',
      author_email='marcelmilcent@gmail.com',
      license='GPLv3.0',
      packages=['bendford'],
      install_requires=[
      	'pandas',
      	'numpy',
      	'matplotlib',
      ],
      zip_safe=False)