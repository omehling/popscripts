from setuptools import setup

setup(name='pop_diagnostics',
      version='0.1',
      description='Tools for analyzing POP2 ocean model output',
      url='https://github.com/omehling/popscripts',
      author='Oliver Mehling, Reyk Börner',
      author_email='r.borner@uu.nl',
      license='MIT',
      packages=['pop_diagnostics'],
      install_requires=[
          'xarray', 'gsw-xarray',
      ],
      zip_safe=False)