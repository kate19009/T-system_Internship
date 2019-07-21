from setuptools import setup, find_packages

setup(
      name='ControlsNN',
      version='1.0',
      description='ControlsNN is a library containing a neural network that is capable of identifying various controls',
      author='Ekaterina Zabrodina, Anna Dudina, Nikita Chikin, Ilya Malov',
      license='MIT',
      packages=find_packages(),

      install_requires=[
          'opencv-python', 'keras', 'tensorflow', 'numpy', 'matplotlib', 'scikit-learn'
      ],

      include_package_data=True,
      data_files=[
          ('', ['ControlsNN/vggnet.model']),
          ('', ['ControlsNN/vggnet_lb.pickle']),
       ],
)
