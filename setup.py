# from setuptools import setup, find_packages  
# setup(name = 'pymodch', packages = find_packages())

from setuptools import setup, find_packages

setup(
    name="pymodch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Aquí puedes listar las dependencias de tu proyecto, por ejemplo:
        'numpy',
        'scipy',
        'matplotlib',
        'emcee',
        'tqdm',
        'dynesty'
    ],
    # extras_require={
    #     # Aquí puedes listar dependencias opcionales, por ejemplo:
    #     # 'dev': [
    #     #     'pytest',
    #     # ],
    # },
)
