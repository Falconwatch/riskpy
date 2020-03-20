from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='riskpy',
    version='0.0.3',
    description="Risk-manager\'s pack",
    url='http://google.com',
    author='Igor Shcherbakov',
    author_email='iashcherbakov@yandex.ru',
    license='MIT',
    packages=find_packages(),
    keywords=['modeling', 'risks', 'statistics'],  # arbitrary keywords
    zip_safe=False,
    test_suite='tests',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'statsmodels',
        'seaborn',
        'scipy'

    ]
)

