""" Atlas
Some description here...
"""

from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


# -----
# Setup
# -----
setup(
    name="atlas",
    description="some description",
    long_description=readme(),
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    url="https://github.com/rileyhickman/atlas",
    author="Riley Hickman",
    author_email="riley.hickman@mail.utoronto.ca",
    # license='XXX',
    packages=["atlas"],
    package_dir={"": "src"},
    zip_safe=False,
    tests_require=["pytest"],
    install_requires=["numpy", "pandas", "rich", "deap", "pymoo",
                      'sobol-seq', "pyDOE", "botorch", 
                      "matter-chimera", "matter-golem"],
    python_requires=">=3.6",
)
