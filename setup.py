import setuptools

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="sport",
    version="1.0",
    author="Yan Zeng",
    author_email="quantsummaries@gmail.com",
    description="SPORT (Scalable Portfolio Optimization Research Tool)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)