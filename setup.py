import setuptools

with open("README.md", 'r') as f:
    long_description = f. read()


setuptools.setup(
        name='rpcv',
        author='Kirill Zinovjev',
        author_email='kzinovjev@gmail.com',
        description="Values and gradients of ring polymer collective variables",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/kzinovjev/rpcv",
        packages=['rpcv'],
        version='0.1'
)
