from distutils.core import setup
import re

with open('README.md') as fp:
    long_description=fp.read()


def find_version():
    with open('textTkAPI/__init__.py') as fp:
        for line in fp:
            match = re.search(r"__version__\s*=\s*'([^']+)'", line)
            if match:
                return match.group(1)
    assert False, 'cannot find version'


setup(
    name='TextTkAPI',
    version=find_version(),
    author='Matthew Sunner',
    author_email='matt@mattsunner.com',
    license='LICENSE.txt',
    description='NLP Pre-Processing REST API.',
    long_description=long_description,
)