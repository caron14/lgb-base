# Base Class of LightGBM

**


# 環境設定
## venv

Build a virtual environment in the local environment.
The environment is built by installing with the following three requirements files.
- requirements-dev.txt
- requirements-test.txt
- requirements.txt

```sh
python -m venv (virtual environment name)
source venv/bin/activate
```

Install shared libraries for each project.

```sh
pip install -r requirements-dev.txt
```

The `requirements-dev.txt` includes the following libraries:

- [isort](https://github.com/PyCQA/isort)
  - It rearranges the order of `import` and `from * import *` according to a uniform rule.
- [black](https://github.com/psf/black)
  - A tool that automatically formats Python code. It formats the code of all development members according to a uniform rule.
- [flake8](https://github.com/PyCQA/flake8)
  - A Python Linter, it points out Python writing that does not conform to [PEP8](https://pep8-ja.readthedocs.io/ja/latest/)

Install libraries for testing.

```sh
pip install -r requirements-test.txt
```

The `requirements-dev.txt` includes the following libraries:

- [mypy](https://github.com/python/mypy)
  - A variable type checker.
- [pytest](https://github.com/pytest-dev/pytest)
  - A library for test scripts.

Install the necessary dependencies for development.

```sh
pip install -r requirements.txt
```


# Getting started
