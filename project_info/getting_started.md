# Getting Started Developing
This document is a brief overview on how to get started working on this library.

## Get Sources
The project is hosted on [GitHub](https://github.com/ddmarshall/IBL), and the code can be forked from there.
The [project documentation](https://ddmarshall.github.io/IBL/index.html) is another resource that can be helpful in getting started using and modifying this project.

## Setup Development Environments
It is intended for development to be done in virtual environments.
The following commands can be used to with a virtual environment.

  * Create virtual environment:  `python3.{VERSION} -m venv ./venvs/{VERSION INFO}/`
  * Activate virtual environment: `source ./venvs/{VERSION INFO}/bin/activate`
  * Deactivate virtual environment: `deactivate`

## Create Editable Install of Project
Once in a virtual environment, an editable install of the project should be done.
The current command is:
> `pip install -e .[examples,tests]`

At this point the project is ready for use.

## Software Development Practices
The overall structure of the project is

  * Main directory: general project items
  * `docs`: location of the generated documentation
  * `examples`: example usage and other instructional items
  * `ibl`: code for the library and associate models
  * `sphinx`: files used to generate the documentation
  * `tests`: unit tests for code
  * `venvs`: directory containing virtual environments

The commands described below should be run from main directory of the project unless otherwise noted.

### Git Flow
The code is developed using the standard [git flow](https://nvie.com/posts/a-successful-git-branching-model/) workflow (also explained [here](https://jeffkreeftmeijer.com/git-flow/)).
The principle concept behind git flow is that there are two branches to a project.
One branch is the `main` branch.
This is the source of all new releases.
The other branch is the `develop` branch.
This branch is where new features and the code for future upgrades is developed.

### Test Driven Development
The main development practice used is a [test driven development](https://en.wikipedia.org/wiki/Test-driven_development) practice.
New features are developed with the development of unit tests that the new code needs to pass.
These tests are located within the `tests` directory and may reside is a sub-directory depending on the code organization.

As the feature evolves, the code (and corresponding tests) are refactored to better achieve the goal of the new feature.
This is results in updated tests along with new tests.
These tests are the used for regression testing as new features are developed.
The following command can be used to run all tests:
> `python -m unittest discover --start-directory tests`

### Static Code Analysis
To perform static code analysis, there are two tools that are in use, `pylint` and `pycodestyle`.
The following command can be used for `pycodestyle`
> `pycodestyle --statistics ibl tests examples`

The results from this are a general, high-level analysis of the code and typically identifies issues that need to be addressed.

Similarly, the command for `pylint` is
> `pylint ibl tests examples`

This provides a more in-depth analysis of the quality of the code.
The aim is to address as many of these messages as possible.
However, it is sometimes unavoidable to adhere to some of the pythonic code readability guidelines.
These situations should be avoided whenever possible.

### Code Coverage
The aim of the testing is to have tests covering 100% of the library code, but that is sometimes difficult to achieve.
Especially when there are errors that are hard to replicate but should be handled. Dadddss is 
To collect the default code coverage data for the project use the following command:
> `python -m coverage run`

Since each individual test can be executed on its own, the following command can be used to collect code coverage data from a particular test:
> `python -m coverage run {py-file including path}`

To get a text based report of the code coverage use the following command:
> `python -m coverage report -m`

This produces a text based coverage report in the console.

To get an HTML version of the coverage report use the following command:
> `python -m coverage html`

This generates a report in HTML format that can be found in `htmlcov/index.html`.

### Type Checking
The aim for this project is to have type-hinting throughout the code to help developers know what types of data are expected and to limit the the introduction of bugs caused by passing invalid data types.
The static testing tool used to check this is `mypy`.
To perform type checking for the entire project use the following command:
> `mypy`

If only a specific directory or file is to be checked that the path (and filename, if desired) can be passed as an argument, such as
> `mypy ./examples`

### Continuous-Integration Testing
Several tests/checks are run on commits to the GitHub repository, via GitHub workflow, to alert the developers of any problems that might be introduced by the commit.
To check that these checks are going pass, the automated testing framework tool `tox` is used.
This will test a variety of python versions (only the versions that are already installed locally and that the project has been setup to test).
It will also go through the code analysis, code coverage, and type checking steps mentioned above.

### Documentation
The documentation for this project is generated using `sphinx`.
The actual documentation files reside in the `docs` directory.
The documentation is first built in the `sphinx` directory.
To build the documentation use the following commands:
```
    cd ./sphinx
    make html
```

Note that the HTML documentation is built in the `./sphinx/build/html/` directory.
Once the documentation changes are ready to be integrated into the project documentation, the following command can be used
```
    cd ./sphinx
    make github
```
This will build the documentation and then copy to necessary files to the `docs` directory.
