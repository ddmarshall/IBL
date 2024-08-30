# Release Steps
This document is intended to provide the actions that are needed to prepare a release.

## Pre-Release Actions
When it is determined that all features and fixes have been included into the `develop` branch these steps should be taken.

- [ ] Ensure all tests are passing
- [ ] Ensure all documentation is generated
- [ ] Ensure all examples run

## Release Actions
Once the code is ready for release these steps need to be taken.
Note that the version number is `X.Y.Z` where `X` is the major number, `Y` is the minor number, and `Z` is the patch number.

- [ ] Create release in Git-Flow with name `X.Y.Z`
    - [ ] Update version numbers in
        - [ ] `README.rst`
        - [ ] `pyproject.toml`
        - [ ] `sphinx/source/conf.py`
    - [ ] Run checks that all version interfaces are correct
    - [ ] Change the badges in `README.rst` from `develop` to `main`
    - [ ] Build documentation and put it in place for github with `make github`
    - [ ] Finish the release in Git-Flow

## Post-Release Actions
After a release has been created these steps need to be taken.

- [ ] Start a new feature called `Start_Development` in Git-Flow
    - [ ] Increment the minor number by 1, remove patch number, and append `.dev`
        - [ ] `README.rst`
        - [ ] `pyproject.toml`
        - [ ] `sphinx/source/conf.py`
    - [ ] Run checks that all version interfaces are correct
    - [ ] Change the badges in `README.rst` from `main` to `develop`
    - [ ] Finish the feature in Git-Flow
