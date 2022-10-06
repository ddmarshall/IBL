# Release Steps
This document is intented to provide the actions that are needed to prepare a release.

## Pre-Release Actions
When it is determined that all features and fixes have been included into the `develop` branch these steps should be taken.

- [ ] Ensure all tests are passing
- [ ] Ensure all documentation is generated
- [ ] Ensure all examples run

## Release Actions
Once the code is ready for release these steps need to be taken.
Note that the version number is `X.Y.Z` where `X` is the major number, `Y` is the minor number, and `Z` is the patch number.

- [ ] Create release in Git-Flow with name `vX.Y.Z`
    - [ ] Update version numbers in
        - [ ] `README.rst`
        - [ ] `stepup.py`
    - [ ] Run checks that all version interfaces are correct
    - [ ] Finish the release in Git-Flow

## Post-Release Actions
After a release has been created these steps need to be taken.

- [ ] Start a new feature in Git-Flow
    - [ ] Increment the patch number by 1 and append `-develop`
    - [ ] Run checks that all version interfaces are correct
    - [ ] Finish the feature in Git-Flow
