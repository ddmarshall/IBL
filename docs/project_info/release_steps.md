# Release Steps
This document is intented to provide the actions that are needed to prepare a release.

## Pre-Release Actions
When it is determined that all features and fixes have been included into the `develop` branch these steps should be taken.

* Ensure that all required tests are passing.

## Release Actions
Once the code is ready for release these steps need to be taken.

* Create release in Git-Flow
    * The release name should be vX.Y.Z where X is the major number, Y is the minor number, and Z is the patch number.
* Update version number.
    * These file need to be updated:
        * README.md
        * setup.py
* Check that all version interfaces are correct.
* Finish the release in Git-Flow.

## Post-Release Actions
After a release has been created these steps need to be taken.

* Start a new feature in Git-Flow.
* Increment the patch number by 1 and append -develop to the version string such as `1.2.3-develop`.
    * See above for files that need to be updated.
* Check that all version interfaces are correct.
* Finish the feature in Git-Flow.
