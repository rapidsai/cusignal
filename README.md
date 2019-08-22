# RAPIDS Standard Repo Template

This is repo is the standard RAPIDS repo with the following items to make all RAPIDS repos consistent:

- GitHub File Templates
  - Issue templates
  - PR template
- GitHub Repo Templates
  - Issue/PR labels
  - Project tracking and release board templates
- Files
  - `CHANGELOG.md` skeleton
  - `CONTRIBUTING.md` skeleton
  - `LICENSE` file with Apache 2.0 License
  - `README.md` skeleton


## Usage for new RAPIDS repos

1. Clone this repo
2. Find/replace all in the clone of `___PROJECT___` and replace with the name of the new library
3. Inspect files to make sure all replacements work and update text as needed
4. Customize issue/PR templates to fit the repo
5. Update `CHANGELOG.md` with next release version, see [changelog format](https://rapidsai.github.io/devdocs/docs/resources/changelog/) for more info
6. Add developer documentation to the end of the `CONTRIBUTING.md` that is project specific and useful for developers contributing to the project
    - The goal here is to keep the `README.md` light, so the development/debugging information should go in `CONTRIBUTING.md`
7. Complete `README.md` with project description, quick start, install, and contribution information
8. Remove everything above the RAPIDS logo below from the `README.md`
9. Check `LICENSE` file is correct
10. Change git origin to point to new repo and push
11. Alert OPS team to copy labels and project boards to new repo

## Usage for existing RAPIDS repos

1. Follow the steps 1-9 above, but add the files to your existing repo and merge
2. Alert OPS team to copy labels and project boards to new repo

## Useful docs to review

- Issue triage & release planning
  - [Issue triage process with GitHub projects](https://rapidsai.github.io/devdocs/docs/releases/triage/)
  - [Release planning with GitHub projects](https://rapidsai.github.io/devdocs/docs/releases/planning/)
- Code release process
  - [Hotfix process](https://rapidsai.github.io/devdocs/docs/releases/hotfix/)
  - [Release process](https://rapidsai.github.io/devdocs/docs/releases/process/)
- Code contributions
  - [Code contribution guide](https://rapidsai.github.io/devdocs/docs/contributing/code/)
  - [Filing issues](https://rapidsai.github.io/devdocs/docs/contributing/issues/)
  - [Filing PRs](https://rapidsai.github.io/devdocs/docs/contributing/prs/)
  - [Code of conduct](https://rapidsai.github.io/devdocs/docs/resources/conduct/)
- Development process
  - [Git branching and merging methodology](https://rapidsai.github.io/devdocs/docs/resources/git/)
  - [Versions and tags](https://rapidsai.github.io/devdocs/docs/resources/versions/)
  - [Changelog format](https://rapidsai.github.io/devdocs/docs/resources/changelog/)
  - [Style guide](https://rapidsai.github.io/devdocs/docs/resources/style/)
  - [Labels](https://rapidsai.github.io/devdocs/docs/maintainers/labels/)

---

# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;___PROJECT___</div>

The [RAPIDS](https://rapids.ai) ___PROJECT___ ..._insert project description_...

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/___PROJECT___/blob/master/README.md) ensure you are on the `master` branch.

## Quick Start

## Install ___PROJECT___

### Conda

### Docker

## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/___PROJECT___/blob/master/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.
