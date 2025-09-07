# Developer guide

## Getting started
This project uses poetry for dependency management and
pre-commit hooks to maintain a consistent style.
To set up your development environment:
1. Install [poetry](https://python-poetry.org/docs/#installation)
2. Clone the repo
3. Set up your virtual environment
4. Run `poetry install` to install dependencies
5. Run `pre-commit install` to set up the git hook scripts


## Editing the GUI
Start by installing
[Qt Creator](https://doc.qt.io/qtcreator/). This software lets you
interactively modify the `.ui` files for the primary interface
and the manual scoring interface.

### Exporting your changes
Once you have made edits in Qt Creator and saved your changes,
you need to update the python representation of the UI by running

```pyside6-uic filename.ui -o filename.py```

where `filename` is either "primary_window" or "viewer_window".
>If for some reason that doesn't work, on Windows you can
>locate `uic.exe` in your PySide6 installation and run
>```<path_to_your_uic.exe> -g python filename.ui -o filename.py```

You then need to open the modified `.py` file and change the last
import statement from

```import resources_rc```

to

```import accusleepy.gui.resources_rc  # noqa F401```

### Updating the resources file
If you want to modify the resources available to the GUI
(e.g., icon image files), you can edit the `resources.qrc`
file using Qt Creator. You then need to update the python
representation by running

```pyside6-rcc resources.qrc -o resources_rc.py```
