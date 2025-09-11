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
Once you have made edits to a `.ui` file in Qt Creator and saved
your changes, you need to update the python representation of the UI.
1. Update the corresponding `.py` file by running
    ```
    pyside6-uic accusleepy/gui/<filename>.ui -o accusleepy/gui/<filename>.py
    ```
    where `<filename>` is either `primary_window` or `viewer_window`.

> [!NOTE]
> If for some reason that doesn't work, on Windows you can
> locate `uic.exe` in your PySide6 installation and run
> ```
> <path_to_your_uic>.exe -g python accusleepy\gui\<filename>.ui -o accusleepy\gui\<filename>.py
> ```
2. `uic` does not create some necessary imports in the modified
   `.py` file, so you will need to add them back.
   Open the file and add the following import statement:
   ```
   import accusleepy.gui.resources_rc  # noqa F401
   ```
   If the file already contains the line `import resources_rc`,
   replace it with the one above.
   If you updated `primary_window.py`, you also need to add:
   ```
   from accusleepy.gui.mplwidget import MplWidget
   ```

### Updating the resources file
If you want to modify the resources available to the GUI
(e.g., icon image files), you can edit the `resources.qrc`
file using Qt Creator. You then need to update the python
representation by running

```
pyside6-rcc accusleepy/gui/resources.qrc -o accusleepy/gui/resources_rc.py
```
