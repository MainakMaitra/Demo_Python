# === Classic Notebook (works with nbextensions) ===
notebook==6.5.6
jupyter_contrib_nbextensions==0.5.1
jupyter_nbextensions_configurator==0.6.4

# === Jupyter Core Packages (Python 3.12 compatible) ===
jupyter_client==8.6.0
jupyter_core==5.5.0
jupyter_server==1.24.0

# === JupyterLab (3.x, compatible with notebook 6.5.6) ===
jupyterlab==3.6.6
jupyterlab_server==2.24.0
jupyterlab_widgets==1.1.7

# === Kernel + Execution Support ===
ipykernel==6.29.4
ipython==8.12.3
traitlets==5.9.0

# === Notebook File Handling ===
nbconvert==6.5.4
nbformat==5.9.2

# === Widgets and UI Add-ons ===
ipywidgets==8.1.2
widgetsnbextension==4.0.10

# === Compatibility Fixes ===
packaging==24.0


python -m jupyter_contrib_nbextensions.application install --user
https://www.lfd.uci.edu/~gohlke/pythonlibs/#jupyter-contrib-core
