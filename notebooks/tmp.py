# %%
from pathlib import Path
startup_path = Path.cwd()
project_path = startup_path.parent

# %%
from kedro.extras.extensions.ipython import reload_kedro
reload_kedro(project_path)

# %%
catalog

# %%
