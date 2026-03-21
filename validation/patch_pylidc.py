path = r'C:\Users\Dan\PycharmProjects\GammaMetric\.venv\Lib\site-packages\pylidc\__init__.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    "import pkg_resources as _pr",
    "import importlib.metadata as _pr"
)

content = content.replace(
    "_dbpath  = _pr.resource_filename('pylidc', 'pylidc.sqlite')",
    "import os as _os; _dbpath = _os.path.join(_os.path.dirname(__file__), 'pylidc.sqlite')"
)

with open(path, 'w') as f:
    f.write(content)

print('Patched successfully')
