import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "server"))


def main() -> None:
    import fastapi
    import numpy
    import pydantic
    import scipy
    import sklearn
    import torch

    import_output = StringIO()
    with redirect_stdout(import_output):
        import server.server as server_module

    result = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "versions": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "fastapi": fastapi.__version__,
            "pydantic": pydantic.__version__,
        },
        "server_app_title": server_module.app.title,
        "model_loaded_on_import": server_module.model is not None,
        "server_import_stdout": import_output.getvalue().strip().splitlines(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
