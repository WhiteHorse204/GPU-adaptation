from pathlib import Path
import sys
import ultralytics
SOURCE_ROOT = Path(ultralytics.__file__).resolve().parent.parent
assert Path(ultralytics.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'Framework must be installed in the selected environment'
assert ultralytics.__version__ == '8.3.63', 'Wrong framework environment'
