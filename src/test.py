# Load and examine
import json

from utils import files

from src.paths import get_paths

paths = get_paths()

pair = files.load_json_line_by_line(paths.FILE_TRAINING_PAIRS)
print(json.dumps(pair, indent=2))
