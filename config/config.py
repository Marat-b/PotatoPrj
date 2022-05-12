import os
from distutils.util import strtobool

from dotenv import load_dotenv

load_dotenv()
DEEPSORT = str(os.getenv("deepsort_checkpo"))
DETECTRON2 = str(os.getenv("detectron2_check"))
MAX_DIST = float(os.getenv("max_dist"))
DISPLAY = bool(strtobool(os.getenv("display")))
DISPLAY_WIDTH = int(os.getenv("display_width"))
DISPLAY_HEIGHT = int(os.getenv("display_height"))
USE_CUDA = bool(strtobool(os.getenv("use_cuda")))
