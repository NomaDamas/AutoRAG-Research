"""Type definitions."""

from io import BytesIO
from pathlib import Path

ImageType = str | bytes | Path | BytesIO
