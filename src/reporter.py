"""
src/reporter.py
===============
Experiment output manager for Jupyter notebooks.
Creates a timestamped directory per experiment and saves
figures, tables, and images into it with a single method call.

------------------------------------------------------------------------------
Classes
------------------------------------------------------------------------------

Reporter
    Initializes an output directory at <report_dir>/<experiment>__<timestamp>
    and exposes save methods for all common output types.

    __init__(report_dir: str | Path, experiment: str = "experiment")
        Input:  report_dir : str | Path  — root reports directory
                experiment : str         — experiment name used as folder prefix
        Output: —
        Side effect: creates output directory, prints its path

    --------------------------------------------------------------------------

    save_figure(fig, name: str, fmt: str = "html") -> Path
        Save a Plotly or Matplotlib figure.
        Plotly figures default to .html; pass fmt="png"/"pdf" for static export.
        Matplotlib figures default to .png (html is remapped to png).

        Input:  fig  : go.Figure | go.FigureWidget | matplotlib Figure
                name : str   — output filename stem
                fmt  : str   — "html" | "png" | "pdf" | "svg"
        Output: Path  — path to saved file

    --------------------------------------------------------------------------

    save_table(df: pd.DataFrame, name: str, fmt: str = "csv") -> Path | None
        Save a DataFrame as CSV, HTML, or both.

        Input:  df   : pd.DataFrame
                name : str  — output filename stem
                fmt  : str  — "csv" | "html" | "both"
        Output: Path  — path to saved file (None if fmt="both")
        Raises: ValueError for unsupported fmt

    --------------------------------------------------------------------------

    save_image(path_or_array, name: str, fmt: str = "png") -> Path
        Save an image from a file path or a numpy array.

        Input:  path_or_array : str | Path | np.ndarray
                name          : str  — output filename stem
                fmt           : str  — "png" | "jpg" | "jpeg"
        Output: Path  — path to saved file
        Raises: TypeError if input is not a path or numpy array

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

_safe_name(name: str) -> str
    Replace any character that is not alphanumeric, underscore, or hyphen
    with an underscore to produce a safe filename stem.

    Input:  name : str
    Output: str  — sanitized filename stem
"""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


class Reporter:

    def __init__(self, report_dir, experiment: str = "experiment"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.out_dir = Path(report_dir) / f"{experiment}__{timestamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reporter ready: {self.out_dir}")

    def _safe_name(self, name: str) -> str:
        return re.sub(r"[^\w\-]", "_", name)

    def save_figure(self, fig, name: str, fmt: str = "html"):
        name = self._safe_name(name)

        # plotly Figure or FigureWidget
        if isinstance(fig, (go.Figure, go.FigureWidget)):
            if fmt == "html":
                path = self.out_dir / f"{name}.html"
                pio.write_html(fig, str(path))
            else:
                path = self.out_dir / f"{name}.{fmt}"
                pio.write_image(fig, str(path))

        # matplotlib Figure
        else:
            fmt = fmt if fmt != "html" else "png"
            path = self.out_dir / f"{name}.{fmt}"
            fig.savefig(str(path), bbox_inches="tight", dpi=150)

        print(f"Saved: {path}")
        return path

    def save_table(self, df: pd.DataFrame, name: str, fmt: str = "csv"):
        name = self._safe_name(name)

        if fmt == "csv":
            path = self.out_dir / f"{name}.csv"
            df.to_csv(path, index=False)

        elif fmt == "html":
            path = self.out_dir / f"{name}.html"
            df.to_html(path, index=False)

        elif fmt == "both":
            self.save_table(df, name, fmt="csv")
            self.save_table(df, name, fmt="html")
            return

        else:
            raise ValueError(f"Unsupported fmt: {fmt}. Use 'csv', 'html', or 'both'.")

        print(f"Saved: {path}")
        return path

    def save_image(self, path_or_array, name: str, fmt: str = "png"):
        import numpy as np
        from PIL import Image

        name = self._safe_name(name)
        path = self.out_dir / f"{name}.{fmt}"

        if isinstance(path_or_array, (str, Path)):
            img = Image.open(path_or_array).convert("RGB")
        elif isinstance(path_or_array, np.ndarray):
            img = Image.fromarray(path_or_array.astype("uint8"))
        else:
            raise TypeError("Expected file path or numpy array.")

        img.save(str(path))
        print(f"Saved: {path}")
        return path