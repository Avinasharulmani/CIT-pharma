from pathlib import Path


REACT_INDEX_HTML = (Path(__file__).resolve().parent.parent / "scratch_served.html").read_text(encoding="utf-8")
