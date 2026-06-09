"""Compatibility entry point for standard SAMD evaluation.

This module delegates to ``evaluation.inference_samd`` so flags such as
``--profile-fusion`` stay defined in one place.
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("evaluation.inference_samd", run_name="__main__")
