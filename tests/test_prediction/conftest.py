"""Shared test fixtures for prediction module tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def pred_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with minimal prediction test data."""
    # ID map
    id_maps_dir = tmp_path / "id_maps"
    id_maps_dir.mkdir()
    id_map_csv = (
        "code,first_name,second_name,web_name,"
        "16-17,17-18,18-19,19-20,20-21,21-22,22-23,23-24,24-25,"
        "fbref,understat,transfermarkt\n"
        "100,Harry,Kane,Kane,,,,10,11,12,13,14,15,abc123,1234,9999\n"
        "200,Mohamed,Salah,Salah,,,,20,21,22,23,24,25,def456,5678,8888\n"
        "300,Virgil,van Dijk,Van Dijk,,,,30,31,32,33,34,35,ghi789,9012,7777\n"
        "400,Alisson,Becker,Alisson,,,,40,41,42,43,44,45,jkl012,3456,6666\n"
    )
    (id_maps_dir / "master_id_map.csv").write_text(id_map_csv, encoding="utf-8")

    return tmp_path
