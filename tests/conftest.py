import pytest
from unittest.mock import patch, mock_open


@pytest.fixture
def open_file():
    with patch("builtins.open", new_callable=mock_open) as m:
        yield m
