import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest import mock

if "pandas" not in sys.modules:
    pd_stub = types.ModuleType("pandas")
    pd_stub.DataFrame = object
    sys.modules["pandas"] = pd_stub

if "sqlalchemy" not in sys.modules:
    sa_stub = types.ModuleType("sqlalchemy")
    sa_stub.create_engine = lambda *a, **kw: None
    sys.modules["sqlalchemy"] = sa_stub

if "vanna" not in sys.modules:
    vanna_stub = types.ModuleType("vanna")
    vanna_stub.openai = types.ModuleType("openai")
    vanna_stub.chromadb = types.ModuleType("chromadb")

    class _VN:
        def __init__(self, *a, **kw):
            pass
        def train(self):
            pass
        def generate_sql(self, q):
            return "SQL"
        def run_sql(self, sql, engine=None):
            return mock.MagicMock(head=lambda x: mock.MagicMock(to_string=lambda index=False: "df"))
        def generate_chart(self, df=None, sql=None):
            return mock.MagicMock(write_html=lambda p: None)

    class OpenAI:
        def __init__(self, *a, **kw):
            pass

    class Chroma:
        def __init__(self, *a, **kw):
            pass

    vanna_stub.Vanna = _VN
    vanna_stub.openai.OpenAI = OpenAI
    vanna_stub.chromadb.Chroma = Chroma
    sys.modules["vanna"] = vanna_stub
    sys.modules["vanna.openai"] = vanna_stub.openai
    sys.modules["vanna.chromadb"] = vanna_stub.chromadb

import os
import pytest

from vanna_adapter.cli import main


@mock.patch('vanna_adapter.cli.Vanna')
@mock.patch('vanna_adapter.cli.Chroma')
@mock.patch('vanna_adapter.cli.OpenAI')
@mock.patch('vanna_adapter.cli.create_engine')
def test_missing_env(mock_engine, mock_openai, mock_chroma, mock_vanna, capsys):
    os.environ.pop('LLM_API_KEY', None)
    os.environ.pop('DB_URL', None)
    os.environ.pop('VANNA_QUERY', None)
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err
    assert 'error' in err

def _setup_env(tmp_path):
    os.environ['LLM_API_KEY'] = 'k'
    os.environ['DB_URL'] = 'db'
    os.environ['VANNA_QUERY'] = 'question'
    os.environ.pop('LLM_MODEL', None)
    os.chdir(tmp_path)


@mock.patch('vanna_adapter.cli.Vanna')
@mock.patch('vanna_adapter.cli.Chroma')
@mock.patch('vanna_adapter.cli.OpenAI')
@mock.patch('vanna_adapter.cli.create_engine')
def test_training_and_query(mock_engine, mock_openai, mock_chroma, mock_vanna, tmp_path, capsys):
    _setup_env(tmp_path)
    instance = mock_vanna.return_value
    instance.generate_sql.return_value = 'SQL'
    instance.run_sql.return_value = mock.MagicMock(head=lambda x: mock.MagicMock(to_string=lambda index=False: 'df'))
    instance.generate_chart.return_value = mock.MagicMock(write_html=lambda p: None)
    main()
    out = capsys.readouterr().out
    assert 'SQL' in out
    assert Path('.trained').exists()


@mock.patch('vanna_adapter.cli.Vanna')
@mock.patch('vanna_adapter.cli.Chroma')
@mock.patch('vanna_adapter.cli.OpenAI')
@mock.patch('vanna_adapter.cli.create_engine')
def test_update_command(mock_engine, mock_openai, mock_chroma, mock_vanna, tmp_path, capsys):
    _setup_env(tmp_path)
    os.environ['VANNA_QUERY'] = '/update'
    main()
    out = capsys.readouterr().out
    assert 'Model updated' in out
