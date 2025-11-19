# vanna_adapter/cli.py (Solución definitiva para v0.7.9)
import json
import os
import sys
from pathlib import Path
import argparse
import io
import hashlib
from contextlib import redirect_stdout
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import openai
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url

# --- IMPORTACIONES CORRECTAS Y VERIFICADAS PARA v0.7.9 ---
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class Vanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Clase combinada de Vanna para RAG local con OpenAI y ChromaDB."""

    def log(self, message: str, title: str = "Info") -> None:
        # Redirigir logs a stderr
        print(f"{title}: {message}", file=sys.stderr)
    def __init__(self, client=None, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)

    def get_ddl(self, engine):
        """Extrae DDL de todas las tablas usando SQLAlchemy."""
        from sqlalchemy import MetaData
        from sqlalchemy.schema import CreateTable

        metadata = MetaData()
        metadata.reflect(bind=engine)
        stmts = []
        for table in metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(bind=engine))
            stmts.append(ddl)
        return "\n\n".join(stmts)

    def train(self, ddl: str = None, **kwargs) -> bool:
        """Almacena las declaraciones DDL en el vector store (entrenamiento RAG) y espera a la indexación."""
        if ddl:
            statements = ddl.split("\n\n") if isinstance(ddl, str) else ddl
            expected = len(statements)
            for stmt in statements:
                stmt = stmt.strip()
                if stmt:
                    self.add_ddl(stmt)
        return True

    
# --------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"


class _silence_stdout:
    """Context manager to suppress stdout from noisy libraries."""

    def __enter__(self):
        self._buf = io.StringIO()
        self._redir = redirect_stdout(self._buf)
        self._redir.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._redir.__exit__(exc_type, exc, tb)
        # drop captured output


def _format_markdown(result: dict) -> str:
    """Return a markdown string built from the result dictionary."""
    sections = []
    sql = result.get("sql")
    if sql:
        # sql may already contain fenced block markers
        if not sql.startswith("```"):
            sql = f"```sql\n{sql}\n```"
        sections.append(f"### SQL\n\n{sql}")
    df_text = result.get("dataframe")
    if df_text:
        sections.append(f"### Dataframe\n{df_text}")
    summary = result.get("summary")
    if summary:
        sections.append(f"### Summary\n{summary}")
    questions = result.get("questions")
    if questions:
        q_text = "\n".join(questions)
        sections.append(f"### Questions\n{q_text}")
    message = result.get("message")
    if message:
        sections.append(f"### Message\n{message}")
    return "\n\n".join(sections)



def _error_exit(message: str) -> None:
    """Imprime un error JSON en stderr y sale con código 1."""
    payload = json.dumps({"error": message}, separators=(",", ":"))
    sys.stderr.write(payload + "\n")
    sys.exit(1)


def _load_env() -> dict:
    """Carga y valida las variables de entorno."""
    env = {
        "api_key": os.getenv("LLM_API_KEY"),
        "db_url": os.getenv("DB_URL"),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    }
    missing = [k for k, v in env.items() if k in {"api_key", "db_url"} and not v]
    if missing:
        _error_exit(f"Missing environment variables: {', '.join(missing)}")
    return env


def _build_vanna(api_key: str, model: str) -> Vanna:
    """Configura e instancia Vanna componiendo sus partes."""
    # Instanciar cliente OpenAI con endpoint personalizado
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("LLM_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    api_type = os.getenv("OPENAI_API_TYPE")
    if api_type:
        client_kwargs["api_type"] = api_type
    api_version = os.getenv("OPENAI_API_VERSION")
    if api_version:
        client_kwargs["api_version"] = api_version
    client = OpenAI(**client_kwargs)
    config = {"model": model}
    return Vanna(client=client, config=config)


def _load_cache_metadata(path: Path) -> dict | None:
    """Lee el archivo de metadatos del caché si existe."""
    try:
        with path.open("r", encoding="utf-8") as meta_file:
            return json.load(meta_file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _headers_match(metadata: dict, headers) -> bool:
    """Compara cabeceras HTTP con los metadatos almacenados."""
    cached_etag = metadata.get("etag")
    remote_etag = headers.get("ETag")
    if cached_etag and remote_etag and remote_etag == cached_etag:
        return True
    cached_last_modified = metadata.get("last_modified")
    remote_last_modified = headers.get("Last-Modified")
    if cached_last_modified and remote_last_modified and remote_last_modified == cached_last_modified:
        return True
    cached_size = metadata.get("size")
    remote_size = headers.get("Content-Length")
    if cached_size is not None and remote_size is not None:
        try:
            if int(remote_size) == cached_size:
                return True
        except ValueError:
            pass
    return False


def _revalidate_cached_sqlite(url: str, metadata: dict) -> tuple[bool, object | None]:
    """Verifica remotamente si el archivo cacheado sigue vigente.

    Retorna (True, None) si podemos reutilizarlo, (False, response) si hay que descargar.
    """
    etag = metadata.get("etag")
    last_modified = metadata.get("last_modified")
    if etag or last_modified:
        request = Request(url, method="GET")
        if etag:
            request.add_header("If-None-Match", etag)
        if last_modified:
            request.add_header("If-Modified-Since", last_modified)
        try:
            response = urlopen(request)
            return False, response
        except HTTPError as exc:
            if exc.code == 304:
                return True, None
            if exc.code == 404:
                _error_exit("Remote SQLite URL not found (404)")
            raise
        except URLError:
            sys.stderr.write("Unable to validate cached SQLite (conditional GET failed). Redownloading.\n")
            return False, None

    request = Request(url, method="HEAD")
    try:
        with urlopen(request) as head_response:
            headers = head_response.info()
    except HTTPError as exc:
        if exc.code == 404:
            _error_exit("Remote SQLite URL not found (404)")
        if exc.code in {405, 501}:
            sys.stderr.write("Remote server does not support HEAD; forcing SQLite redownload.\n")
            return False, None
        raise
    except URLError as exc:
        sys.stderr.write(f"Unable to validate cached SQLite ({exc}). Redownloading.\n")
        return False, None

    if _headers_match(metadata, headers):
        return True, None
    return False, None


def _maybe_materialize_sqlite(db_url: str, work_dir: Path) -> str:
    """Descarga un SQLite remoto (http/https) y devuelve un URL sqlite:/// local."""
    if not db_url:
        return db_url
    raw_url = db_url.strip()
    lower = raw_url.lower()
    if not lower.startswith(("http://", "https://")):
        return raw_url
    parsed = urlparse(raw_url)
    filename = Path(parsed.path).name
    if not filename:
        _error_exit("Remote SQLite URL must include a filename")
    if not filename.endswith((".sqlite", ".sqlite3", ".db")):
        _error_exit("Remote SQLite URL must end with .sqlite/.sqlite3/.db")
    target_dir = work_dir / "databases"
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()[:12]
    local_path = (target_dir / f"{slug}-{filename}").resolve()
    metadata_path = local_path.with_suffix(local_path.suffix + ".json")
    metadata = _load_cache_metadata(metadata_path)
    download_response = None
    if metadata and metadata.get("source_url") == raw_url and local_path.exists():
        cached_size = metadata.get("size")
        try:
            local_size = local_path.stat().st_size
        except OSError:
            local_size = None
        if cached_size is not None and local_size is not None and cached_size != local_size:
            sys.stderr.write(f'Size mismatch for cached SQLite "{local_path}", re-downloading\n')
        else:
            reuse, response = _revalidate_cached_sqlite(raw_url, metadata)
            if reuse:
                sys.stderr.write(f'Reusing cached SQLite database at "{local_path}"\n')
                return f"sqlite:///{local_path}"
            if response is not None:
                download_response = response
    tmp_path = local_path.with_name(local_path.name + ".tmp")
    digest = hashlib.sha256()
    response_headers = None
    try:
        if download_response is None:
            request = Request(raw_url, method="GET")
            download_response = urlopen(request)
        with download_response as response, tmp_path.open("wb") as output:
            response_headers = response.info()
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
    except (HTTPError, URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, HTTPError) and exc.code == 304:
            sys.stderr.write(f'Reusing cached SQLite database at "{local_path}"\n')
            return f"sqlite:///{local_path}"
        _error_exit(f"Failed to download SQLite database from {raw_url}: {exc}")
    tmp_path.replace(local_path)
    sha256 = digest.hexdigest()
    metadata = {
        "source_url": raw_url,
        "sha256": sha256,
        "size": local_path.stat().st_size,
        "filename": filename,
    }
    headers = response_headers or {}
    for header_name, meta_key in (("ETag", "etag"), ("Last-Modified", "last_modified")):
        header_value = headers.get(header_name)
        if header_value:
            metadata[meta_key] = header_value
    metadata_path.write_text(json.dumps(metadata, indent=2))
    sys.stderr.write(f'Downloaded SQLite database to "{local_path}"\n')
    return f"sqlite:///{local_path}"


def _connect_database(vn: Vanna, url) -> None:
    """Conecta a Vanna con la base de datos indicada por el URL."""
    backend = (url.drivername or "").split("+", 1)[0]
    if backend in {"postgresql", "postgres"}:
        vn.connect_to_postgres(
            host=url.host,
            dbname=url.database,
            user=url.username,
            password=url.password,
            port=url.port,
        )
        return
    if backend == "sqlite":
        connector = getattr(vn, "connect_to_sqlite", None)
        if connector is None:
            _error_exit("SQLite connections are not supported by this Vanna build")
        db_path = url.database
        if not db_path:
            _error_exit("SQLite URL must include a database path")
        if db_path != ":memory:":
            db_path = str(Path(db_path).expanduser())
        connector(path=db_path)
        return
    _error_exit(f"Unsupported database dialect: {backend}")


def _train(vn: Vanna, engine, train_flag: Path) -> None:
    """Obtiene DDL y entrena a Vanna."""
    ddl = vn.get_ddl(engine=engine)
    vn.train(ddl=ddl)
    train_flag.touch()


def _generate_chart(vn: Vanna, df: pd.DataFrame, sql: str, question: str, work_dir: Path) -> str | None:
    """Genera un gráfico (best-effort)."""
    try:
        plotly_code = vn.generate_plotly_code(question=question, sql=sql, df=df)
        fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
        output = work_dir / "plot.html"
        fig.write_html(output)
        return str(output)
    except Exception:
        return None


def main() -> None:
    """Punto de entrada principal del CLI."""
    try:
        parser = argparse.ArgumentParser(description="CLI para Vanna Adapter")
        parser.add_argument("--query", required=True, help="Query message to process")
        parser.add_argument("--graph", action="store_true", help="Visualize graph")
        parser.add_argument("--questions", action="store_true", help="Generate followup questions")
        parser.add_argument("--global-questions", action="store_true", help="Generate global questions")
        args = parser.parse_args()

        work_dir = Path.cwd()
        train_flag = work_dir / ".trained"

        env = _load_env()
        env["db_url"] = _maybe_materialize_sqlite(env["db_url"], work_dir)
        os.environ["DB_URL"] = env["db_url"]
        vn = _build_vanna(env["api_key"], env["model"])
        url = make_url(env["db_url"])
        _connect_database(vn, url)
        engine = create_engine(env["db_url"])
        
        query = args.query.strip()

        if query.lower() == "/update":
            train_flag.unlink(missing_ok=True)

        if not train_flag.exists():
            try:
                with _silence_stdout():
                    _train(vn, engine, train_flag)
                if query.lower() == "/update":
                    sys.stdout.write(_format_markdown({"message": "Model updated"}) + "\n")
                    return
            except Exception as e:
                _error_exit(f"Error training model: {e}")



        # Ejecutar flujo RAG oficial de Vanna
        do_graph = args.graph
        try:
            with _silence_stdout():
                sql, df, fig = vn.ask(
                    question=query,
                    print_results=False,
                    auto_train=False,
                    visualize=do_graph,
                    allow_llm_to_see_data=True
                )
        except Exception as e:
            _error_exit(str(e))

        text = df.head(1000).to_markdown(index=False) if df is not None else ""
        plot_path = None
        if fig is not None:
            output = work_dir / "plot.html"
            fig.write_html(output)
            plot_path = str(output)

        # Generar respuesta en lenguaje natural
        with _silence_stdout():
            summary = vn.generate_summary(question=query, df=df)
        # Opcional: preguntas de seguimiento si se pasa --questions
        do_questions = args.questions
        questions = []
        if do_questions:
            with _silence_stdout():
                questions = vn.generate_followup_questions(question=query, sql=sql, df=df)
        # Opcional: preguntas globales si se pasa --global-questions
        do_global_questions = args.global_questions
        global_questions = []
        if do_global_questions:
            with _silence_stdout():
                global_questions = vn.generate_questions()
        result = {
            "sql": sql,
            "dataframe": text,
            "summary": summary,
            "plot_path": plot_path,
        }
        if do_questions:
            result["questions"] = questions
        if do_global_questions:
            result["global_questions"] = global_questions
        sys.stdout.write(_format_markdown(result) + "\n")

    except Exception as e:
        _error_exit(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
