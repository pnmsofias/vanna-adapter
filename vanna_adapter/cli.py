# vanna_adapter/cli.py (Solución definitiva para v0.7.9)
import json
import os
import sys
from pathlib import Path
import argparse
import io
from contextlib import redirect_stdout

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
        vn = _build_vanna(env["api_key"], env["model"])
        # Conectar Vanna a Postgres para introspección de datos
        url = make_url(env["db_url"])
        vn.connect_to_postgres(
            host=url.host,
            dbname=url.database,
            user=url.username,
            password=url.password,
            port=url.port
        )
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

        text = df.head(1000).to_string(index=False) if df is not None else ""
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
