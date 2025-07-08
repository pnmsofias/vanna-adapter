# vanna_adapter/cli.py (Solución definitiva para v0.7.9)
import json
import os
import sys
from pathlib import Path

import pandas as pd
import re
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url

# --- IMPORTACIONES CORRECTAS Y VERIFICADAS PARA v0.7.9 ---
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class Vanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Clase combinada de Vanna para RAG local con OpenAI y ChromaDB."""
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

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

TRAIN_FLAG = Path('.trained')
DEFAULT_MODEL = "gpt-4o-mini"


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
        "query": os.getenv("VANNA_QUERY"),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    }
    missing = [k for k, v in env.items() if k in {"api_key", "db_url", "query"} and not v]
    if missing:
        _error_exit(f"Missing environment variables: {', '.join(missing)}")
    return env


def _build_vanna(api_key: str, model: str) -> Vanna:
    """Configura e instancia Vanna componiendo sus partes."""
    # Permitir cambiar la URL base del LLM mediante variable de entorno
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url

    config = {'api_key': api_key, 'model': model}
    return Vanna(config=config)


def _train(vn: Vanna, engine) -> None:
    """Obtiene DDL y entrena a Vanna."""
    ddl = vn.get_ddl(engine=engine)
    vn.train(ddl=ddl)
    TRAIN_FLAG.touch()


def _generate_chart(vn: Vanna, df: pd.DataFrame, sql: str, question: str) -> str | None:
    """Genera un gráfico (best-effort)."""
    try:
        plotly_code = vn.generate_plotly_code(question=question, sql=sql, df=df)
        fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
        fig.write_html("plot.html")
        return "plot.html"
    except Exception:
        return None


def _quote_table_names(sql: str, engine) -> str:
    """Quote CamelCase table names using SQLAlchemy metadata reflection."""
    from sqlalchemy import MetaData
    metadata = MetaData()
    metadata.reflect(bind=engine)
    for name in metadata.tables:
        # Replace unquoted table names with quoted ones
        pattern = rf'(?<!")\b{name}\b(?!")'
        sql = re.sub(pattern, f'"{name}"', sql)
    return sql


def _process_query(vn: Vanna, engine, query: str) -> None:
    """Stub: _process_query está reemplazado por vn.ask() en main"""
    pass


def main() -> None:
    """Punto de entrada principal del CLI."""
    try:
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
        
        query = env["query"].strip()

        if query.lower() == "/update":
            TRAIN_FLAG.unlink(missing_ok=True)
        
        if not TRAIN_FLAG.exists():
            try:
                _train(vn, engine)
                if query.lower() == "/update":
                    sys.stdout.write(json.dumps({"message": "Model updated"}, separators=(",", ":")) + "\n")
                    return
            except Exception as e:
                _error_exit(f"Error training model: {e}")



        # Ejecutar flujo RAG oficial de Vanna
        try:
            sql, df, fig = vn.ask(
                question=query,
                print_results=False,
                auto_train=False,
                visualize=False,
                allow_llm_to_see_data=True
            )
        except Exception as e:
            _error_exit(str(e))

        text = df.head(1000).to_string(index=False) if df is not None else ""
        plot_path = None
        if fig is not None:
            fig.write_html("plot.html")
            plot_path = "plot.html"

        # Generar respuesta en lenguaje natural
        summary = vn.generate_summary(question=query, df=df)
        # Opcional: preguntas de seguimiento si se pasa --questions
        do_questions = "--questions" in sys.argv
        questions = []
        if do_questions:
            questions = vn.generate_followup_questions(question=query, sql=sql, df=df)
        result = {
            "sql": f"```sql\n{sql}\n```",
            "dataframe": text,
            "summary": summary,
            "plot_path": plot_path,
        }
        if do_questions:
            result["questions"] = questions
        sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")

    except Exception as e:
        _error_exit(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
