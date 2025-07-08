import json
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from vanna.local import VannaLocal

TRAIN_FLAG = Path('.trained')
DEFAULT_MODEL = "gpt-4o-mini"


def _error_exit(message: str) -> None:
    payload = json.dumps({"error": message}, separators=(",", ":"))
    sys.stderr.write(payload + "\n")
    sys.exit(1)


def _load_env() -> dict:
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


def _build_vanna(api_key: str, model: str) -> VannaLocal:
    llm = OpenAI(config={"api_key": api_key, "model": model})
    vectorstore = Chroma()
    return VannaLocal(llm=llm, vectorstore=vectorstore)


def _train(vn: VannaLocal) -> None:
    vn.train()
    TRAIN_FLAG.touch()


def _generate_chart(vn: VannaLocal, df: pd.DataFrame, sql: str) -> str | None:
    try:
        fig = vn.generate_chart(df=df, sql=sql)
        fig.write_html("plot.html")
        return "plot.html"
    except Exception:
        return None


def _process_query(vn: VannaLocal, engine, query: str) -> None:
    sql = vn.generate_sql(query)
    df = vn.run_sql(sql, engine=engine)
    text = df.head(1000).to_string(index=False)
    plot_path = _generate_chart(vn, df, sql)
    result = {
        "sql": f"```sql\n{sql}\n```",
        "dataframe": text,
        "plot_path": plot_path,
    }
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")


def main() -> None:
    env = _load_env()
    vn = _build_vanna(env["api_key"], env["model"])
    engine = create_engine(env["db_url"])
    query = env["query"].strip()

    if query.lower() == "/update":
        TRAIN_FLAG.unlink(missing_ok=True)
        try:
            _train(vn)
        except Exception:
            _error_exit("Error training model")
        sys.stdout.write(json.dumps({"message": "Model updated"}, separators=(",", ":")) + "\n")
        return

    if not TRAIN_FLAG.exists():
        try:
            _train(vn)
        except Exception:
            _error_exit("Error training model")

    try:
        _process_query(vn, engine, query)
    except Exception:
        _error_exit("Error processing query")


if __name__ == "__main__":
    main()
