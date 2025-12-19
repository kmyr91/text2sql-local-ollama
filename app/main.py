import os
import re
from collections import defaultdict
from typing import Any, Dict, Set, Tuple

import psycopg2
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Text2SQL Project")


# -----------------------------
# DB connection
# -----------------------------
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "chinook"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )

def get_table_columns(table: str):
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s
    ORDER BY ordinal_position;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (table,))
            return [r[0] for r in cur.fetchall()]

# -----------------------------
# Schema helpers
# -----------------------------
def get_real_tables_set() -> Set[str]:
    q = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema='public' AND table_type='BASE TABLE'
    """
    tables = set()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            for (t,) in cur.fetchall():
                tables.add(t.lower())
    return tables


def get_fk_edges_set() -> Set[Tuple[str, str, str, str]]:
    """
    Returns FK edges in BOTH directions:
      (table, col, other_table, other_col)
    so joins can be validated regardless of which side is written first.
    """
    q = """
    SELECT
      tc.table_name AS table_name,
      kcu.column_name AS column_name,
      ccu.table_name AS foreign_table_name,
      ccu.column_name AS foreign_column_name
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
     AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = 'public';
    """
    edges = set()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            for t, c, ft, fc in cur.fetchall():
                t, c, ft, fc = t.lower(), c.lower(), ft.lower(), fc.lower()
                edges.add((t, c, ft, fc))
                edges.add((ft, fc, t, c))
    return edges


def get_schema_text() -> str:
    q = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;
    """

    tables = defaultdict(list)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            for table_name, column_name, data_type in cur.fetchall():
                tables[table_name].append((column_name, data_type))

    lines = []
    for t, cols in tables.items():
        cols_str = ", ".join([f"{c} ({dt})" for c, dt in cols])
        lines.append(f"{t}: {cols_str}")

    # add FK relationships for better joins
    fk_edges = get_fk_edges_set()
    fk_lines = []
    # show only one direction for readability
    seen = set()
    for (a, ac, b, bc) in fk_edges:
        key = tuple(sorted([(a, ac), (b, bc)]))
        if key in seen:
            continue
        seen.add(key)
        fk_lines.append(f"{a}.{ac} <-> {b}.{bc}")

    return "TABLES:\n" + "\n".join(lines) + "\n\nFOREIGN KEYS:\n" + "\n".join(sorted(fk_lines))


# -----------------------------
# SQL safety + cleaning
# -----------------------------
BANNED = ["drop", "delete", "update", "insert", "alter", "truncate", "grant", "revoke"]


def clean_llm_sql(text: str) -> str:
    t = (text or "").strip()

    # strip markdown fences
    t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    t = t.replace("```", "").strip()

    # strip leading labels
    t = re.sub(r"^(sql|query)\s*:\s*", "", t, flags=re.IGNORECASE).strip()

    # if model added extra text, start at first WITH/SELECT
    m = re.search(r"\b(with|select)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start():].strip()

    return t


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    # block multi-statement: keep only first statement
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    return sql


def validate_select_only(sql: str) -> None:
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")

    for kw in BANNED:
        if re.search(rf"\b{kw}\b", s):
            raise HTTPException(status_code=400, detail=f"Blocked keyword: {kw}")


def ensure_limit(sql: str, limit: int = 100) -> str:
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {limit}"


def validate_joins_follow_fk(sql: str) -> None:
    """
    Validates JOIN equality conditions (a.col = b.col) follow real foreign keys.
    Skips validation if FK constraints are not present.
    Skips conditions that involve CTEs (non-table names).
    """
    fk_edges = get_fk_edges_set()
    if not fk_edges:
        return  # no FK constraints present; can't validate joins

    real_tables = get_real_tables_set()

    # Build alias -> table map from FROM/JOIN
    alias_map: Dict[str, str] = {}

    # capture FROM table [alias]
    m = re.search(r"\bfrom\s+(\w+)(?:\s+(\w+))?", sql, flags=re.IGNORECASE)
    if m:
        table = m.group(1)
        alias = m.group(2) or table
        alias_map[alias.lower()] = table.lower()

    # capture JOIN table [alias]
    for jm in re.finditer(r"\bjoin\s+(\w+)(?:\s+(\w+))?", sql, flags=re.IGNORECASE):
        table = jm.group(1)
        alias = jm.group(2) or table
        alias_map[alias.lower()] = table.lower()

    # check each equality in ON clauses (handles ON ... AND ... by finding all = pairs)
    for om in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", sql, flags=re.IGNORECASE):
        a, ac, b, bc = om.group(1), om.group(2), om.group(3), om.group(4)

        if a.lower() not in alias_map or b.lower() not in alias_map:
            raise HTTPException(status_code=400, detail="JOIN uses an undefined alias.")

        ta, tb = alias_map[a.lower()], alias_map[b.lower()]

        # skip if either side isn't a real table (likely a CTE)
        if ta not in real_tables or tb not in real_tables:
            continue

        if (ta, ac.lower(), tb, bc.lower()) not in fk_edges:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JOIN (not a foreign key): {ta}.{ac} = {tb}.{bc}",
            )


# -----------------------------
# SQL runner
# -----------------------------
def run_sql(sql: str) -> Dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return {"columns": [], "rows": []}
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return {"columns": cols, "rows": rows}


# -----------------------------
# LLM (Ollama) + hints
# -----------------------------
def build_hints(question: str) -> str:
    q = question.lower()

    # Small “domain hint” that fixes the common mistake you just saw
    if "artist" in q and ("revenue" in q or "sales" in q):
        return (
            "For artist revenue in Chinook, join: artist -> album -> track -> invoice_line "
            "and compute SUM(invoice_line.unit_price * invoice_line.quantity). "
            "Do NOT add extra WHERE filters unless the user explicitly asks."
        )


    if "customer" in q and ("invoice" in q or "spent" in q or "amount" in q):
        return (
            "Customer spending comes from invoice.total joined via "
            "customer.customer_id = invoice.customer_id."
        )

    return ""


def ollama_generate_sql(question: str, schema_text: str, extra_context: str = "") -> str:
    system_instructions = (
        "Write ONE PostgreSQL SQL query to answer the user question.\n"
        "Rules:\n"
        "- Output ONLY SQL (no markdown, no explanation).\n"
        "- Use only tables/columns from the schema.\n"
        "- Use the FOREIGN KEYS section for joins.\n"
        "- Every alias used must be defined in FROM/JOIN.\n"
        "- SELECT or WITH only.\n"
        "- Always include LIMIT 100.\n"
    )

    hints = build_hints(question)

    prompt = f"SCHEMA:\n{schema_text}\n\nQUESTION:\n{question}\n"
    if hints:
        prompt += f"\nHINT:\n{hints}\n"
    if extra_context:
        prompt += f"\nFEEDBACK (SQL error or validation failure):\n{extra_context}\n"
    prompt += "\nSQL:"

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_instructions,
        "stream": False,
        "options": {"temperature": 0},
    }

    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=300)
    r.raise_for_status()

    raw = (r.json().get("response") or "").strip()
    return clean_llm_sql(raw)


# -----------------------------
# API models
# -----------------------------
class RunSQLRequest(BaseModel):
    sql: str


class AskRequest(BaseModel):
    question: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"schema_text": get_schema_text()}


@app.post("/run_sql")
def run_sql_endpoint(req: RunSQLRequest):
    sql = normalize_sql(req.sql)
    validate_select_only(sql)
    sql = ensure_limit(sql, 200)
    return run_sql(sql)


@app.post("/ask")
def ask(req: AskRequest):
    if os.getenv("LLM_PROVIDER", "ollama").lower() != "ollama":
        raise HTTPException(status_code=400, detail="Set LLM_PROVIDER=ollama in .env")

    schema_text = get_schema_text()

    # Attempt 1
    sql1 = ensure_limit(normalize_sql(ollama_generate_sql(req.question, schema_text)), 100)

    try:
        validate_select_only(sql1)
        validate_joins_follow_fk(sql1)
        result1 = run_sql(sql1)
        return {"sql": sql1, "result": result1}
    except Exception as e1:
        feedback = str(e1)
        extra = ""

        # If Postgres says "column X.Y does not exist", tell the model the real columns of that table.
        m = re.search(r"column\s+(\w+)\.(\w+)\s+does not exist", feedback, flags=re.IGNORECASE)
        if m:
            alias = m.group(1).lower()
            bad_col = m.group(2).lower()

            # Build alias_map from sql1
            alias_map = {}
            mm = re.search(r"\bfrom\s+(\w+)(?:\s+(\w+))?", sql1, flags=re.IGNORECASE)
            if mm:
                t = mm.group(1).lower()
                a = (mm.group(2) or t).lower()
                alias_map[a] = t

            for jm in re.finditer(r"\bjoin\s+(\w+)(?:\s+(\w+))?", sql1, flags=re.IGNORECASE):
                t = jm.group(1).lower()
                a = (jm.group(2) or t).lower()
                alias_map[a] = t

            if alias in alias_map:
                table = alias_map[alias]
                cols = get_table_columns(table)
                extra = (
                    f"You used {alias}.{bad_col} but it does not exist. "
                    f"Table '{table}' columns are: {', '.join(cols)}. "
                    "Remove that filter/column and use only valid columns from schema."
                )
            else:
                extra = "Remove any reference to non-existent columns. Use ONLY columns shown in the schema."

        # Attempt 2 (retry once)
        sql2 = ensure_limit(
            normalize_sql(
                ollama_generate_sql(req.question, schema_text, extra_context=feedback + "\n" + extra)
            ),
            100,
        )

        try:
            validate_select_only(sql2)
            validate_joins_follow_fk(sql2)
            result2 = run_sql(sql2)
            return {"sql": sql2, "result": result2, "note": "Fixed after 1 retry"}
        except Exception as e2:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Failed even after 1 retry",
                    "first_sql": sql1,
                    "first_error": feedback,
                    "second_sql": sql2,
                    "second_error": str(e2),
                },
            )
