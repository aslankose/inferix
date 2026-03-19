from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from coordination.api import nodes, tasks, tokens, grid, inference
from coordination.config import settings
from coordination.db import get_db, check_connection
from coordination.ledger.ledger import ledger

app = FastAPI(
    title=       settings.APP_NAME,
    version=     settings.VERSION,
    description= (
        "Coordination Layer for the Inferix framework. "
        "Manages contributor nodes, dispatches compute tasks, "
        "issues FLOP-denominated tokens, and maintains the token ledger."
    ),
    docs_url=    "/docs",
    redoc_url=   "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=  ["*"],
    allow_methods=  ["*"],
    allow_headers=  ["*"],
)

app.include_router(nodes.router)
app.include_router(tasks.router)
app.include_router(tokens.router)
app.include_router(grid.router)
app.include_router(inference.router)


@app.get("/", tags=["Info"])
def root():
    return {
        "name":    settings.APP_NAME,
        "version": settings.VERSION,
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Info"])
def health(db: Session = Depends(get_db)):
    db_ok = check_connection()
    return {
        "status":         "healthy" if db_ok else "degraded",
        "database":       "connected" if db_ok else "disconnected",
        "ledger_entries": ledger.total_entries(db),
        "ledger_valid":   ledger.verify_chain(db),
        "total_supply":   round(ledger.total_supply(db), 6),
    }
