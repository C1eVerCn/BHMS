#!/usr/bin/env python3
"""用本地知识库 JSON 初始化 Neo4j 图谱。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from kg.graphrag_engine import Neo4jKnowledgeGraph


def main() -> None:
    settings = get_settings()
    graph = Neo4jKnowledgeGraph(
        knowledge_path=settings.knowledge_path,
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    graph.ensure_seeded()
    graph.close()
    print(f"Neo4j graph initialized from {settings.knowledge_path}")


if __name__ == '__main__':
    main()
