# SQL Query Abstraction

import db_schema
from qdmr_identifier import *


class SQLQuery:
    def __init__(self, schema_path, query=None):
        self.schema_path = schema_path
        self.query = query
        self.results = None
        self.tables = None
        self.columns = None
        self.subqueries = None
        self.schema = db_schema.DBSchema(self.schema_path)

    def execute(self):
        results = set()
        conn = db_schema.sqlite3.connect(self.schema_path)
        c = conn.cursor()
        for row in c.execute(self.query):
            results.add(row)
        conn.close()
        self.results = results
        return True

    def add_subquery(self, subquery):
        if self.subqueries == None:
            self.subqueries = []
        self.subqueries += [subquery]
        return True

    def ground(self, qdmr, question=None):
        if self.query is not None:
            return False

        # parse QDMR
        qdmr_steps = QDMRProgramBuilder(qdmr).build_steps()
        string_steps = [str(step) for step in qdmr_steps]
        print(string_steps)

        # add referenced subqueries
        # build SQL query
        return True
