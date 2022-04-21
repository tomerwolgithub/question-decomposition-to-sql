# A SQL query parser - for grounding evaluation
import re

from eval_exec.utils import get_table_and_column


class SQLParser(object):
    def __init__(self):
        self.tables = None
        self.columns = None

    def parse(self, query, schema):
        query_tables = set()
        query_columns = set()
        query = query.lower()
        tokens = query.split()
        for tok in tokens:
            if re.match(r't[0-9]\.', tok):
                query_columns.add(tok)
        # get tables in sql query
        from_clause = query.split('from')[1].split('where')[0]
        from_tokens = from_clause.split()
        schema_tables = schema.tables()
        schema_tables_lowercase = [name.lower() for name in schema_tables]
        for tok in from_tokens:
            if tok in schema_tables_lowercase:
                query_tables.add(tok)
        self.tables = list(query_tables)
        columns = list(query_columns)
        if len(self.tables) == 1:
            # all columns in query belong to a single table
            table_name = self.tables[0]
            for tok in tokens:
                # parse token from 'op()' and 'table.col'
                tok = tok.split('(')[1] if '(' in tok else tok
                tok = tok.split(')')[0] if ')' in tok else tok
                tok = tok.split('.')[1] if '.' in tok else tok
                schema_columns = schema.columns()
                for col in schema_columns:
                    if col == tok:
                        col_full_name = "%s.%s" % (table_name, col)
                        query_columns.add(col_full_name)
            self.columns = list(query_columns)
            return True
        # more than one table in query
        # replace column table alias T1.col --> table_name.col
        aliases = re.findall(r'as\st[0-9]', query)
        alias_map = {}
        for alias in aliases:
            table_alias = alias.split()[-1]
            prefix = query.split(alias)[0]
            table_name = prefix.split()[-1]
            alias_map[table_alias] = table_name
        self.columns = []
        for col in columns:
            for alias in alias_map.keys():
                if alias in col:
                    column_full = col.replace(alias, alias_map[alias])
                    self.columns += [column_full]
        self.columns = list(set(self.columns))
        return True

    def get_table_aliases(self, query):
        """Returns map from table alias (t#) to its name"""
        query = query.lower()
        aliases = re.findall(r'as\st[0-9]', query)
        alias_map = {}
        for alias in aliases:
            table_alias = alias.split()[-1]
            prefix = query.split(alias)[0]
            table_name = prefix.split()[-1]
            alias_map[table_alias] = table_name
        # map from table alias (e.g. t1) to its name
        return alias_map

    def extract_values(self, query, schema):
        query = query.lower()
        value_to_col = {}
        # Find all values based on string delimiters
        single_paren_vals = [item.group(0) for item in re.finditer(r'\'.*?\'', query)]
        double_paren_vals = [item.group(0) for item in re.finditer(r'\".*?\"', query)]
        number_vals = [item.group(0) for item in re.finditer(r'[0-9]+', query)]
        # filter numbers in table aliases e.g., 1 in T1
        number_vals = list(filter(lambda x: (" %s" % x) in query, number_vals))
        vals = single_paren_vals + double_paren_vals + number_vals
        # Map values to corresponding columns
        for value in vals:
            # SQL satement will be: "table.col operator value", e.g.:
            # T2.allergytype  =  "food"
            # name LIKE '%Led%'
            table = None
            prefix = query.split(value)[0]
            aliased_column = prefix.split()[-2]
            column_names = schema.column_names()
            schema_columns = schema.columns()
            if "." in aliased_column:
                # column is either aliased T#.col or table.col
                aliased_table, col = get_table_and_column(aliased_column)
                table = self.get_aliased_table(aliased_table, query, schema)
            elif aliased_column.lower() not in column_names:
                # nearest token is not column name
                # return the nearest column name instead
                preceding_toks = prefix.lower().split()
                for i in reversed(range(len(preceding_toks))):
                    if preceding_toks[i] in column_names:
                        aliased_column = preceding_toks[i]
                        break
            else:
                # no aliased table in query
                # find nearest table to the column name
                col = aliased_column
                col_match_positions = [m.start() for m in re.finditer(col, query)]
                last_match_pos = col_match_positions[-1]
                preceding_toks = query[:last_match_pos].split()
                table_names = schema.tables()
                for i in reversed(range(len(preceding_toks))):
                    if preceding_toks[i] in table_names:
                        table = preceding_toks[i]
                        full_col_name = "%s.%s" % (table, col)
                        if full_col_name in schema_columns:
                            # validate full column name is valid
                            break
            # non-number values have parentheses
            value_no_paren = value[1:-1] if not value.isdigit() else value
            if value_no_paren.startswith("%") \
                    and value_no_paren.endswith("%"):
                # value extracted from LIKE '%%' statement
                value_no_paren = value_no_paren[1:-1]
            if table:
                value_to_col[value_no_paren.strip()] = "%s.%s".strip() % (table, col)
        return value_to_col

    def get_aliased_table(self, aliased_table, query, schema):
        """
        Receive table name referenced query and retreive its actual table
        Handles:
            Spider aliases format e.g., T#.column
            ATIS aliases format e.g., table_#.column
        """
        table_aliases = self.get_table_aliases(query)
        if re.match(r't[0-9]', aliased_table):
            return table_aliases[aliased_table]
        if re.match(r'.*\_[0-9]', aliased_table):
            # remove the '_#' suffix
            actual_table = '_'.join(aliased_table.split('_')[:-1])
            if actual_table in schema.tables():
                return actual_table
        return aliased_table