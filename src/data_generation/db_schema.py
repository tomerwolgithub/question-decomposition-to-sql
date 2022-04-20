import random
import itertools
from schema_parser import *
from graph_utils import *
from utils import get_table_and_column


class DBSchema:
    def __init__(self, schema_path):
        self.schema_path = schema_path
        schema_parser = SchemaParser()
        self.schema = schema_parser.parse(schema_path, None)
        tables = self.schema["table_names_original"]
        columns = self.schema["column_names_original"]
        count = 0
        self.num_to_column = {}
        for pair in columns:
            table_id, column_name = pair
            if column_name == '*':
                count += 1
                continue
            table_name = tables[table_id]
            self.num_to_column[count] = "%s.%s" % (table_name, column_name)
            self.num_to_column[count] = self.num_to_column[count].lower()
            count += 1

    def primary_keys(self):
        pk = self.schema["primary_keys"]
        return [self.num_to_column[key].lower() for key in pk]

    def foreign_keys(self, original=None):
        fk = self.schema["foreign_keys"]  # list of lists of size 2
        if not original:
            # augment foreign keys using identical column names
            fk += self.identical_column_names()
            fk.sort()
            fk = list(fk for fk, _ in itertools.groupby(fk))  # remove duplicates in list of lists
        fk_pairs = [(self.num_to_column[key1], self.num_to_column[key2]) for (key1, key2) in fk]
        # reverse pairs
        fk_pairs += [(self.num_to_column[key2], self.num_to_column[key1]) for (key1, key2) in fk]
        fk_pairs = list(set(fk_pairs))  # remove duplicates
        return fk_pairs

    def identical_column_names(self):
        """Return all schema columns with identical names.
        Only column names of length > 1 are considered (e.g., player_id).
        Only column names that contain the word "update" are discarded."""
        identical_colname_pairs = []
        for i in self.num_to_column.keys():
            colname_i = self.num_to_column[i].split(".")[1]
            if len(colname_i.split("_")) > 1 and "update" not in colname_i:
                for j in self.num_to_column.keys():
                    colname_j = self.num_to_column[j].split(".")[1]
                    if i != j and colname_i == colname_j:
                        identical_colname_pairs += [[i, j]]
        return identical_colname_pairs

    def column_to_num(self, table, col):
        column_name = "%s.%s" % (table, col)
        for num in self.num_to_column.keys():
            curr_col = self.num_to_column[num].lower()
            if curr_col == column_name.lower():
                return num
        raise ValueError("Invalid column name: %s" % column_name)

    def add_foreign_key(self, table, col, other_table, other_col):
        """Add new foreign key constraint to schema"""
        col_num = self.column_to_num(table, col)
        other_col_num = self.column_to_num(other_table, other_col)
        self.schema["foreign_keys"] += [[col_num, other_col_num]]
        return True

    def columns(self):
        return [i.lower() for i in list(self.num_to_column.values())]

    def column_names(self):
        return [col.split(".")[1] for col in self.columns()]

    def tables(self):
        return [i.lower() for i in self.schema['table_names_original']]

    def column_type(self, col):
        """
        Returns value type stored in column

        Parameters
        ----------
        col : str
            Schema full column name


        Returns
        -------
        str
            Column type (e.g., text, number)
        """
        for i in self.num_to_column.keys():
            if col == self.num_to_column[i]:
                return self.schema["column_types"][i]
        return None

    def default_column(self, table_name):
        """
        Returns a default column for a table

        Parameters
        ----------
        table_name : str
            Schema table name


        Returns
        -------
        str
            default column representing the table
        """
        if table_name not in self.tables():
            return None
        if self.get_table_pk(table_name) is not None:
            # return primary key
            return self.get_table_pk(table_name)
        # iterate over all table columns
        cols = self.get_table_columns(table_name)
        for col in cols:
            _, column_name = get_table_and_column(col)
            for trigger in ["name", "title", "id"]:
                if trigger in column_name:
                    return col
        # otherwise return random column
        i = random.randint(0, len(cols) - 1)
        return cols[i]

    def get_table_columns(self, table_name):
        table_columns = []
        for col in self.columns():
            if col.startswith(table_name + "."):
                table_columns += [col]
        return table_columns

    def get_table_pk(self, table_name):
        table_primary_keys = []
        pk = self.primary_keys()
        for key in pk:
            key_table = key.split('.')[0]
            if table_name == key_table:
                table_primary_keys += [key]
        assert len(table_primary_keys) <= 1
        if table_primary_keys == []:
            return None
        return table_primary_keys[0]

    def build_graph(self):
        table_arr = self.tables()
        graph = {}
        for tab in table_arr:
            graph[tab] = []
        fk = self.foreign_keys()
        for pair in fk:
            tab1, tab2 = pair
            tab1 = tab1.split('.')[0]
            tab2 = tab2.split('.')[0]
            graph[tab1] += [tab2]
            graph[tab2] += [tab1]
        for vertex in graph.keys():
            graph[vertex] = list(set(graph[vertex]))  # remove duplicates
        return graph

    def optimal_join_path(self, from_table, to_table):
        """
        Returns optimal join paths between two tables

        Parameters
        ----------
        from_table : str
            Schema table name
        to_table : str
            Another schema table name

        Returns
        -------
        list
            List representing a join path:
            [col1, col2, ...]
        """
        sorted_join_paths = self.join_paths(from_table, to_table)
        if not sorted_join_paths:
            return sorted_join_paths
        shortest_path = sorted_join_paths[0]
        optimal_path = shortest_path
        # favor bridge table paths if exist
        bridge_tables = ["%s_%s" % (from_table, to_table), \
                         "%s_%s" % (to_table, from_table)]
        for path in sorted_join_paths:
            if len(path) > len(optimal_path):
                break
            if (bridge_tables[0] in path or \
                    bridge_tables[1] in path):
                optimal_path = path
        return optimal_path

    def join_paths(self, from_table, to_table):
        """
        Returns all schema join paths between two tables

        Parameters
        ----------
        from_table : str
            Schema table name

        to_table : str
            Another schema table name

        Returns
        -------
        list
            List of lists, each inner list is a join path:
            [col1, col2, ...]
        """
        if from_table == to_table:
            # Inner join - get table primary keys
            pk = self.get_table_pk(from_table)
            return [pk]
        else:
            # build schema graph
            graph = self.build_graph()
            if not has_path(graph, from_table, to_table):
                # if no path exists in schema graph
                return None
            return find_shortest_paths(graph, from_table, to_table)
        return True

    def fix_triangle_join(self, constraint):
        """
        Fixes join constraint if the schema & constraint
        constitute a 'triangle join'.
        Trianlge join is when for constraint (t1.col1, t2.col2)
        exist the following 3 joins:
            (t1.col1, t2.col2), (t1.col3, t2.col2), (t1.col1, t1.col3)
        E.g.:
            1. (border.border, state.name), (border.state, state.name),
                (border.border, border.state)
            2. (father.father, person.name), (father.name, person.name),
                (father.father, father.name)

        Parameters
        ----------
        constraint : tuple
            Pair of (t1.col1, t2.col2) of column names
            to appear in join path

        Returns
        -------
        tuple
            (t1.col3, t2.col2) if triangle join exists,
            otherwise (t1.col1, t2.col2)
        """
        if constraint is None:
            return None
        col1, col2 = constraint
        # check for join triangle
        table1_other_col = self.get_triangle_join_col(col1, col2)
        if table1_other_col is not None:
            return table1_other_col, col2
        table2_other_col = self.get_triangle_join_col(col2, col1)
        if table2_other_col is not None:
            return col1, table2_other_col
        return constraint

    def get_triangle_join_col(self, col1, col2):
        """
        Returns third column involved in triangle join, otherwise None

        Triangle join is when for constraint (t1.col1, t2.col2)
        exist the following 3 joins:
            (t1.col1, t2.col2), (t1.col3, t2.col2), (t1.col1, t1.col3)
        """
        table1, _ = get_table_and_column(col1)
        table2, _ = get_table_and_column(col2)
        fks = self.foreign_keys()
        if table1 == table2 or (col1, col2) not in fks:
            return None
        # (col1,col2) are joined
        for fk_pair in fks:
            col, other_col = fk_pair
            if col2 == other_col:
                # (col,col2) are joined
                table, _ = get_table_and_column(col)
                if (col != col1 and table == table1 and \
                        (col1, col) in fks):
                    # (table1.col1,table1.col) are joined
                    return col
        return None

    def join_path_chain(self, join_path, constraint=None):
        """
        Returns a list of column pairs representing the join path

        Parameters
        ----------
        join_path : list
            Join path list of schema tables
        constraint : tuple
            Pair of (t1.col1, t2.col2) of column names
            to appear in join path

        Returns
        -------
        list
            List of pairs, constituting the join path:
            [(col1, col2), ...]
        """
        join_chain = []
        fks = self.foreign_keys()
        constraint = self.fix_triangle_join(constraint)
        for i in range(len(join_path) - 1):
            from_table = join_path[i]
            to_table = join_path[i + 1]
            found_pair = False
            if constraint is not None:
                col_from, col_to = constraint
                # search for fk containing both constraint columns
                # or at least one of the columns
                for fk_pair in fks:
                    if col_from in fk_pair and col_to in fk_pair:
                        fk_tables = [col.split(".")[0] for col in fk_pair]
                        if [from_table, to_table] == fk_tables:
                            join_chain += [fk_pair]
                            found_pair = True
                            break
                if not found_pair:
                    for fk_pair in fks:
                        if col_from in fk_pair or col_to in fk_pair:
                            fk_tables = [col.split(".")[0] for col in fk_pair]
                            if [from_table, to_table] == fk_tables:
                                join_chain += [fk_pair]
                                found_pair = True
                                break

            if not found_pair:
                candidate_fks = []
                for fk_pair in fks:
                    # find relevant
                    fk_tables = [col.split(".")[0] for col in fk_pair]
                    if [from_table, to_table] == fk_tables:
                        candidate_fks += [fk_pair]
                for fk_pair in candidate_fks:
                    if "id" in fk_pair[0] or "id" in fk_pair[1]:
                        # heuristic to favor join paths containing id columns
                        join_chain += [fk_pair]
                        found_pair = True
                        break
                if not found_pair and len(candidate_fks) > 0:
                    join_chain += [candidate_fks[0]]
        return join_chain
