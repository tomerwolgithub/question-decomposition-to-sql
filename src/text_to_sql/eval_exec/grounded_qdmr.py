from eval_exec.qdmr_grounding import QDMRGrounding, is_reference
from eval_exec.qdmr_identifier import QDMRStep
from eval_exec.utils import get_table_and_column
from collections import OrderedDict

ARG_DELIMITER = ","
STEP_DELIMITER = ";"
COMPARATORS = ["=", ">", ">=", "<", "<=", "LIKE", "like"]  ###TODO: handle start and end LIKE conditions in grounding script


def extract_step_type(step_enc):
    return step_enc.split("(")[0].strip()


def extract_step_arguments(step_enc):
    def after_opening_paren(str):
        return str.split("(", 1)[1]

    def before_closing_paren(str):
        return str.rsplit(")", 1)[0]

    inside_parentheses = before_closing_paren(after_opening_paren(step_enc))
    args_list = inside_parentheses.split(ARG_DELIMITER)
    stripped_args = [arg.strip() for arg in args_list]
    return stripped_args


def parse_grounded_encoding(qdmr_enc):
    def parse_grounded_step(step_enc):
        type = extract_step_type(step_enc)
        args = extract_step_arguments(step_enc)
        return QDMRStep(step_enc, type, args)

    encoded_steps = qdmr_enc.split(STEP_DELIMITER)
    qdmr_steps = [parse_grounded_step(step) for step in encoded_steps]
    return qdmr_steps


def create_grounded_select_step(column):
    return QDMRStep(None, "select", [column])


def qdmr_grounding_instance(question, schema):
    return QDMRGrounding(None, question, schema)


def is_distinct_step(qdmr_step):
    if qdmr_step.arguments[0] == "distinct":
        return True
    return False


def is_condition_argument(argument):
    parts = argument.split()
    return len(parts) >= 3 and (parts[1] in COMPARATORS)


def cond_to_tuple(condition_str):
    col, comparator, value = condition_str.split()
    value = value.strip()
    return col.strip(), comparator.strip(), value.strip()


def select_all(project_col):
    return project_col.strip() == "*"


def get_ref_number(ref):
    return ref.replace("#", "").strip()


def sql_condition_triple(condition_phrase):
    condition_tokens = condition_phrase.split()
    aggr_col, comparator = condition_tokens[:2]
    value = ' '.join(condition_tokens[2:])  # handle BETWEEN conditions
    return aggr_col, comparator, value


class GroundedQDMR(object):
    def __init__(self, qdmr_enc, question, schema, enc_no_ref=None):
        self.index = 0
        self.sql_steps = {}
        self.qdmr = qdmr_enc
        self.question = question
        self.schema = schema
        self.argument_delimiter = ","
        self.steps = parse_grounded_encoding(qdmr_enc)
        self.grounding_instance = qdmr_grounding_instance(question, schema)

    def to_sql(self):
        for i, step in enumerate(self.steps):
            sql_clauses = None
            op = step.operator
            final_step = (i == len(self.steps) - 1)
            if op == "select":
                sql_clauses = self.select_to_sql(step)
            elif op == "project":
                sql_clauses = self.project_to_sql(step)
            elif op == "filter":
                sql_clauses = self.filter_to_sql(step)
            elif op == "aggregate":
                sql_clauses = self.aggregate_to_sql(step)
            elif op == "group":
                sql_clauses = self.group_to_sql(step, final_step)
            elif op == "comparative":
                sql_clauses = self.comparative_to_sql(step)
            elif op == "comparative_group":
                sql_clauses = self.comparative_group_to_sql(step)
            elif op == "superlative":
                sql_clauses = self.superlative_to_sql(step)
            elif op == "intersection":
                sql_clauses = self.intersection_to_sql(step)
            elif op == "union":
                sql_clauses = self.union_to_sql(step)
            elif op == "union_column":
                sql_clauses = self.union_column_to_sql(step)
            elif op == "sort":
                sql_clauses = self.sort_to_sql(step)
            elif op == "discard":
                sql_clauses = self.discard_to_sql(step)
            elif op == "arithmetic":
                sql_clauses = self.arithmetic_to_sql(step)
            else:
                sql_clauses = None
                assert sql_clauses is not None
            self.sql_steps[str(i + 1)] = sql_clauses
            self.index += 1
        return self.sql_steps

    def column_to_step(self, column):
        step = create_grounded_select_step(column)
        return self.select_to_sql(step)

    def select_to_sql(self, step):
        distinct = is_distinct_step(step)
        args = step.arguments[1:] if distinct else step.arguments
        assert len(args) == 1  # select argument is either a single column or condition
        col = args[0]
        conditions = []
        if is_condition_argument(args[0]):
            cond = sql_condition_triple(args[0])
            col = cond[0]
            conditions = [cond]
        table, _ = get_table_and_column(col)
        return self.grounding_instance.grounding_to_sql("select", \
                                                        select_cols=[col], \
                                                        from_tables=[table], \
                                                        conditions=conditions, \
                                                        distinct=distinct)

    def project_to_sql(self, step):
        distinct = is_distinct_step(step)
        args = step.arguments[1:] if distinct else step.arguments
        col, ref = args
        ref_num = get_ref_number(ref)
        ref_step = self.sql_steps[ref_num]
        if select_all(col):
            return self.grounding_instance.grounding_to_sql("project", \
                                                            select_cols=["*"], \
                                                            from_tables=ref_step["FROM"], \
                                                            join_pairs=ref_step["JOIN"], \
                                                            conditions=ref_step["WHERE"], \
                                                            group=ref_step["GROUP"], \
                                                            having=ref_step["HAVING"], \
                                                            order_by=ref_step["ORDER BY"], \
                                                            order=ref_step["ORDER"], \
                                                            union=ref_step["UNION"], \
                                                            discard=ref_step["DISCARD"], \
                                                            intersect=ref_step["INTERSECT"], \
                                                            projected_ref_idx=ref_num)
        # join path inference between referenced step and projection
        project_table, _ = get_table_and_column(col)
        ref_column = ref_step["SELECT"][0]
        from_tables, join_clause = self.grounding_instance.join_tables(ref_step["FROM"], ref_step["JOIN"], \
                                                                       [project_table], [], \
                                                                       constraint=(ref_column, col))
        ###TODO: handle project self-joins
        conds = [self.grounding_instance.subquery_condition(ref_step)]
        return self.grounding_instance.grounding_to_sql("project", \
                                                        select_cols=[col], \
                                                        from_tables=from_tables, \
                                                        join_pairs=join_clause, \
                                                        conditions=conds, \
                                                        distinct=distinct, \
                                                        projected_ref_idx=ref_num)

    def filter_to_sql(self, step):
        distinct = is_distinct_step(step)
        args = step.arguments[1:] if distinct else step.arguments
        # three cases filter is a: (1) condition; (2) joined column; (3) distinct
        ref_step = self.sql_steps[get_ref_number(args[0])]
        if len(args) == 1 and is_reference(args[0]):
            # E.g., filter( distinct, #2 )
            return self.grounding_instance.grounding_to_sql("filter", \
                                                            select_cols=ref_step["SELECT"], \
                                                            from_tables=ref_step["FROM"], \
                                                            join_pairs=ref_step["JOIN"], \
                                                            conditions=ref_step["WHERE"], \
                                                            group=ref_step["GROUP"], \
                                                            having=ref_step["HAVING"], \
                                                            order_by=ref_step["ORDER BY"], \
                                                            order=ref_step["ORDER"], \
                                                            union=ref_step["UNION"], \
                                                            discard=ref_step["DISCARD"], \
                                                            intersect=ref_step["INTERSECT"], \
                                                            distinct=distinct)
        # filter( #2, cond1, cond2, ...) or filter( #2, table.col )
        filter_tables, filter_columns, filter_conditions = [], [], []
        if len(args[1:]) == 1 and len(args[1].split()) == 1:
            joined_column = args[1]
            filter_columns = [joined_column]
            table, _ = get_table_and_column(joined_column)
            filter_tables = [table]
        else:
            filter_conditions = [sql_condition_triple(cond) for cond in args[1:]]
            filter_columns = [sql_condition_triple(cond)[0] for cond in args[1:]]
            for column in filter_columns:
                table, _ = get_table_and_column(column)
                filter_tables += [table]
        # join all extracted columns with the referenced query
        new_from, new_join = ref_step["FROM"], ref_step["JOIN"]
        for i in range(len(filter_tables)):
            table = filter_tables[i]
            col = filter_columns[i]
            constraint = (ref_step["SELECT"][0], col)
            new_from, new_join = self.grounding_instance.join_tables(new_from, new_join, [table], [],
                                                                     constraint=constraint)
        all_conds = ref_step["WHERE"] + filter_conditions
        if self.grounding_instance.has_conflicting_conditions(filter_conditions, ref_step["WHERE"]):
            # self-join conflicting conditions, create subquery
            all_conds = filter_conditions + [self.subquery_condition(ref_step)]
        return self.grounding_instance.grounding_to_sql("filter",
                                                        select_cols=ref_step["SELECT"],
                                                        from_tables=new_from,
                                                        join_pairs=new_join,
                                                        conditions=all_conds,
                                                        group=ref_step["GROUP"],
                                                        having=ref_step["HAVING"],
                                                        order_by=ref_step["ORDER BY"],
                                                        order=ref_step["ORDER"],
                                                        union=ref_step["UNION"],
                                                        discard=ref_step["DISCARD"],
                                                        intersect=ref_step["INTERSECT"],
                                                        distinct=distinct)

    def aggregate_to_sql(self, step):
        distinct = is_distinct_step(step)
        args = step.arguments[1:] if distinct else step.arguments
        aggregate, ref = args
        ref_step = self.sql_steps[get_ref_number(ref)]
        select_col = "%s(%s)" % (aggregate.upper(), ref_step["SELECT"][0])
        order_by = ref_step["ORDER BY"]
        order = ref_step["ORDER"]
        if ref_step["GROUP"] != [] and ref_step["GROUP"] is not None:
            # if ref step is a "group" handle ref as a subquery condition
            if aggregate in ["max", "min"]:
                # max / min on a referenced group - add order clause
                select_col = ref_step["SELECT"][0]
                order_by = ref_step["SELECT"]
                order = ["DESC LIMIT 1"] if aggregate == "max" else ["ASC LIMIT 1"]
            else:
                return self.grounding_instance.grounding_to_sql("aggregate", \
                                                                select_cols=[select_col], \
                                                                from_tables=ref_step["FROM"], \
                                                                join_pairs=ref_step["JOIN"], \
                                                                conditions=[self.grounding_instance.subquery_condition(
                                                                    ref_step)], \
                                                                aggregate=aggregate, \
                                                                distinct=ref_step["distinct"])
        return self.grounding_instance.grounding_to_sql("aggregate", \
                                                        select_cols=[select_col], \
                                                        from_tables=ref_step["FROM"], \
                                                        join_pairs=ref_step["JOIN"], \
                                                        conditions=ref_step["WHERE"], \
                                                        aggregate=aggregate, \
                                                        group=ref_step["GROUP"], \
                                                        having=ref_step["HAVING"], \
                                                        order_by=order_by, \
                                                        order=order, \
                                                        union=ref_step["UNION"], \
                                                        discard=ref_step["DISCARD"], \
                                                        intersect=ref_step["INTERSECT"], \
                                                        distinct=ref_step["distinct"])

    def group_to_sql(self, step, final_step):
        aggregate, values, keys = step.arguments
        keys_step = self.sql_steps[get_ref_number(keys)] if is_reference(keys) else \
            self.column_to_step(keys)  # keys is a column string: "number of #1 for each year"
        values_step = self.sql_steps[get_ref_number(values)] if is_reference(values) else \
            self.column_to_step(values)  # values is a column string: "maximum height for each #3"
        from_tables, join_clause = self.grounding_instance.join_tables(values_step["FROM"], values_step["JOIN"], \
                                                                       keys_step["FROM"], keys_step["JOIN"])
        return self.grounding_instance.grounding_to_sql("group", \
                                                        select_cols=[
                                                            "%s(%s)" % (aggregate.upper(), values_step["SELECT"][0])], \
                                                        from_tables=from_tables, \
                                                        join_pairs=join_clause, \
                                                        conditions=values_step["WHERE"] + keys_step["WHERE"], \
                                                        group=keys_step["SELECT"], \
                                                        aggregate=aggregate, \
                                                        final_step=final_step, \
                                                        discard=values_step["DISCARD"], \
                                                        intersect=values_step["INTERSECT"])

    def superlative_to_sql(self, step):
        return self.grounding_instance.ground_superlative_step(step, grounded_qdmr_steps=self.sql_steps)

    def comparative_to_sql(self, step):
        keys, values, condition = step.arguments
        values_step = self.sql_steps[get_ref_number(values)]
        keys_step = self.sql_steps[get_ref_number(keys)]
        cond_triple = sql_condition_triple(condition)
        # two cases:
        #   (1) condition value is string grounded in column;
        #   (2) condition value is subquery
        cond_column, cond_comparator, cond_val = cond_triple
        if is_reference(cond_val):
            # (2) compared value is a reference to subquery
            val = "( %s )" % self.sql_steps[get_ref_number(cond_val)]["SQL"]
            cond_triple = (cond_column, cond_comparator, val)
        cond_table, _ = get_table_and_column(cond_column)
        from_tables, join_clause = self.grounding_instance.join_tables([cond_table], [], \
                                                                       keys_step["FROM"], keys_step["JOIN"])
        conditions = values_step["WHERE"] + keys_step["WHERE"] + [cond_triple]
        if self.grounding_instance.has_conflicting_conditions([cond_triple],
                                                              values_step["WHERE"]+keys_step["WHERE"]):
            # self-join conflicting conditions, create subquery
            subquery_step = self.grounding_instance.grounding_to_sql("comparative", \
                                                                     select_cols=keys_step["SELECT"], \
                                                                     from_tables=from_tables, \
                                                                     join_pairs=join_clause, \
                                                                     conditions=keys_step["WHERE"]+keys_step["WHERE"])
            conditions = [cond_triple] + [self.grounding_instance.subquery_condition(subquery_step)]
        return self.grounding_instance.grounding_to_sql("comparative", \
                                                        select_cols=keys_step["SELECT"], \
                                                        from_tables=from_tables, \
                                                        join_pairs=join_clause, \
                                                        conditions=conditions)

    def comparative_group_to_sql(self, step):
        keys, values, having = step.arguments
        group_step = self.sql_steps[get_ref_number(values)]  # values refers to group step
        keys_step = self.sql_steps[get_ref_number(keys)]
        return self.grounding_instance.grounding_to_sql("comparative_group", \
                                                        select_cols=keys_step["SELECT"], \
                                                        from_tables=group_step["FROM"], \
                                                        join_pairs=group_step["JOIN"], \
                                                        conditions=group_step["WHERE"] + keys_step["WHERE"], \
                                                        group=keys_step["SELECT"], \
                                                        having=[sql_condition_triple(having)])

    def intersection_to_sql(self, step):
        distinct = is_distinct_step(step)
        args = step.arguments[1:] if distinct else step.arguments
        proj_column = args[0]
        step.arguments = args
        return self.grounding_instance.ground_intersection_step(step,
                                                                grounded_project_col=proj_column,
                                                                grounded_qdmr_steps=self.sql_steps)

    def union_to_sql(self, step):
        refs = step.arguments
        ref_steps = [self.sql_steps[get_ref_number(r)] for r in refs]
        return self.grounding_instance.ground_union_step(step, grounded_qdmr_ref_steps=ref_steps)

    def union_column_to_sql(self, step):
        refs = step.arguments
        ref_steps = [self.sql_steps[get_ref_number(r)] for r in refs]
        return self.grounding_instance.ground_column_union_step(step, grounded_qdmr_ref_steps=ref_steps)

    def discard_to_sql(self, step):
        input_orig, input_discarded = step.arguments
        orig_step = self.sql_steps[get_ref_number(input_orig)] if is_reference(input_orig) else \
            self.column_to_step(input_orig)  # orig is a column string: "instructors besides #2"
        discarded_step = self.sql_steps[get_ref_number(input_discarded)] if is_reference(input_discarded) else \
            self.column_to_step(input_discarded)  # discarded is a column string: "#2 besides cats"
        return self.grounding_instance.grounding_to_sql("discard", \
                                                        select_cols=orig_step["SELECT"], \
                                                        from_tables=orig_step["FROM"], \
                                                        join_pairs=orig_step["JOIN"], \
                                                        conditions=orig_step["WHERE"], \
                                                        group=orig_step["GROUP"], \
                                                        having=orig_step["HAVING"], \
                                                        order_by=orig_step["ORDER BY"], \
                                                        order=orig_step["ORDER"], \
                                                        union=orig_step["UNION"], \
                                                        discard=[(orig_step["SELECT"][0], discarded_step["SQL"])], \
                                                        intersect=orig_step["INTERSECT"])

    def sort_to_sql(self, step):
        ref = step.arguments[0]
        order_cols = step.arguments[1:-1]
        order = step.arguments[-1]
        ref_step = self.sql_steps[get_ref_number(ref)]
        # join the results step with the order steps
        from_tables, join_clause = ref_step["FROM"], ref_step["JOIN"]
        for full_column in order_cols:
            table, col = get_table_and_column(full_column)
            from_tables, join_clause = self.grounding_instance.join_tables(from_tables, join_clause, \
                                                                           [table], [])
        select_cols, group, having = ref_step["SELECT"], ref_step["GROUP"], ref_step["HAVING"]
        # TODO: potential issue when original order step is a group step. \
        #  A solution is changing sort encoding to: sort(#ref1, #ref2,..,#refk, order) & inspecting #ref2
        return self.grounding_instance.grounding_to_sql("sort", \
                                                        select_cols=select_cols, \
                                                        from_tables=list(dict.fromkeys(from_tables)), \
                                                        join_pairs=list(dict.fromkeys(join_clause)), \
                                                        conditions=ref_step["WHERE"], \
                                                        group=group, \
                                                        having=having, \
                                                        order_by=order_cols, \
                                                        order=[order])

    def arithmetic_to_sql(self, step):
        arith_op, ref1, ref2 = step.arguments
        subquery1 = self.sql_steps[get_ref_number(ref1)]["SQL"]
        subquery2 = self.sql_steps[get_ref_number(ref2)]["SQL"]
        select_col = "(%s) %s (%s)" % (subquery1, arith_op, subquery2)
        return self.grounding_instance.grounding_to_sql("arithmetic", \
                                                        select_cols=[select_col])
