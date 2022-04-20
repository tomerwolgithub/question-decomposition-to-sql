import copy
import re
import itertools

from qdmr_grounding import extract_comparator


def get_condition(ground_step_dict):
    return ground_step_dict["WHERE"]


def get_columns(ground_step_dict):
    return ground_step_dict["SELECT"]


def get_distinct(ground_step_dict):
    return "distinct , " if ground_step_dict["distinct"] else ""


def get_group(ground_step_dict):
    return ground_step_dict["GROUP"]


def get_group_select(ground_step_dict):
    group_select = ground_step_dict["SELECT"]
    col = group_select[0]
    # "COUNT(DISTINCT car_makers.maker)" --> DISTINCT car_makers.maker
    regex = re.compile(".*?\((.*?)\)")
    result = re.findall(regex, col)
    return result


def get_having_clause(ground_step_dict):
    return ground_step_dict["HAVING"]


def get_order_clause(ground_step_dict):
    return ground_step_dict["ORDER"]


def get_order_columns(ground_step_dict):
    return ground_step_dict["ORDER BY"]


def get_superlative_agg(step):
    order_clause_str = get_order_clause(step)[0].lower()
    agg = "max" if order_clause_str.startswith("desc ") else "min"
    return agg


def get_superlative_arg_k(step):
    order_clause_str = get_order_clause(step)[0].lower()
    if "limit" in order_clause_str:
        k = order_clause_str.split("limit")[1].strip()
        return k
    return "1"


def get_join_clause(ground_step_dict):
    return ground_step_dict["JOIN"]


def is_reference(phrase):
    phrase = phrase.strip()
    return re.match("^#[0-9]*$", phrase)


def extract_refs(phrase):
    refs = []
    toks = [i.replace(",", "").strip() for i in phrase.split()]
    return list(filter(lambda x: is_reference(x), toks))


def get_arithmetic_op(arithmetic_phrase):
    op_map = {}
    op_map["sum"] = "+"
    op_map["difference"] = "-"
    op_map["multiplication"] = "*"
    op_map["division"] = "/"
    if arithmetic_phrase not in op_map.keys():
        return None
    return op_map[arithmetic_phrase]


def get_new_joined_cols(new_step, old_step):
    new_step_cols = list(itertools.chain.from_iterable(get_join_clause(new_step)))
    old_step_cols = list(itertools.chain.from_iterable(get_join_clause(old_step)))
    added_cols = list(filter(lambda col: col not in old_step_cols, new_step_cols))
    return added_cols


def get_new_joined_column(new_step, old_step):
    """
    Return the last new column added to the join clause
    which is the grounded column of a filter step phrase
    as it is at the end of the added join path.
    """
    added_columns = get_new_joined_cols(new_step, old_step)
    if added_columns != []:
        return [added_columns[-1]]
    return added_columns


def grounded_step_encoding(data, step):
    d = copy.deepcopy(data)
    op = d["op"]
    # check if distinct exists
    distinct = get_distinct(step)
    argument_list = d["arguments"]
    return "%s ( %s%s )" % (op, distinct, parse_string_list(argument_list))


def parse_string_list(string_list):
    if isinstance(string_list[0], (list, tuple)):
        new_list = [" ".join(element_list) for element_list in string_list]
        string_list = new_list
    if isinstance(string_list[-1], (list, tuple)):
        # conditions list
        string_list[-1] = format_conditions(string_list[-1])
    return " , ".join(string_list)


def format_conditions(conds_list):
    cond_strings = []
    for cond_triple in conds_list:
        cond_strings += [" ".join(cond_triple)]
    return " , ".join(cond_strings)


def no_reference_encoding(encoded_steps):
    step_phrases = {}
    for i in range(len(encoded_steps)):
        step = encoded_steps[i]
        ref_idxs = [int(ref.replace("#", "")) for ref in extract_refs(step)]
        sorted_ref_idxs = sorted(ref_idxs, key=int, reverse=True)
        enc_step = step
        for idx in sorted_ref_idxs:
            # go over references in desc order to avoid #1 #1x replacement issues
            enc_step = enc_step.replace(f"#{idx}", step_phrases[idx])
        step_phrases[i + 1] = enc_step
    return step_phrases[len(encoded_steps)]


def has_reference_encoding(encoded_steps):
    return " ; ".join(encoded_steps)


# parse grounded steps

def grounded_select(step, qdmr_args=None):
    data = {}
    data["op"] = "select"
    sql_cond = get_condition(step)
    # select column or select DB value
    data["arguments"] = sql_cond if len(sql_cond) > 0 else get_columns(step)
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_project(step, qdmr_args):
    data = {}
    data["op"] = "project"
    project_cols = get_columns(step)
    ref = qdmr_args[1]
    data["arguments"] = project_cols + [ref]
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_filter(step, qdmr_args, grounded_steps):
    data = {}
    data["op"] = "filter"
    ref, _ = qdmr_args
    conditions = get_condition(step)
    ref_step = grounded_steps[ref.replace("#", "")]
    prev_conditions = get_condition(ref_step)
    new_conds = list(filter(lambda cond: cond not in prev_conditions, conditions))
    # three cases filter is a: (1) condition; (2) joined column; (3) distinct
    args_list = [new_conds] if len(new_conds) > 0 else get_new_joined_column(step, ref_step)
    data["arguments"] = [ref] + args_list
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_aggregate(step, qdmr_args):
    data = {}
    data["op"] = "aggregate"
    data["arguments"] = qdmr_args
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_group(step, qdmr_args):
    data = {}
    data["op"] = "group"
    aggregate, values, keys = qdmr_args
    values = [values] if is_reference(values) else get_group_select(step)
    keys = [keys] if is_reference(keys) else get_group(step)
    data["arguments"] = [aggregate] + values + keys
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_superlative(step, qdmr_args):
    data = {}
    data["op"] = "superlative"
    min_max = get_superlative_agg(step)
    keys, values = qdmr_args[1:]
    arg_k = get_superlative_arg_k(step)
    data["arguments"] = [min_max, keys, values, arg_k]
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_comparative(step, qdmr_args, grounded_steps):
    data = {}
    data["op"] = "comparative"
    keys, values, condition = qdmr_args
    # two cases: (1) value is string grounded in column; (2) value is subquery
    # (1) extract the new condition (col, comp, val) from the grounded conditions clause
    conditions = get_condition(step)
    keys_step = grounded_steps[keys.replace("#", "")]
    values_step = grounded_steps[values.replace("#", "")]
    prev_conditions = get_condition(keys_step) + get_condition(values_step)
    new_conds = list(filter(lambda cond: cond not in prev_conditions, conditions))
    # (2) extract the comparator and referenced subquery from the arguments
    comparator, value = extract_comparator(condition)
    if is_reference(value):
        new_conds = [[cond[0], cond[1], value] for cond in new_conds]
    data["arguments"] = [keys, values, new_conds]
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_comparative_group(step, qdmr_args):
    data = {}
    data["op"] = "comparative_group"
    keys, values, condition = qdmr_args
    data["arguments"] = [keys, values, get_having_clause(step)]
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_intersection(step, qdmr_args):
    data = {}
    data["op"] = "intersection"
    projection = get_columns(step)
    refs = qdmr_args[1:]
    data["arguments"] = projection + refs
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_union_column(step, qdmr_args):
    data = {}
    data["op"] = "union_column"
    data["arguments"] = qdmr_args
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_union(step, qdmr_args):
    data = {}
    data["op"] = "union"
    data["arguments"] = qdmr_args
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_discard(step, qdmr_args):
    data = {}
    data["op"] = "discard"
    input_orig, input_discarded = qdmr_args
    orig = [input_orig] if is_reference(input_orig) else get_columns(step)
    discarded = [input_discarded] if is_reference(input_discarded) else get_columns(step)
    data["arguments"] = orig + discarded
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_sort(step, qdmr_args, grounded_steps, encoding):
    def order_columns_is_subquery(clause):
        return "SELECT " in clause and " FROM " in clause

    def sql_step_id(sql, grnd_steps):
        for num in grnd_steps:
            if sql in grnd_steps[num]["SQL"]:
                return int(num) - 1
        return None

    def get_grounded_enc_subquery(sql, grnd_steps, encoding_list):
        step_id = sql_step_id(sql, grnd_steps)
        return encoding_list[step_id]["string"]

    data = {}
    data["op"] = "sort"
    results_phrase = qdmr_args[0]
    results_ref = extract_refs(results_phrase)[0]
    order_cols = get_order_columns(step)
    if order_columns_is_subquery(order_cols[0]):
        # order column is a sql subquery, extract the *encoded* subquery
        order_cols = [get_grounded_enc_subquery(order_cols[0], grounded_steps, encoding)]
    order = get_order_clause(step)
    data["arguments"] = [results_ref] + order_cols + order
    data["string"] = grounded_step_encoding(data, step)
    return data


def grounded_arithmetic(step, qdmr_args):
    data = {}
    data["op"] = "arithmetic"
    arith, ref1, ref2 = qdmr_args
    arith_op = get_arithmetic_op(arith)
    assert arith_op is not None
    data["arguments"] = [arith_op, ref1, ref2]
    data["string"] = grounded_step_encoding(data, step)
    return data


def encode_qdmr_steps(qdmr_steps, grounded_steps):
    encoding = []
    for num in grounded_steps:
        step = grounded_steps[num]
        operator = step["op"]
        sql = step["SQL"]
        qdmr_step = qdmr_steps[int(num) - 1]
        if operator == "select":
            encoding += [grounded_select(step)]
        if operator == "project":
            encoding += [grounded_project(step, qdmr_step.arguments)]
        if operator == "filter":
            encoding += [grounded_filter(step, qdmr_step.arguments, grounded_steps)]
        if operator == "aggregate":
            encoding += [grounded_aggregate(step, qdmr_step.arguments)]
        if operator == "group":
            encoding += [grounded_group(step, qdmr_step.arguments)]
        if operator == "superlative" or operator == "superlative_group":
            encoding += [grounded_superlative(step, qdmr_step.arguments)]
        if operator == "comparative":
            encoding += [grounded_comparative(step, qdmr_step.arguments, grounded_steps)]
        if operator == "comparative_group":
            encoding += [grounded_comparative_group(step, qdmr_step.arguments)]
        if operator == "intersection":
            encoding += [grounded_intersection(step, qdmr_step.arguments)]
        if operator == "union_column":
            encoding += [grounded_union_column(step, qdmr_step.arguments)]
        if operator == "union":
            encoding += [grounded_union(step, qdmr_step.arguments)]
        if operator == "discard":
            encoding += [grounded_discard(step, qdmr_step.arguments)]
        if operator == "sort":
            encoding += [grounded_sort(step, qdmr_step.arguments, grounded_steps, encoding)]
        if operator == "arithmetic":
            encoding += [grounded_arithmetic(step, qdmr_step.arguments)]
    encoded_strs = [enc_step["string"] for enc_step in encoding]
    return encoded_strs
