import re


def remove_quotation_marks(string):
    if string.startswith("'") and string.endswith("'"):
        return string[1:-1]
    if string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    return string


def sql_quotation_values(sql):
    """
    Returns list of lowercase quotation value in SQL query
    """

    def remove_like_pct(string):
        string = string.replace("%", "")
        return string

    query = sql.lower()
    value_to_col = {}
    # Find all values based on string delimiters
    single_paren_vals = [item.group(0) for item in re.finditer(r'\'.*?\'', query)]
    double_paren_vals = [item.group(0) for item in re.finditer(r'\".*?\"', query)]
    vals_list = single_paren_vals + double_paren_vals
    return [remove_quotation_marks(remove_like_pct(val)) for val in vals_list]


def val_sql_quotation(value):
    return [f"'{value}'", f"'%{value}%'", f'\"{value}\"', f'\"%{value}%\"']


def sql_value_case(sql, value):
    """returns a non-numeric value in the case it appears in the original SQL query"""

    def escape_parentheses(value):
        return value.replace("(", "\(").replace(")", "\)")

    value_quotation = val_sql_quotation(value)
    for quote_val in value_quotation:
        escaped_val = escape_parentheses(quote_val)
        if re.search(escaped_val, sql, re.IGNORECASE):
            return re.search(escaped_val, sql, re.IGNORECASE).group(0)
    return None


def fix_sql_casing(pred_sql, gold_sql):
    gold_values = sql_quotation_values(gold_sql)
    fixed_sql = pred_sql
    for val in gold_values:
        val_case_quotes = sql_value_case(gold_sql, val)
        if val_case_quotes is not None:
            val_case = remove_quotation_marks(val_case_quotes)
            for pred_val_cased in val_sql_quotation(val_case):
                # quoted value as it appears in the *predicted* sql
                fixed_sql = fixed_sql.replace(pred_val_cased.lower(),
                                              pred_val_cased) if pred_val_cased.lower() in fixed_sql else fixed_sql
    return fixed_sql
