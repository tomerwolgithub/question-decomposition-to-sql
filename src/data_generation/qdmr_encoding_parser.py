import pyparsing as pp
from collections import namedtuple

from qdmr_encoding import is_reference

QDMR_STEP_DELIMITER = ";"

op_list = ["select", "project", "filter", "aggregate", "group", "superlative", "comparative",
           "comparative_group", "intersection", "union_column", "union", "discard", "sort", "arithmetic"]
comparators = ["=", ">", "<", ">=", "<=", "!=", "LIKE", "like", "BETWEEN", "start", "end"]
aggregates = ["COUNT", "SUM", "AVG", "MIN", "MAX", "count", "sum", "avg", "min", "max"]
arithmetics = ["+", "-", "*", "/"]
OP = pp.oneOf(op_list)
COMP = pp.oneOf(comparators)
AGGR = pp.oneOf(aggregates)
ARITHMETIC = pp.oneOf(arithmetics)
LP = pp.Literal("(").suppress()
RP = pp.Literal(")").suppress()
COMMA = pp.Literal(",").suppress()
String = pp.Word(pp.alphanums + "_" + "-" + "." + "%" + "*" + "/")
SingleQuoteString = pp.QuotedString(quoteChar="'", unquoteResults=False)
DoubleQuoteString = pp.QuotedString(quoteChar='"', unquoteResults=False)
QuotedString = SingleQuoteString | DoubleQuoteString
ConditionPrefix = AGGR + pp.Literal("(") + String + pp.Literal(")") | String
BetweenValue = pp.Word(pp.alphanums) + pp.Literal("AND") + pp.Word(pp.alphanums)
BasicCondition = pp.Group(ConditionPrefix + COMP + pp.OneOrMore(String))
Atom = BasicCondition | ConditionPrefix | QuotedString | ARITHMETIC
SExpr = pp.Forward()
FormulaCondition = ConditionPrefix + COMP + SExpr
SExprList = pp.Group((FormulaCondition | SExpr | Atom) + pp.ZeroOrMore(COMMA + (FormulaCondition | SExpr | Atom)))
SExpr << (OP + LP + SExprList + RP)

Node = namedtuple("Node", ["operator", "arguments"])


def parseAction(string, location, tokens):
    return Node(operator=tokens[0], arguments=tokens[1:])


SExpr.setParseAction(parseAction)


def pprint(node, tab=""):
    print(tab + u"|--" + str(node.operator))
    new_tab = tab + "    "
    for arg in node.arguments[0]:
        if isinstance(arg, Node):
            pprint(arg, new_tab)
        else:
            print(new_tab + arg)


def formula_dfs(node, stack):
    s = "%s ( " % str(node.operator)
    space = " "
    for i in range(len(node.arguments[0])):
        arg = node.arguments[0][i]
        comma = "" if i == 0 else " , "
        if isinstance(arg, Node):
            last_token = s[-1]
            # handle case where argument is formula value of a condition
            delimiter = comma if last_token not in comparators else space
            s += delimiter + str(formula_dfs(arg, stack)[0])
        elif isinstance(arg, pp.ParseResults):
            s += comma + ' '.join(arg)  # argument is a simple condition list
        else:
            # handle case where argument is the comparator of a formula condition
            s += comma + str(arg) if arg not in comparators else space + str(arg)
    s += " )"
    stack += [s]
    return s, stack


def dfs_ref_substitution(dfs_qdmr_steps):
    def remove_references(steps_list):
        return list(filter(lambda x: not is_reference(x), steps_list))

    ret_steps = []
    for i in range(len(dfs_qdmr_steps)):
        next_ref = "#%s" % (len(ret_steps) + 1)
        next_step = dfs_qdmr_steps[i]
        new_steps = []
        if not is_reference(next_step):
            for step in dfs_qdmr_steps[i + 1:]:
                new_steps += [step.replace(next_step, next_ref)]
            dfs_qdmr_steps = dfs_qdmr_steps[:i + 1] + new_steps
            ret_steps += [next_step]
    return remove_references(dfs_qdmr_steps)


def formula_qdmr_to_ref_steps(qdmr_formula_encoding):
    parsed = SExpr.parseString(qdmr_formula_encoding)
    dfs_steps = formula_dfs(parsed[0], [])[1]
    return dfs_ref_substitution(dfs_steps)


def formula_to_ref_encoding(qdmr_formula_encoding):
    ref_steps = formula_qdmr_to_ref_steps(qdmr_formula_encoding)
    delim = " %s " % QDMR_STEP_DELIMITER
    return delim.join(ref_steps)
