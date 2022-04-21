import sqlite3
import time
import threading
import sys
from collections import OrderedDict

# from wrapt_timeout_decorator import *


TIMEOUT = 60


# TIMEOUT = 90 # 90 seconds for Academic, IMDB & Yelp


def interrupt_sqlite(connection):
    print('Interrupted sqlite connection', file=sys.stderr)
    connection.interrupt()


# @timeout(dec_timeout=TIMEOUT, use_signals=False)
def execute_sql(db, sql):
    """
    Returns a list of tuple that are the query results

    Parameters
    ----------
    db : str
        Full path to DB schema
    sql : str
        SQL query to be executed


    Returns
    -------
    list
        List of tuple that are the query results
    """
    conn = sqlite3.connect(db)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    c = conn.cursor()
    try:
        c.execute(sql)
    except:
        return None
    return c.fetchall()


def normalize_tuple(tup):
    # cast all tuple values to strings
    norm_vars = [str(var) for var in tup]
    return tuple(norm_vars)


def normalize_denotation(denotation_list, distinct=None):
    if not denotation_list:
        return denotation_list
    # remove duplicates
    denotation_list = list(OrderedDict.fromkeys(denotation_list)) if distinct else denotation_list
    sort_tuples = [sorted(normalize_tuple(tup)) for tup in denotation_list]
    return sorted(sort_tuples)  # sort results set


def correct_denotation(pred_sql, gold_sql, db_path, distinct=None):
    gold_denotation = execute_sql(db_path, gold_sql)
    pred_denotation = execute_sql(db_path, pred_sql)
    if gold_denotation == pred_denotation:
        # unnormalized denotations
        return True
    gold_denotation_norm = normalize_denotation(gold_denotation, distinct=distinct)
    pred_denotation_norm = normalize_denotation(pred_denotation, distinct=distinct)
    return gold_denotation_norm == pred_denotation_norm
