import pandas as pd
from pandas.io.json import json_normalize

import json
import re


def remove_space_between_quotes(string):
    matches = re.findall(r'\"(.+?)\"', string)  # match text between two quotes
    for m in matches:
        trimmed_m = m.strip()
        string = string.replace('\"%s\"' % m, '\"%s\"' % trimmed_m)
    return string


def add_space_between_conds(sql):
    for op in ['>=', '<=', '>', '>', '!=']:
        sql = sql.replace(op, " %s " % op)
    sql = sql.replace("in(", "in (")
    sql = sql.replace("IN(", "IN (")
    sql = sql.replace("=(", " = (")
    sql = sql.replace("='", " = '")
    sql = sql.replace('="', ' = "')
    sql = sql.replace("  ", " ")
    return sql


def change_not_equal_op(sql):
    sql = sql.replace("<>", "!=")
    return sql


def format_sql(sql):
    sql = change_not_equal_op(sql)
    sql = remove_space_between_quotes(sql)
    sql = add_space_between_conds(sql)
    return sql


def load_spider_data(dataset_path):
    """
    Reads query & DB info from Spider dataset

    Parameters
    ----------
    dataset_path : str
        Full path to dataset json

    Returns
    -------
    dict
        Dict of (db_id, question, query)
    """
    examples = {}
    with open(dataset_path) as f:
        data = json.load(f)
        for i in range(len(data)):
            db_id = data[i]["db_id"]
            sql = data[i]["query"]
            question = data[i]["question"]
            split = "train" if "train" in dataset_path else "dev"
            index = "SPIDER_%s_%d" % (split, i)
            examples[index] = {}
            examples[index]["db_id"] = db_id
            examples[index]["question"] = question
            examples[index]["query"] = sql
    assert len(data) == len(examples)
    return examples


def load_spider_others_data(dataset_path, target_db=None):
    """
    Reads query & DB info from Spider dataset

    Parameters
    ----------
    dataset_path : str
        Full path to dataset json
    target_db : str
        Filter according to specific db_id

    Returns
    -------
    dict
        Dict of (db_id, question, query)
    """
    examples = {}
    with open(dataset_path) as f:
        data = json.load(f)
        for i in range(len(data)):
            db_id = data[i]["db_id"]
            if target_db and db_id != target_db:
                continue
            sql = data[i]["query"]
            question = data[i]["question"]
            examples[question] = {}
            examples[question]["db_id"] = db_id
            examples[question]["question"] = question
            examples[question]["query"] = sql
    assert len(data) == len(examples)
    return examples


def load_washington_data(dataset_path, db):
    """
    Reads question & SQL query from the Washington &
    Texas universities text-to-SQL datasets:
    Academic, ATIS, Geo, IMDB, Yelp

    Parameters
    ----------
    dataset_path : str
        Full path to dataset txt file

    Returns
    -------
    dict
        Dict of (db_id, question, query)
    """
    examples = {}
    delimiter = "|||"
    queries_delimiter = "|"
    valid_dbs = ["academic", "atis", "geo", "imdb", "yelp"]
    assert db in valid_dbs
    with open(dataset_path) as f:
        for line in f:
            line_data = line.split(delimiter)
            question, sql = line.split(delimiter)[1:] if db in ["imdb", "yelp"] else line.split(delimiter)
            question = question.strip()
            sql = sql.strip()
            if queries_delimiter in sql:
                sql = sql.split(queries_delimiter)[0].strip()
            sql = format_sql(sql)
            examples[question] = {}
            examples[question]["db_id"] = db
            examples[question]["question"] = question
            examples[question]["query"] = sql
    return examples


def load_break_qdmrs(data_path, dataset_name=None):
    """
    Reads qdmr structures from Break

    Parameters
    ----------
    dataset_path : str
        Full path to dataset csv
    dataset_name : str
        Prefix of a specific dataset to load

    Returns
    -------
    Dataframe
        Dict of Spider_id --> ('question_id', 'question_text', 'decomposition')
    """
    df = pd.read_csv(data_path)
    if dataset_name is not None:
        df = df[df['question_id'].str.contains(dataset_name)]
    df = df[['question_id', 'question_text', 'decomposition']]
    df = df.set_index('question_id')
    return df.groupby('question_id').apply(lambda dfg: dfg.to_dict(orient='list')).to_dict()


def build_grounding_data(qdmr_data, text2sql_data, text2sql_format, \
                         db=None, split=None, output=None, to_csv=None):
    """
    Builds a json/csv file containing qdmr to sql examples

    Parameters
    ----------
    qdmr_data : str
        Path to file mapping text questions to their QDMRs
    text2sql_data : str
        Path to file containing text to SQL data
        complete with schema details
    text2sql_format : str
        Format of the text2sql file (spider/others/washington)
    db : str
        Particular db name (e.g.,"atis")
    split : str
        Dataset split, train/dev/test
    output : str
        Path to save the grounding data file


    Returns
    -------
    dict
        dictionary containing examples of
        id, db, question, sql, qdmr
    """
    grounding_data = {}
    assert text2sql_format in ["spider", "others", "washington"]
    if split is not None:
        assert split in ["train", "dev", "test"]
    if text2sql_format == "spider":
        dataset_name = "SPIDER_%s" % split
        qdmr = load_break_qdmrs(qdmr_data, dataset_name=dataset_name)
        print("* QDMR structures loaded: %s" % len(qdmr))
        text2sql = load_spider_data(text2sql_data)
        for example_id in qdmr:
            try:
                assert (qdmr[example_id]['question_text'][0] == \
                        text2sql[example_id]['question'])
            except:
                print("example_id: ", example_id)
                print("qdmr[example_id]['question_text'][0]: ", qdmr[example_id]['question_text'][0])
                print("text2sql[example_id]['question']", text2sql[example_id]['question'])
                print("***")
            grounding_data[example_id] = text2sql[example_id]
            grounding_data[example_id]['qdmr'] = qdmr[example_id]['decomposition'][0]
    elif text2sql_format == "others":
        qdmr = load_break_qdmrs(qdmr_data, dataset_name="ACADEMIC_train")
        qdmr.update(load_break_qdmrs(qdmr_data, dataset_name="GEO_train"))
        print("* QDMR structures loaded: %s" % len(qdmr))
        text2sql = load_spider_others_data(text2sql_data)
        for example_id in qdmr:
            key = qdmr[example_id]['question_text'][0].replace('"', '\"').strip()
            if key in text2sql.keys():
                grounding_data[example_id] = text2sql[key]
                grounding_data[example_id]['qdmr'] = qdmr[example_id]['decomposition'][0]
            else:
                print("*** Could not find key: ", key)
    elif text2sql_format == "washington":
        assert db is not None
        dataset_name = "%s_" % db.upper()
        if split is not None:
            dataset_name = "%s_%s" % (db.upper(), split)
        if db == "atis":
            qdmr = load_break_qdmrs(qdmr_data, dataset_name=dataset_name)
        elif db == "academic":
            qdmr = load_break_qdmrs(qdmr_data, dataset_name=dataset_name)
        elif db == "geo":
            qdmr = load_break_qdmrs(qdmr_data, dataset_name=dataset_name)
        elif db == "imdb":
            qdmr = load_break_qdmrs(qdmr_data, dataset_name=dataset_name)
        elif db == "yelp":
            qdmr = load_break_qdmrs(qdmr_data, dataset_name=dataset_name)
        else:
            raise Exception("Invalid db name: %s" % db)
        print("* QDMR structures loaded: %s" % len(qdmr))
        text2sql = load_washington_data(text2sql_data, db)
        for example_id in qdmr:
            key = qdmr[example_id]['question_text'][0].strip()
            if key in text2sql.keys():
                grounding_data[example_id] = text2sql[key]
                grounding_data[example_id]['qdmr'] = qdmr[example_id]['decomposition'][0]
            else:
                print("*** Could not find key: ", key)
    print("* Overall grounding data examples: %s" % len(grounding_data))
    if output:
        with open("%s.json" % output, "w") as fp:
            json.dump(grounding_data, fp)
            print("* Saved grounding data examples to %s.json" % output)
        if to_csv:
            df = pd.DataFrame.from_dict(grounding_data, orient="index").reset_index()
            df.to_csv("%s.csv" % output)
            print("* Saved grounding data examples to %s.csv" % output)
    return grounding_data