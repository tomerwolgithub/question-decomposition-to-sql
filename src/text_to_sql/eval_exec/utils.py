import re

DELIMITER = ';'
REF = '#'


def parse_decomposition(qdmr):
    """Parses the decomposition into an ordered list of steps

    Parameters
    ----------
    qdmr : str
        String representation of the QDMR

    Returns
    -------
    list
        returns ordered list of qdmr steps
    """
    # remove digit commas 1,000 --> 1000
    matches = re.findall(r"[\d,]+[,\d]", qdmr)
    for m in matches:
        no_comma = m.replace(",", "")
        qdmr = qdmr.replace(m, no_comma)
    # parse commas as separate tokens
    qdmr = qdmr.replace(",", " , ")
    crude_steps = qdmr.split(DELIMITER)
    steps = []
    for i in range(len(crude_steps)):
        step = crude_steps[i]
        tokens = step.split()
        step = ""
        # remove 'return' prefix
        for tok in tokens[1:]:
            step += tok.strip() + " "
        step = step.strip()
        steps += [step]
    return steps


def get_table_and_column(full_column_name):
    return full_column_name.split(".")


def extract_comparator(condition):
    """
    Returns comparator and value of a
    QDMR comparative step condition

    Parameters
    ----------
    condition : str
        Phrase representing condition of a QDMR step

    Returns
    -------
    tuple
        (comparator, value)
    """
    # extract comparative
    numbers = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", \
               "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    comparatives = {}
    comparatives["BETWEEN"] = ["between"]
    comparatives[">"] = ["more than", "above", "larger than", "larger", \
                         "older than", "older", "higher than", "higher", \
                         "greater than", "greater", "bigger than", "bigger", \
                         "after", "over"]
    comparatives[">="] = ["at least"]
    comparatives["<"] = ["less than", "under", "lower than", "lower", \
                         "younger than", "younger", "before", "below", \
                         "shorter than", "smaller than", "smaller"]
    comparatives["<="] = ["at most"]
    comparatives["!="] = ["is not"]
    comparatives["start"] = ['start with', 'starts with', 'begin']
    comparatives["end"] = ['end with', 'ends with']
    comparatives["LIKE"] = ["the letter", "the string", "the word", "the phrase", \
                            "contain", "include", "has", "have", \
                            "contains", "substring", "includes"]
    comparatives["="] = ['is equal to', 'equal to', 'same as', \
                         'is ', 'are ', 'was ']
    unformatted = {}
    unformatted[">="] = ["or later", "or more", "or after"]
    unformatted["<="] = ["or earlier", "or less", "or before"]
    ###TODO: handle "NOT LIKE"
    comp = None
    for c in comparatives.keys():
        if comp:
            break
        for trigger in comparatives[c]:
            if trigger in condition:
                comp = c
                break
    if comp:
        # extract value/reference
        value_phrase = condition.split(trigger)[1].strip()
        if comp == "BETWEEN":
            # "between num1 AND num2"
            return comp, value_phrase.upper()
        elif comp:
            # check for unformatted comparators in value phrase
            for c in unformatted.keys():
                for trigger in unformatted[c]:
                    if trigger in condition:
                        comp = c
                        value_phrase = condition.split(trigger)[0].strip()
                        break
        for tok in value_phrase.split():
            if tok.isnumeric():
                return comp, tok
            if tok in numbers.keys():
                return comp, numbers[tok]
        return comp, value_phrase
    return "=", None
