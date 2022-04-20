from qdmr_editor import *
from qdmr_grounding import *
import re


class GroundingRepair(object):
    def __init__(self, grounding_test_example):
        self.grounding_example = grounding_test_example

    def repair(self):
        return self._repair()

    def _repair(self):
        raise NotImplementedError
        return True


class AggrDistinctGroundingRepair(GroundingRepair):
    """
    Repair grounded SQL by replacing outer SELECT COUNT/SUM(...)
    clause with SELECT COUNT/SUM(DISTINCT ...)

    Also handles 'LIKE' to '=' (if necessary)
    """

    def __init__(self, grounding_test_example):
        super(AggrDistinctGroundingRepair, self).__init__(grounding_test_example)

    def _repair(self):
        grounding = self.grounding_example.grounding
        sql = grounding.get_grounded_sql()
        select_clause = sql.split("FROM")[0]
        if ("COUNT(" in select_clause and \
                "COUNT(DISTINCT " not in select_clause):
            repaired_sql = sql.replace("COUNT(", "COUNT(DISTINCT ", 1)
            self.grounding_example.set_grounded_sql(repaired_sql)
            if self.grounding_example.correct_denotation(distinct=True):
                return True
            elif "LIKE '" in sql:
                repaired_sql = self.replace_like_with_equals(repaired_sql)
                self.grounding_example.set_grounded_sql(repaired_sql)
                if self.grounding_example.correct_denotation(distinct=True):
                    return True
        elif ("SUM(" in select_clause and \
              "SUM(DISTINCT " not in select_clause):
            repaired_sql = sql.replace("SUM(", "SUM(DISTINCT ", 1)
            self.grounding_example.set_grounded_sql(repaired_sql)
            if self.grounding_example.correct_denotation(distinct=True):
                return True
            elif "LIKE '" in sql:
                repaired_sql = self.replace_like_with_equals(repaired_sql)
                self.grounding_example.set_grounded_sql(repaired_sql)
                if self.grounding_example.correct_denotation(distinct=True):
                    return True
        self.grounding_example.set_grounded_sql(sql)
        return False

    def replace_like_with_equals(self, sql):
        matches = re.findall(r"LIKE '%(.+?)%'", sql)  # match text between two quotes
        for m in matches:
            value = m.replace("LIKE '%", "").replace("%", "").strip()
            sql = sql.replace(f"LIKE '%{m}%'", f"= '{value}'")
        return sql


class LikeEqualsGroundingRepair(GroundingRepair):
    """
    Repair grounded SQL by replacing "column LIKE '%value%'"
    clauses with equality clauses "column = 'value'"
    """

    def __init__(self, grounding_test_example):
        super(LikeEqualsGroundingRepair, self).__init__(grounding_test_example)

    def _repair(self):
        grounding = self.grounding_example.grounding
        sql = grounding.get_grounded_sql()
        select_clause = sql.split("FROM")[0]
        if "LIKE '" in sql:
            repaired_sql = self.replace_like_with_equals(sql)
            self.grounding_example.set_grounded_sql(repaired_sql)
            if self.grounding_example.correct_denotation(distinct=True):
                return True
        self.grounding_example.set_grounded_sql(sql)
        return False

    def replace_like_with_equals(self, sql):
        matches = re.findall(r"LIKE '%(.+?)%'", sql)  # match text between two quotes
        for m in matches:
            value = m.replace("LIKE '%", "").replace("%", "").strip()
            sql = sql.replace(f"LIKE '%{m}%'", f"= '{value}'")
        return sql


class CountSumGroundingRepair(GroundingRepair):
    """
    Repair original QDMR decomposition and gold SQL mismatch
    by replacing count/sum group and aggregate steps.

    E.g.:
        count --> sum : "number of #2" --> "sum of #2"
        group_count --> project : "number of #2 for each #1" --> "#2"
        group_sum --> project : "sum of #2 for each #1" --> "#2"

    Also calls the column grounding repairs (if necessary)
    """

    def __init__(self, grounding_test_example):
        super(CountSumGroundingRepair, self).__init__(grounding_test_example)
        self.valid_aggregates = ["count", "sum"]

    def _repair(self):
        origin_qdmr = self.grounding_example.qdmr
        replacement_qdmrs = self.build_replacement_qdmrs(origin_qdmr)
        if replacement_qdmrs == []:
            return False
        for rep_qdmr in replacement_qdmrs:
            self.grounding_example.set_qdmr(rep_qdmr)
            try:
                ground = self.grounding_example.ground_example()
            except:
                continue
            if self.grounding_example.correct_denotation(distinct=True):
                return True
            else:
                col_ground_repair = ColumnGroundingRepair(self.grounding_example)
                if col_ground_repair.repair():
                    return True
        self.grounding_example.set_qdmr(origin_qdmr)
        return False

    def build_replacement_qdmrs(self, qdmr):
        replacement_qdmrs = []
        for agg in self.valid_aggregates:
            replacement_qdmrs += [self.replace_aggregate(qdmr, agg)]
            for other_agg in self.valid_aggregates:
                if agg != other_agg:
                    replacement_qdmrs += [self.replace_aggregate(qdmr, agg, other_agg)]
        replacement_qdmrs = list(filter(lambda x: x != qdmr, replacement_qdmrs))
        return list(set(replacement_qdmrs))

    def replace_aggregate(self, qdmr, aggregate, replacement=None):
        """
        Replace QDMR step aggregates (count/sum) in aggregate and group steps.
        Aggregate is either replaced with its reference step or with another aggregate.
        """
        assert aggregate in self.valid_aggregates
        assert replacement is None or replacement in self.valid_aggregates
        trigger = "number of #" if aggregate == "count" else "sum of #"
        if replacement:
            replacement_trigger = "number of #" if replacement == "count" else "sum of #"
        steps = parse_decomposition(qdmr)
        replaced = False
        for i in range(len(steps)):
            if trigger in steps[i]:
                replaced = True
                ref = int(extract_refs(steps[i])[0].replace("#", ""))
                steps[i] = steps[ref - 1] if replacement is None \
                    else steps[i].replace(trigger, replacement_trigger)
        if not replaced:
            return qdmr
        new_qdmr = ' ;return '.join(steps)
        return "return %s" % new_qdmr


class SuperlativeGroundingRepair(GroundingRepair):
    """
    Repair original QDMR decomposition and gold SQL mismatch
    by replacing project/aggregate/filter/comparative steps with superlatives.

    E.g.:
        project --> superlative : "youngest #1" -->
                                    "youngest type[#1] of #1"
                                    "#1 where #2 is highest/lowest"
        aggregate --> superlative : "largest of #2" -->
                                        "largest type[#2] of #2"
                                        "#2 where #3 is highest"
        filter --> superlative : "#1 that is the youngest" -->
                                    "type[#1] that is the youngest of #1"
                                    "#1 where #2 is highest/lowest"
        comparative --> superlative : "#1 where #2 is tallest" -->
                                        "#1 where #2 is highest/lowest"

    Also calls the column grounding repairs (if necessary)
    """

    def __init__(self, grounding_test_example):
        super(SuperlativeGroundingRepair, self).__init__(grounding_test_example)

    def _repair(self):
        origin_qdmr = self.grounding_example.qdmr
        replacement_qdmrs = self.build_replacement_qdmrs(origin_qdmr)
        if replacement_qdmrs == []:
            return False
        for rep_qdmr in replacement_qdmrs:
            self.grounding_example.set_qdmr(rep_qdmr)
            try:
                ground = self.grounding_example.ground_example()
            except:
                continue
            if self.grounding_example.correct_denotation(distinct=True):
                return True
            else:
                col_ground_repair = ColumnGroundingRepair(self.grounding_example)
                if col_ground_repair.repair():
                    return True
        self.grounding_example.set_qdmr(origin_qdmr)
        return False

    def build_replacement_qdmrs(self, qdmr):
        replacement_qdmrs = []
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        for i in range(len(builder.steps)):
            step = builder.steps[i]
            step_num = i + 1
            if step.operator == "project":
                replacement_qdmrs += self.replace_project_sup(qdmr, step_num)
            elif step.operator == "aggregate":
                replacement_qdmrs += self.replace_aggregate_sup(qdmr, step_num)
            elif step.operator == "filter":
                replacement_qdmrs += self.replace_filter_sup(qdmr, step_num)
            elif step.operator == "comparative":
                replacement_qdmrs += self.replace_comparative_sup(qdmr, step_num)
            else:
                continue
        replacement_qdmrs = list(filter(lambda x: x != qdmr, replacement_qdmrs))
        return list(set(replacement_qdmrs))

    def replace_project_sup(self, qdmr, i):
        """replaces project step with project + superlative steps.
        return list with two new qdmrs (min & max), or an empty list if
        phrase is not a superlative phrase.
        E.g. project --> superlative : "youngest #1" -->
                                        "youngest type[#1] of #1"
                                        "#1 where #2 is highest/lowest"
        """
        replacement_qdmrs = []
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        step = builder.steps[i - 1]
        phrase, ref = step.arguments
        phrase = phrase.replace("#REF", "")
        if not is_superlative_phrase(phrase):
            return replacement_qdmrs
        editor = QDMREditor(qdmr)
        select_step = editor.get_step_type_phrase(i)
        project_step = "%s of %s" % (select_step, ref)
        editor.add_new_step(i, select_step)  # editor.add_new_step(i, select_step)
        for agg in ["max", "min"]:
            superlative_step = self.superlative_template(ref, "#%s" % i, agg)
            editor.replace_step(i + 1, superlative_step)
            replacement_qdmrs += [editor.get_qdmr_text()]
        return replacement_qdmrs

    def replace_aggregate_sup(self, qdmr, i):
        """replaces aggregate step with project + superlative steps.
        return list with one new qdmr (min/max), or an empty list if
        aggregate is not min/max.
        E.g. aggregate --> superlative : "largest of #2" -->
                                            "largest type[#2] of #2"
                                            "#2 where #3 is highest"
        """
        replacement_qdmrs = []
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        step = builder.steps[i - 1]
        agg, ref = step.arguments
        if agg not in ["max", "min"]:
            return replacement_qdmrs
        editor = QDMREditor(qdmr)
        sup_phrase = editor.get_step(i).replace(ref, "").strip()
        ref_idx = int(ref.replace("#", ""))
        ref_type = editor.get_step_type_phrase(ref_idx)
        select_step = "%s %s" % (sup_phrase, ref_type)
        project_step = "%s of %s" % (select_step, ref)
        editor.add_new_step(i, project_step)  # editor.add_new_step(i, select_step)
        superlative_step = self.superlative_template(ref, "#%s" % i, agg)
        editor.replace_step(i + 1, superlative_step)
        return [editor.get_qdmr_text()]

    def replace_filter_sup(self, qdmr, i):
        """replaces filter step with project + superlative steps.
        return list with two new qdmrs (min & max), or an empty list if
        phrase is not a superlative phrase.
        E.g. filter --> superlative : "#1 that is the youngest" -->
                                        "type[#1] that is youngest of #1"
                                        "#1 where #2 is highest/lowest"
        """
        replacement_qdmrs = []
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        step = builder.steps[i - 1]
        ref, phrase = step.arguments
        if not is_superlative_phrase(phrase):
            return replacement_qdmrs
        editor = QDMREditor(qdmr)
        select_step = "%s %s" % (editor.get_step_type_phrase(i), phrase)
        project_step = "%s of %s" % (select_step, ref)
        editor.add_new_step(i, project_step)  # editor.add_new_step(i, select_step)
        for agg in ["max", "min"]:
            superlative_step = self.superlative_template(ref, "#%s" % i, agg)
            editor.replace_step(i + 1, superlative_step)
            replacement_qdmrs += [editor.get_qdmr_text()]
        return replacement_qdmrs

    def replace_comparative_sup(self, qdmr, i):
        """replaces comparative step with superlative step.
        return list with two new qdmrs (min & max), or an empty list if
        phrase is not a superlative phrase.
        E.g. comparative --> superlative : "#1 where #2 is tallest" -->
                                            "#1 where #2 is highest/lowest"
        """
        replacement_qdmrs = []
        builder = QDMRProgramBuilder(qdmr)
        builder.build()
        step = builder.steps[i - 1]
        ent_ref, val_ref, phrase = step.arguments
        if not is_superlative_phrase(phrase):
            return replacement_qdmrs
        editor = QDMREditor(qdmr)
        for agg in ["max", "min"]:
            superlative_step = self.superlative_template(ent_ref, val_ref, agg)
            editor.replace_step(i, superlative_step)
            replacement_qdmrs += [editor.get_qdmr_text()]
        return replacement_qdmrs

    def superlative_template(self, entities_ref, values_ref, agg):
        assert agg in ["min", "max"]
        agg_token = "highest" if agg == "max" else "lowest"
        return "%s where %s is %s" % (entities_ref, values_ref, agg_token)


class ColumnGroundingRepair(GroundingRepair):
    """
    Repair grounded SQL by iterating the potential
    column grounding for phrases in select/project/filter steps.
    Currently implemented only for a single phrase grounding repair.

    Also calls the grounding example syntax repairs (if necessary)
    """

    def __init__(self, grounding_test_example):
        super(ColumnGroundingRepair, self).__init__(grounding_test_example)

    def _repair(self):
        ground_assignments = self.build_grounded_assignments(k=20)
        count = 1
        for assignment in ground_assignments:
            try:
                ground = self.grounding_example.ground_example(assignment=assignment)
            except:
                continue
            if self.grounding_example.correct_denotation(distinct=True):
                return True
            else:
                if self.grounding_example.syntax_repairs():
                    return True
        return False

    def extract_step_phrase_groundings(self):
        grounding = self.grounding_example.grounding
        return grounding.steps_phrase_groundings

    def get_top_assignment(self, steps_groundings):
        assignment = {}
        for step_name in steps_groundings.keys():
            top_phrase_ground = steps_groundings[step_name][0][0]
            assignment[step_name] = top_phrase_ground
        return assignment

    def build_grounded_assignments(self, k, bfs_order=True):
        """
        Builds ordered grounding assignments based on the top-k
        column groundings to phrases in steps (select, project, filter).
        Assignments are probed in a BFS order based on the grounded steps
        and candidate column groundings. Implemented s.t. only one candidate
        is replaced per assignment (to minimize assignment num),
        total num of assignments is n*k, where n is num of steps.
        E.g:
            step1-cand2, step2-cand1,..., stepn-cand1
            step1-cand1, step2-cand2,..., stepn-cand1
            step1-cand1, step2-cand1,..., stepn-cand2
            step1-cand3, step2-cand1,..., stepn-cand1
            ...
            step1-cand1, step2-cand1,..., stepn-candk

        Parameters
        ----------
        k : int
            Top-k column groundings used to build assignments
        bfs_order : bool
            Use BFS to order output grounding assignments

        Returns
        -------
        list
            List of grounding assignments (dict from step to column)
        """
        assignments_stack = []
        assert bfs_order
        step_phrase_groundings = self.extract_step_phrase_groundings()
        original_assignment = self.get_top_assignment(step_phrase_groundings)
        for j in range(1, k):
            # BFS order on candidates - steps
            for step_name in step_phrase_groundings.keys():
                step_candidates_num = len(step_phrase_groundings[step_name])
                if j >= step_candidates_num:
                    continue
                new_assignment = original_assignment.copy()
                top_phrase_ground = step_phrase_groundings[step_name][j][0]
                new_assignment[step_name] = top_phrase_ground
                assignments_stack += [new_assignment]
        return assignments_stack

