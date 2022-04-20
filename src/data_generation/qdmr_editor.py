from utils import *
from qdmr_identifier import *

class QDMREditor(object):
    def __init__(self, qdmr_text):
        self.qdmr_steps = {}
        steps_list = parse_decomposition(qdmr_text)
        for i in range(len(steps_list)):
            self.qdmr_steps[i + 1] = steps_list[i]

    def get_step(self, step_num):
        assert step_num in self.qdmr_steps.keys()
        return self.qdmr_steps[step_num]

    def replace_step(self, step_num, step):
        self.qdmr_steps[step_num] = step

    def add_new_step(self, step_num, step):
        new_steps = {}
        new_steps[step_num] = step
        for i in self.qdmr_steps.keys():
            orig_step = self.qdmr_steps[i]
            if i < step_num:
                new_steps[i] = orig_step
            elif i >= step_num:
                new_steps[i + 1] = self.refs_one_up(orig_step, step_num, len(self.qdmr_steps))
        self.qdmr_steps = new_steps

    def refs_one_up(self, qdmr_text, start_idx, end_idx):
        target_refs_map = {}
        for i in range(start_idx, end_idx + 1):
            target_refs_map["#%s" % i] = "#%s" % (i + 1)
        new_qdmr_step = ""
        for tok in qdmr_text.split():
            if tok in target_refs_map.keys():
                new_qdmr_step += "%s " % target_refs_map[tok]
            else:
                new_qdmr_step += "%s " % tok
        return new_qdmr_step.strip()

    def step_type_phrases(self):
        qdmr_text = self.get_qdmr_text()
        builder = QDMRProgramBuilder(qdmr_text)
        builder.build()
        type_phrases = {}
        for i in range(len(builder.steps)):
            step = builder.steps[i]
            op = step.operator
            if op == "select":
                type_phrases[i + 1] = step.arguments[0]
            elif op == "project":
                ref_phrase, ref_idx = step.arguments
                ref_idx = int(ref_idx.replace("#", ""))
                ref_type = type_phrases[ref_idx]
                type_phrases[i + 1] = ref_phrase.replace("#REF", ref_type)
            elif op in ["filter", "aggregate", "superlative", "comparative", \
                        "sort", "discard", "intersection", "union"]:
                ref_idx = step.arguments[1] if op in ["aggregate", "superlative"] else step.arguments[0]
                ref_idx = int(ref_idx.replace("#", ""))
                type_phrases[i + 1] = type_phrases[ref_idx]
            else:
                type_phrases[i + 1] = None
        return type_phrases

    def get_step_type_phrase(self, step_num):
        type_phrases = self.step_type_phrases()
        return type_phrases[step_num]

    def get_qdmr_text(self):
        qdmr = ""
        for i in range(len(self.qdmr_steps)):
            qdmr += "return %s; " % self.qdmr_steps[i + 1]
        return qdmr.strip()[:-1]
