"""Cached statement loader. Reads class_0.txt and class_1.txt ONCE."""
import os
import logging
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)


class StatementCache:
    """Loads and caches general statements from data/general_statements/.

    The original code re-reads these files for every single concept (100+ times).
    This class reads them once and reuses for all concepts.
    """

    def __init__(self, data_dir: str, datasize: str = "single"):
        self._data_dir = os.path.join(data_dir, "general_statements")
        self._datasize = datasize
        self._class0: List[str] = []
        self._class1: List[str] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        def read_lines(fname: str) -> List[str]:
            path = os.path.join(self._data_dir, fname)
            with open(path, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        self._class0 = read_lines("class_0.txt")
        self._class1 = read_lines("class_1.txt")

        if self._datasize in ("double", "triple"):
            self._class0 += read_lines("class_0_a.txt")
            self._class1 += read_lines("class_1_a.txt")
        if self._datasize == "triple":
            self._class0 += read_lines("class_0_b.txt")
            self._class1 += read_lines("class_1_b.txt")

        self._loaded = True
        log.info(
            "Loaded %d class_0 + %d class_1 statements (datasize=%s)",
            len(self._class0), len(self._class1), self._datasize,
        )

    @property
    def class0(self) -> List[str]:
        self._load()
        return self._class0

    @property
    def class1(self) -> List[str]:
        self._load()
        return self._class1

    @property
    def all_statements(self) -> List[str]:
        """All statements from both classes."""
        self._load()
        return self._class0 + self._class1

    def get_unpaired_dataset(
        self,
        concept: str,
        positive_template: str,
        negative_template: str,
        tokenizer,
        seed: int = 0,
    ) -> Dict:
        """Generate unpaired dataset for direction training (stage 1).

        Returns dict with 'inputs' (list of formatted strings) and 'labels' (list of 0/1).
        This replaces all 12 dataset functions in the original datasets.py.
        """
        import random as _random
        rng = _random.Random(seed)

        self._load()
        pos_prompts = []
        neg_prompts = []

        # Positive: concept prefix + class_1 statements
        for stmt in self._class1:
            text = positive_template.format(concept=concept, statement=stmt)
            chat = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pos_prompts.append(formatted)

        # Negative: no prefix + class_0 statements
        for stmt in self._class0:
            text = negative_template.format(statement=stmt)
            chat = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            neg_prompts.append(formatted)

        # Interleave: pos, neg, pos, neg, ...
        inputs = []
        labels = []
        min_len = min(len(pos_prompts), len(neg_prompts))
        for i in range(min_len):
            inputs.append(pos_prompts[i])
            labels.append(1)
            inputs.append(neg_prompts[i])
            labels.append(0)

        return {"inputs": inputs, "labels": labels}

    def get_paired_dataset(
        self,
        concept: str,
        positive_template: str,
        negative_template: str,
        tokenizer,
        seed: int = 0,
    ) -> Dict:
        """Generate paired dataset for attention extraction (stage 0).

        Returns dict with 'inputs' (list of (pos_str, neg_str) tuples) and 'labels'.
        """
        self._load()
        pairs = []
        labels = []

        all_stmts = self._class0 + self._class1
        for stmt in all_stmts:
            pos_text = positive_template.format(concept=concept, statement=stmt)
            neg_text = negative_template.format(statement=stmt)
            pos_chat = [{"role": "user", "content": pos_text}]
            neg_chat = [{"role": "user", "content": neg_text}]
            pos_formatted = tokenizer.apply_chat_template(pos_chat, tokenize=False, add_generation_prompt=True)
            neg_formatted = tokenizer.apply_chat_template(neg_chat, tokenize=False, add_generation_prompt=True)
            pairs.append((pos_formatted, neg_formatted))
            labels.append(1)

        return {"inputs": pairs, "labels": labels}
