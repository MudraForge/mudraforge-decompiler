import unittest
import torch

from compiler.neural import inference, literals
from compiler.neural.logic_reasoner import DualStreamStatutoryParser
from compiler.neural.statutory_neural_parser import (
    BIO_TAGS,
    RELATION_TYPES,
    SEMANTIC_ROLES,
    SPAN_UNITS,
    V2_BIO_TAGS,
    V2_RELATION_TYPES,
    V2_SEMANTIC_ROLES,
)


class TestModelV2Commit1(unittest.TestCase):
    def test_literals_reexported_identically_by_inference(self):
        for name in (
            "parse_numeric_quantity",
            "parse_minor_units",
            "parse_rate_bps",
            "parse_iso_date_str",
            "parse_unit_and_type",
            "parse_comparison_operator",
            "segment_clauses",
            "normalize_symbol_id",
            "stitch_classified_spans",
        ):
            self.assertIs(getattr(inference, name), getattr(literals, name))

        self.assertEqual(literals.parse_numeric_quantity("$36,900"), 36900)
        self.assertEqual(literals.parse_minor_units("$5,535"), 553500)
        self.assertEqual(literals.parse_rate_bps("28%"), 2800)

    def test_v1_model_state_dict_unchanged(self):
        v1 = DualStreamStatutoryParser(raw_vocab_size=128)
        sd = v1.state_dict()
        self.assertFalse(any("unit_classifier" in k for k in sd))

        raw_ids = torch.randint(1, 50, (1, 12))
        log_ids = torch.randint(1, 50, (1, 12))
        out = v1(raw_ids, log_ids, span_indices=[(1, 2), (4, 5)])
        self.assertIsNone(out["unit_logits"])
        self.assertEqual(out["role_logits"].shape, (2, len(SEMANTIC_ROLES)))
        self.assertEqual(out["rel_logits"].shape, (2, 2, len(RELATION_TYPES)))
        self.assertEqual(out["bio_logits"].shape, (12, len(BIO_TAGS)))

    def test_v2_model_emits_typed_unit_logits_and_exact_relations(self):
        v1 = DualStreamStatutoryParser(raw_vocab_size=128, num_span_units=0)
        v2 = DualStreamStatutoryParser(
            raw_vocab_size=128,
            num_roles=len(V2_SEMANTIC_ROLES),
            num_relations=len(V2_RELATION_TYPES),
            num_bio_tags=len(V2_BIO_TAGS),
            num_span_units=len(SPAN_UNITS),
        )
        sd_v2 = v2.state_dict()
        self.assertIn("logic_reasoner.unit_classifier.weight", sd_v2)
        self.assertIn("logic_reasoner.unit_classifier.bias", sd_v2)
        unit_head_params = sum(
            p.numel() for name, p in v2.named_parameters() if "unit_classifier" in name
        )
        self.assertEqual(unit_head_params, 128 * len(SPAN_UNITS) + len(SPAN_UNITS))
        self.assertLess(
            sum(p.numel() for p in v2.parameters()) - sum(p.numel() for p in v1.parameters()),
            150000,
        )

        raw_ids = torch.randint(1, 50, (1, 12))
        log_ids = torch.randint(1, 50, (1, 12))
        out = v2(raw_ids, log_ids, span_indices=[(1, 2), (4, 5), (7, 8)])
        self.assertIsNotNone(out["unit_logits"])
        self.assertEqual(out["unit_logits"].shape, (3, len(SPAN_UNITS)))
        self.assertEqual(out["role_logits"].shape, (3, len(V2_SEMANTIC_ROLES)))
        self.assertEqual(out["rel_logits"].shape, (3, 3, len(V2_RELATION_TYPES)))
        self.assertEqual(out["bio_logits"].shape, (12, len(V2_BIO_TAGS)))
        self.assertIn("REL_MIN_CAP", V2_RELATION_TYPES)
        self.assertIn("REL_SUB_CREDIT", V2_RELATION_TYPES)
        self.assertIn("REL_BASE_TAX_ADD", V2_RELATION_TYPES)
        self.assertIn("REL_DEFINED_TERM_OF", V2_RELATION_TYPES)
        self.assertIn("REL_SUBSTITUTE_WHEN", V2_RELATION_TYPES)


if __name__ == "__main__":
    unittest.main()
