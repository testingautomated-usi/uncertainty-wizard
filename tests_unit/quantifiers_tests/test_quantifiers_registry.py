from unittest import TestCase

from uncertainty_wizard.quantifiers import QuantifierRegistry, VariationRatio


class TestMutualInformation(TestCase):

    # Note that the correct registering of all default quantifiers is tested in the corresponding quantifiers tests

    def test_error_if_invalid_quantifier_alias(self):
        with self.assertRaises(ValueError):
            QuantifierRegistry.find("nonexistent_q_hi1ö2rn1ld")

    def test_error_if_alias_already_exists(self):
        with self.assertRaises(ValueError):
            QuantifierRegistry.register(VariationRatio())
