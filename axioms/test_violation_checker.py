import itertools
from unittest import TestCase

from axioms.violation_checker import RX_XR, RX_RR


class TestRX_XR(TestCase):
    runs = {
        "aa", "ab", "ax",
        "ba", "bb", "bx",
        "xa", "xb", "xx"

    }
    pairs = itertools.product(runs, runs)
    matches = [("ax", "xa"), ("ax", "xb"),
               ("bx", "xb"), ("bx", "xa"), ]

    def test_get_displace_letters(self):
        rule = RX_XR()
        self.assertEqual([0, 1], rule.get_displace_letters("ax", "xa"))
        self.assertEqual([0, 2], rule.get_displace_letters("axxx", "xxax"))
        self.assertEqual([2, 4], rule.get_displace_letters("aaaax", "aaxaa"))
        self.assertEqual([3, 4], rule.get_displace_letters("aaaxa", "aaaax"))
        self.assertEqual([0, 4], rule.get_displace_letters("xaaaa", "aaaax"))
        self.assertEqual([0, 3, 4], rule.get_displace_letters("xaaaa", "aaaxx"))

    def test_match(self):
        rule = RX_XR()
        self.assertEqual(True, rule.match("ax", "xa"))
        self.assertEqual(True, rule.match("axxx", "xxax"))
        self.assertEqual(True, rule.match("aaaax", "aaxaa"))
        self.assertEqual(True, rule.match("aaaxa", "aaaax"))
        self.assertEqual(True, rule.match("xaaaa", "aaaax"))
        self.assertEqual(False, rule.match("xaaaa", "aaaxx"))
        self.assertEqual(False, rule.match("ax", "ax"))
        self.assertEqual(False, rule.match("bx", "ax"))
        self.assertEqual(True, rule.match("bx", "xa"))
        self.assertEqual(False, rule.match("b", "xa"))
        self.assertEqual(False, rule.match("xx", "xa"))

    def test_is_leq(self):
        rule = RX_XR()
        self.assertEqual(False, rule.is_leq("ax", "xa"))
        self.assertEqual(False, rule.is_leq("axxx", "xxax"))
        self.assertEqual(False, rule.is_leq("aaaax", "aaxaa"))
        self.assertEqual(True, rule.is_leq("aaaxa", "aaaax"))
        self.assertEqual(True, rule.is_leq("xaaaa", "aaaax"))
        self.assertEqual(False, rule.is_leq("axxx", "xxax"))
        self.assertEqual(False, rule.is_leq("ax", "ax"))
        self.assertEqual(False, rule.is_leq("bx", "ax"))
        self.assertEqual(False, rule.is_leq("bx", "xa"))
        self.assertEqual(False, rule.is_leq("xx", "xa"))

        self.assertEqual(True, rule.is_leq("xa", "ax"))
        self.assertEqual(True, rule.is_leq("xxax", "axxx"))
        self.assertEqual(True, rule.is_leq("aaxaa", "aaaax"))
        self.assertEqual(True, rule.is_leq("aaaxa", "aaaax"))
        self.assertEqual(True, rule.is_leq("xaaaa", "aaaax"))
        self.assertEqual(True, rule.is_leq("xxax", "axxx"))


class TestRX_RR(TestCase):
    def test_match(self):
        rule = RX_RR()
        self.assertEqual(True, rule.match("ax", "aa"))
        self.assertEqual(True, rule.match("axxx", "xxxx"))
        self.assertEqual(True, rule.match("aaaax", "aaaxx"))
        self.assertEqual(False, rule.match("aaaxx", "aaaa"))
        self.assertEqual(True, rule.match("xaaaa", "aaaaa"))
        self.assertEqual(False, rule.match("ax", "ax"))
        self.assertEqual(False, rule.match("bx", "ax"))
        self.assertEqual(True, rule.match("bx", "ba"))
        self.assertEqual(False, rule.match("b", "xa"))
        self.assertEqual(True, rule.match("xx", "xb"))

    def test_is_leq(self):
        rule = RX_RR()
        self.assertEqual(True, rule.is_leq("ax", "aa"))
        self.assertEqual(False, rule.is_leq("axxx", "xxxx"))
        self.assertEqual(False, rule.is_leq("aaaax", "aaaxx"))
        self.assertEqual(False, rule.is_leq("aaaxx", "aaaa"))
        self.assertEqual(True, rule.is_leq("xaaaa", "aaaaa"))
        self.assertEqual(False, rule.is_leq("ax", "ax"))
        self.assertEqual(False, rule.is_leq("bx", "ax"))
        self.assertEqual(True, rule.is_leq("bx", "ba"))
        self.assertEqual(False, rule.is_leq("b", "xa"))
        self.assertEqual(True, rule.is_leq("xx", "xb"))
