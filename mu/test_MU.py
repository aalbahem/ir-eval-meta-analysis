import csv
import os
from unittest import TestCase

from math import log

from mu import MU, to_dict


class TestMU(TestCase):

    T = {"t1"}
    R = {"s1","s2","s3"}
    M = {"m1","m2","m3"}
    S_M = {
        "s1":{"t1":{"m1":1,"m2":0.8,"m3":1}},
        "s2":{"t1":{"m1":0.5,"m2":0.3,"m3":0.2}},
        "s3":{"t1":{"m1":0.2,"m2":0.4,"m3":0.5}}
        }



    def test_pm_ij(self):
        m="m1"
        mu = MU(self.T,self.R, self.M, self.S_M)
        mu.delta_m(m)
        self.assertEqual(3.0/6,mu.pm_ij(m))


    def test_pM_ij(self):
        m="m1"
        M=["m2","m3"]

        mu = MU(self.T,self.R, self.M, self.S_M)
        mu.delta_M(m,M)
        self.assertEqual(3.0 / 6,mu.pM_ij(m))

    def test_pmM_ij(self):
        m="m1"
        M=["m2","m3"]
        mu = MU(self.T,self.R, self.M, self.S_M)
        mu.delta_m(m)
        mu.delta_M(m,M)
        mu.delta_mM(m,M)
        self.assertEqual(2.0 / 6,mu.pmM_ij(m))

    def test_pmi(self):
        m = "m1"
        M = ["m2", "m3"]
        mu = MU(self.T,self.R, self.M, self.S_M)
        mu.delta_m(m)
        mu.delta_M(m, M)
        mu.delta_mM(m,M)
        self.assertEqual(log((2.0 / 6)/((3.0/6.0) * (3.0 / 6) ),2),mu.pmi(m))

    def test_delta_m(self):
        self.skipTest("No implemented")

    def test_delta_M(self):
        self.skipTest("No implemented")

    def test_mu(self):
        m = "m1"
        mu = MU(self.T, self.R, self.M, self.S_M)
        pmis=mu.mu()
        self.assertEqual(log((2.0 / 6) / ((3.0 / 6.0) * (3.0 / 6)), 2), pmis[m])
