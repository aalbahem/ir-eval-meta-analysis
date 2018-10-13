import os
import re

import itertools

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def xor(a, b):
    return (a and not b) or (not a and b)


class Rule(object):
    name = "NA"


    def sign(self, x, z):
        """
          The implementation should return -, <=, >=, <, >, or =
        """
        return "-"

    @staticmethod
    def same_r(y):
        if Rule.is_empty(y):
            return False

        y_0 = y[0]
        is_all_same_r = len(y) == y.count(y_0)
        return is_all_same_r

    @staticmethod
    def has_x(y):
        return y.count("x") > 0

    @staticmethod
    def has_r(y,r):
        return y.count(r) > 0

    @staticmethod
    def is_empty(y):
        return len(y) == 0

    @staticmethod
    def is_same(y, z):
        return y == z

    @staticmethod
    def eqaul_len(y, z):
        return len(y) == len(z)

    @staticmethod
    def end_with_x(y):
        if Rule.is_empty(y):
            return False
        return y[-1]=="x"

    @staticmethod
    def is_repeated_x(y):
        if not Rule.has_x(y):
            return False
        x_n = y.count("x")
        for i in range(0,x_n):
            if (y[y.find("x")+i]!="x"):
                return False

        return True

    @staticmethod
    def is_repeated_r(y,r):
        if not Rule.has_r(y,r):
            return False
        r_n = y.count(r)

        for i in range(0, r_n):
            if (y[y.find(r) + i] != r):
                return False

        return True

    @staticmethod
    def is_diverse(y):
        if Rule.is_empty(y):
            return False

        no_x = y.replace("x","")
        if len(no_x) <=1:
            return False;

        return no_x.count(no_x[0])!=len(no_x)

    @staticmethod
    def same_aspects(y,z):
        y_a = set(y)
        z_a = set(z)

        return len(y_a) == len(z_a) and len(y_a.intersection(z_a)) == len(y_a)

    @staticmethod
    def aspects(y):
        y_a = set(y)

        return list(y_a)

    @staticmethod
    def same_start(y,z):
        s = Rule.LCP(y,z)
        return len(s) > 0 and y.startswith(s) and z.startswith(s)

    @staticmethod
    def LCS(X, Y):
        m = len(X)
        n = len(Y)
        # An (m+1) times (n+1) matrix
        C = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    C[i][j] = C[i - 1][j - 1] + 1
                else:
                    C[i][j] = max(C[i][j - 1], C[i - 1][j])
        return C

    @staticmethod
    def LCP(y, z):
            for j, k in zip(y, z):
                if j != k:
                    break
                yield j

    @staticmethod
    def diversity_index(y):
        for i in range(1,len(y)):
            if y[i] !=y[i-1]:
                return i;



class NA(Rule):
    name = "NA"

    def sign(self, y, z):
        return "-"


class Path(Rule):
    name = "path"

    def __init__(self,G):
        self.G = G

    def match(self,y, z):
        y_z_path =nx.has_path(self.G,y,z)
        z_y_path = nx.has_path(self.G,z,y)
        return y_z_path or z_y_path

    def sign(self, y, z):
        is_match = self.match(y,z)
        has_path = nx.has_path(self.G,y,z)
        return is_match and has_path



class _R(Rule):
    name = "_R"

    @staticmethod
    def match(y, z):
        is_empty = xor(len(y) == 0, len(z) == 0)

        y_x_count = str(y).count("x")
        z_x_count = str(z).count("x")

        is_all_R = y_x_count == 0 and z_x_count == 0
        return is_empty and is_all_R

    def sign(self, y, z):

        is_match = _R.match(y,z)

        if len(y) == 0 and len(z)==0:
            return False

        return is_match and len(y) ==0

class _RX(Rule):
    name = "<Rx"

    @staticmethod
    def match(y, z):
        is_empty = xor(len(y) == 0, len(z) == 0)

        first_is_r = (len(y) != 0 and y[0]!="x") or (len(z) != 0 and z[0]!="x")
        end_x = xor(Rule.end_with_x(y), Rule.end_with_x(z))
        return is_empty and first_is_r and end_x

    def sign(self, y, z):

        is_match = _RX.match(y,z)

        return is_match and Rule.is_empty(y)

class SX_S(Rule):
    name = "Sx<S"

    @staticmethod
    def match(y, z):
        equal = len(y) == len(z)

        if equal:
            return False

        # examtly one extra
        y_n = len(y)
        z_n = len(z)
        d_n = y_n - z_n

        if abs(d_n) != 1:
            return False

        s_x, s = y, z
        if len(z) > len(y):
            s_x, s = z, y

        one_x_only = len(s_x) == len(s) +1


        s_d = s_x.replace(s, "")

        # n = len(s_d)
        # x_n = re.compile("[x]{{{0},{0}}}".format(1))
        # if (not equal) and s_x.startswith(s) and x_n.match(s_d):
        #     return True


        if (not equal) and one_x_only and s_x.startswith(s) and s_x[-1] == "x":
            return True

        return False

    def sign(self, sx, s):
        is_match = SX_S.match(sx, s)
        return is_match and len(sx)>len(s)

class Nx_Mx(Rule):
        name = "Nx_Mx"

        @staticmethod
        def match(y, z):
            same = Rule.is_same(y,z)
            same_size = Rule.eqaul_len(y,z)
            end_x = Rule.end_with_x(y) and Rule.end_with_x(z)

            return  (not same) and same_size and end_x

        def sign(self, y, z):
            is_match = Nx_Mx.match(y, z)
            less = is_less(y[0:len(y)-1],z[0:len(z)-1])
            return is_match and less

class XS_S(Rule):
            name = "xS<S"

            @staticmethod
            def match(y, z):
                equal = len(y) == len(z)

                if equal:
                    return False

                # examtly one extra
                y_n = len(y)
                z_n = len(z)
                d_n = y_n - z_n

                if abs(d_n) != 1:
                    return False

                x_s, s = y, z
                if len(z) > len(y):
                    s_x, s = z, y

                one_x_only = len(x_s) == len(s) + 1

                if (not equal) and one_x_only and x_s.endswith(s) and x_s[0] == "x":
                    return True

                return False

            def sign(self, xs, s):
                is_match = XS_S.match(xs, s)
                return is_match and len(xs) > len(s)


#
# class SXn_XnS(Rule):
#     name = "SX_XS"
#
#     @staticmethod
#     def match(y, z):
#         equal = len(y) == len(z)
#
#         if not equal:
#             return False
#
#         l = len(y)
#
#         x_n = y.count("x")
#         if (x_n ==0 or x_n !=z.count("x")):
#             return
#
#         r_n = l-x_n
#         xnr_req = "x{{{},{}}}[^x]{{{},{}}}".format(x_n,x_n,r_n,r_n)
#         nrx_req = "[^x]{{{},{}}}x{{{},{}}}".format(r_n,r_n,x_n,x_n)
#
#
#         return (re.match(xnr_req,y) !=None and re.match(nrx_req,z)!=None) or (re.match(xnr_req,y)!=None and re.match(nrx_req,z)!=None)
#
#
#
#     def is_less(self, xs, sx):
#         is_match = SXn_XnS.match(xs, sx)
#
#         verdict = (xs[0]=="x")
#         return is_match and verdict

class RX_XR(Rule):
    name = "xR<Rx"

    @staticmethod
    def match(y, z):
        equal = len(y) == len(z)

        if not equal or y ==z:
            return False

        l = len(y)

        x_n = y.count("x")
        if (x_n ==0 or x_n !=z.count("x") or l == x_n):
            return

        r_n = l - x_n
        y_r = y.replace("x","")
        z_r = z.replace("x","")
        same_r =y.count(y_r[0]) == r_n and  z.count(y_r[0]) == r_n

        repeated_x = Rule.is_repeated_x(y) and Rule.is_repeated_x(z)
        return same_r and repeated_x

        # r_n = l-x_n
        # xnr_req = "x{{{},{}}}[^x]{{{},{}}}".format(x_n,x_n,r_n,r_n)
        # nrx_req = "[^x]{{{},{}}}x{{{},{}}}".format(r_n,r_n,x_n,x_n)
        #
        #
        # return (re.match(xnr_req,y) !=None and re.match(nrx_req,z)!=None) or (re.match(xnr_req,y)!=None and re.match(nrx_req,z)!=None)



    def sign(self, y, z):
        is_match = RX_XR.match(y, z)

        if (not is_match):
            return False
        x_n = y.count("x")
        p_y, p_z = y.find("x"),z.find("x")
        verdict = (p_y < p_z)
        while (not verdict):
            y = str(y).replace("x","",1)
            z = str(z).replace("x","",1)
            p_y = y.find("x")
            p_z = z.find("x")
            verdict = (p_y < p_z)
            if  (p_y == -1):
                break
        # while  < z.find("x")
        # xs.fin
        # for x in
        return is_match and verdict





# class RnR_XRR(Rule):
#     name = "R"
#
#     @staticmethod
#     def match(y, z):
#         equal = len(y) == len(z)
#
#         if not equal:
#             return False
#
#         l = len(y)
#
#         x_m = y.count("x")
#
#         if (x_m !=z.count("x")):
#             return False
#
#         r_n = l-x_m
#
#         Rn1Xm1_req = "[^x]{{{},{}}}x[^x]{{{},{}}}".format(r_n - 1, r_n - 1, x_m, x_m)
#         RnXm_req = "[^x]{{{},{}}}x{{{},{}}}".format(r_n,r_n,x_m,x_m)
#
#         y_match_Rn1Xm1 = re.match(Rn1Xm1_req,y)
#         z_match_RnXm  = re.match(RnXm_req,z)
#
#         y_match_RnXm = re.match(RnXm_req,y)
#         z_match_Rn1Xm1  = re.match(Rn1Xm1_req,z)
#
#         return (y_match_Rn1Xm1 != None and z_match_RnXm !=None) or (y_match_RnXm != None and z_match_Rn1Xm1 !=None)
#
#
#
#     def is_less(self, xs, sx):
#         is_match = RXR_XRR.match(xs, sx)
#
#         verdict = (xs[0]=="x")
#         return is_match and verdict
#


# # class RXR_XRR(Rule):
# #     name = "RXR_XRR"
# #
# #     @staticmethod
# #     def match(y, z):
# #         equal = len(y) == len(z)
# #
# #         if not equal:
# #             return False
# #
# #         l = len(y)
# #
# #         x_m = y.count("x")
# #
# #         if (x_m !=z.count("x") or x_m ==0):
# #             return False
# #         repeated_x_reg = "[x]{{{},{}}}".format(x_m,x_m);
# #         repeated_y = re.match(repeated_x_reg,y)
# #         repeated_z = re.match(repeated_x_reg,z)
# #         repeated_x=   repeated_y!=None and repeated_z!=None
# #
# #         if not repeated_x:
# #             return False
# #
# #         r_n = l-x_m
# #
# #         # "xbb", "bxb"
# #         RnXm_req   = "[^x]{{{},{}}}[x]{{{},{}}}[^x]{{{},{}}}".format(0, r_n, 0, x_m,0,r_n)
# #         Rn1Xm1_req = "[^x]{{{},{}}}[x]{{{},{}}}[^x]{{{},{}}}".format(0, r_n-1, 0, x_m+1,0,r_n-1)
# #
# #
# #         y_match_Rn1Xm1 = re.match(Rn1Xm1_req,y)
# #         z_match_RnXm  = re.match(RnXm_req,z)
# #
# #         y_match_RnXm = re.match(RnXm_req,y)
# #         z_match_Rn1Xm1  = re.match(Rn1Xm1_req,z)
# #
# #         return (y_match_Rn1Xm1 != None and z_match_RnXm !=None) or (y_match_RnXm != None and z_match_Rn1Xm1 !=None)
#
#
#
#     def is_less(self, xs, sx):
#         is_match = RXR_XRR.match(xs, sx)
#
#         verdict = (xs[0]=="x")
#         return is_match and verdict

class S_S(Rule):
    name = "S=S"

    @staticmethod
    def match(y, z):
        equal = y == z

        return equal

    def sign(self, y, z):
        is_match = S_S.match(y, z)

        #S is equal or less to itself.
        return is_match and False

class S_SR(Rule):
    name = "S<Sr"

    @staticmethod
    def match(y, z):

        s, s_r = y, z
        if len(y) > len(z):
            s, s_r = z, y

        one_r_only = len(s) == len(s_r)-1

        return one_r_only and s_r.startswith(s) and s_r[-1] !="x"

    def sign(self, s, sr):
        is_match = S_SR.match(s, sr)
        return is_match and len(s) < len(sr)

class P_NN(Rule):
    name = "Sn=Sn"

    @staticmethod
    def match(y, z):
        equal = len(y) == len(z)

        if not equal or len(y)==0 or len(z) ==0:
            return False
        n = len(y)

        same_parent = y[0:n - 1] == z[0:n - 1]

        if not same_parent:
            return False

        p = y[0:n - 1]
        r1 = y[n - 1]
        r2 = z[n - 1]

        is_new_new = p.find(r1) == -1 and p.find(r2) == -1  # if the parent does not have both of them.
        is_not_x = not (r1 =="x" or r2 =="x")
        return equal and same_parent and is_new_new and is_not_x

    def sign(self, pr, pn):
        is_match = P_NN.match(pr, pn)

        n = len(pr)
        p = pr[0:n - 1]
        r1 = pr[n - 1]
        r2 = pn[n - 1]

        # if both contain novel data, then they are equal
        verdict = False

        return is_match and verdict


class P_RN(Rule):
    name = "Sr<Sn"

    @staticmethod
    def match(y, z):
        equal = len(y) == len(z)

        if not equal or len(y) ==0 or len(z) ==0:
            return False
        n = len(y)

        same_parent = y[0:n - 1] == z[0:n - 1]

        if not same_parent:
            return False

        p = y[0:n - 1]
        r1 = y[n - 1]
        r2 = z[n - 1]

        is_new = xor(p.find(r1) == -1, p.find(r2) == -1)  # if the parent does not have one of them.
        is_not_x = not (r1 == "x" or r2 == "x")

        return equal and same_parent and is_new and is_not_x

    def sign(self, pr, pn):
        is_match = P_RN.match(pr, pn)

        n = len(pr)
        p = pr[0:n - 1]
        r1 = pr[n - 1]
        r2 = pn[n - 1]

        # if a pn is more diverse as it contains the new bit, then pr is defintlly worse
        verdict = p.find(r2) == -1

        return is_match and verdict

class RRR_RNM(Rule):
            name = "RRR_RNM"

            @staticmethod
            def match(y, z):
                same = Rule.is_same(y, z)
                same_size = Rule.eqaul_len(y, z)
                any_zero = Rule.is_empty(y) or Rule.is_empty(z)
                same_r = Rule.same_r(y) or Rule.same_r(z)
                has_x = Rule.has_x(y) or Rule.has_x(z)
                is_diverse = xor(Rule.is_diverse(y), Rule.is_diverse(z))

                if (not same_size) or any_zero or (not same_r) or same or has_x:
                    return False

                return True and is_diverse

            def sign(self, y, z):
                is_match = RRR_RNM.match(y, z)

                rrr, rnm = y, z

                if (Rule.same_r(z)):
                    rrr, rnm = z, y

                return is_match and rrr == y

class RM_MR(Rule):
        name = "rrnn< rnnr"

        @staticmethod
        def match(y, z):
            equal_length = Rule.eqaul_len(y,z)
            has_x = Rule.has_x(y) or Rule.has_x(z)

            if not equal_length or y == z or  has_x:
                return False

            l = len(y)

            aspects = Rule.aspects(y)
            same_aspets = Rule.same_aspects(y,z)
            only_2_aspects = len(aspects) == 2

            if not (only_2_aspects and equal_length and same_aspets):
                return False


            r, x = aspects[0], aspects[1]
            x_n = y.count(x)
            r_n = y.count(r)

            if x_n ==r_n and Rule.is_repeated_r(y,x) and Rule.is_repeated_r(y,r) and Rule.is_repeated_r(z,x) and Rule.is_repeated_r(z,r):
                return False

            if not (Rule.is_repeated_r(y,x) and Rule.is_repeated_r(z,x)):
                r_n, x_n = x_n, r_n
                r, x = x, r

            if (x_n == 0 or x_n != z.count(x) or l == x_n):
                return

            r_n = l - x_n
            y_r = y.replace(x, "")
            z_r = z.replace(x, "")
            same_r = y.count(y_r[0]) == r_n and z.count(y_r[0]) == r_n

            repeated_r = Rule.is_repeated_r(y,x) and Rule.is_repeated_r(z,x)
            return same_r and repeated_r

            # same = Rule.is_same(y,z)
            # same_aspects = Rule.same_aspects(y,z)
            # same_size = Rule.eqaul_len(y,z)
            # any_zero = Rule.is_empty(y) or Rule.is_empty(z)
            # has_no_x =  not (Rule.has_x(y) or Rule.has_x(z))
            # is_diverse = Rule.is_diverse(y) and Rule.is_diverse(z)
            #
            # aspects = Rule.aspects(y)
            # only_2_aspects = len(aspects) ==2
            #
            # if not (only_2_aspects or len(y) > 2):
            #     return False
            # a,b = list(aspects)[0],list(aspects)[1]
            # a_n,b_m = y.count(a), y.count(b)
            # a_n_reg = re.compile("({}){{{},{}}}({}){{},{}}".format(a,a_n,a_n,b,b_m,b_m))
            # a_n_1_reg = re.compile("({}){{{},{}}}({}){{},{}}".format(a, a_n, a_n, b, b_m, b_m)),
            # # for a in aspects:
            # #     if z.count(a) != y.count(a):
            # #         return False
            #
            # return (not same) and same_aspects and same_size and (not any_zero) and has_no_x and is_diverse and only_2_aspects

        # def is_less(self, y, z):
        #     is_match = RM_MR.match(y, z)
        #
        #     rnmn, rm = y, z
        #
        #     if (Rule.same_r(z)):
        #         rnmn, rm = z, y
        #
        #     return is_match and rnmn, rm

        def sign(self, y, z):
            is_match = RM_MR.match(y, z)

            if (not is_match):
                return False
            aspects = list(Rule.aspects(y))

            r, x = aspects[0], aspects[1]
            x_n = y.count(x)
            r_n = y.count(r)

            if not (Rule.is_repeated_r(y, x) and Rule.is_repeated_r(z, x)):
                r_n, x_n = x_n, r_n
                r, x = x, r


            p_y, p_z = y.find(x), z.find(x)
            verdict = (p_y > p_z)
            while p_y ==p_z:
                y = str(y).replace(x, "", 1)
                z = str(z).replace(x, "", 1)
                p_y = y.find(x)
                p_z = z.find(x)
                verdict = (p_y > p_z)
                if (p_y == -1 or p_z > -1):
                    break

            verdict = Rule.diversity_index(y) > Rule.diversity_index(z)



            return is_match and verdict

class RRN_RNR(Rule):
            name = "rrnn<rnrn"

            @staticmethod
            def match(y, z):

                same = Rule.is_same(y,z)
                equal_length = Rule.eqaul_len(y,z)
                same_aspects = Rule.same_aspects(y,z)
                same_size = Rule.eqaul_len(y,z)
                any_zero = Rule.is_empty(y) or Rule.is_empty(z)
                has_no_x =  not (Rule.has_x(y) or Rule.has_x(z))
                is_diverse = Rule.is_diverse(y) and Rule.is_diverse(z)

                aspects = Rule.aspects(y)

                only_2_aspects = len(aspects) ==2

                eqaul_aspects_lenght = True


                for a in aspects:
                    if y.count(a) != z.count(a):
                        eqaul_aspects_lenght = False
                        break;



                if not (only_2_aspects and same_aspects and eqaul_aspects_lenght and same_size and equal_length and (not same) and (not any_zero) and has_no_x and is_diverse):
                    return False

                a, b = aspects[0], aspects[1]
                same_start = y.startswith(a) and z.startswith(a)

                if not same_start:
                    return False

                a_n,b_m = y.count(a), y.count(b)
                rrnn_reg = re.compile("{}{{{},{}}}{}{{{},{}}}".format(a,a_n,a_n,b,b_m,b_m))


                if  (rrnn_reg.match(y) ==None  and rrnn_reg.match(z)==None):
                    return False
                else:
                    return True

                rnrr_reg = re.compile("({}){{{},{}}}({})+".format(a, 1, a_n - 1, b))
                rrnn, rnrr = y, z


                return

            def sign(self, y, z):
                is_match = RRN_RNR.match(y, z)

                aspects = Rule.aspects(y)

                a, b = aspects[0],aspects[1]
                a_n, b_m = y.count(a), y.count(b)
                rrnn_reg = re.compile("({}){{{},{}}}({}){{{},{}}}".format(a, a_n, a_n, b, b_m, b_m))

                verdict = rrnn_reg.match(y) != None

                return is_match and verdict


# def anb_an_1ba(anb="", an_1ba=""):
#     """ Valid if <b>y</b> is part of <b>z</b> and the difference is are all relevant documents """
#
#     if len(anb) != len(an_1ba):
#         return False
#
#     if anb.find("x") > -1 or an_1ba.find("x") > -1:
#         return False
#     l = len(anb)
#
#     r_anb = "a{{{},{}}}b".format(l - 1, l - 1)
#     r_an_1ba = "a{{{},{}}}ba".format(l - 2, l - 2)
#     if re.match(r_anb, anb) and re.match(r_an_1ba, an_1ba):
#         print ("Matching anb_an_1ba")
#         return True
#
#     return False
#
# def xn_an_1ba(anb="", an_1ba=""):
#     """ Valid if <b>y</b> is part of <b>z</b> and the difference is are all relevant documents """
#
#     if len(anb) != len(an_1ba):
#         return False
#
#     if anb.find("x") > -1 or an_1ba.find("x") > -1:
#         return False
#     l = len(anb)
#
#     r_anb = "a{{{},{}}}b".format(l - 1, l - 1)
#     r_an_1ba = "a{{{},{}}}ba".format(l - 2, l - 2)
#     if re.match(r_anb, anb) and re.match(r_an_1ba, an_1ba):
#         print ("Matching anb_an_1ba")
#         return True
#
#     return False



def get_pre_graph_rules():
    # return [_R.name,_RX.name, S_S.name, S_SR.name, SX_S.name,XS_S.name, P_RN.name, P_NN.name,RRR_RNM.name, RX_XR.name,RM_MR.name,RRN_RNR.name]
    return [S_SR.name, SX_S.name, P_RN.name, ]


def get_graph_independent_matching_rules(y, z):
    """ Get which rules that match the two string"""


    rules = []
    # if SXn_XnS.match(y, z):
    #     return SXn_XnS()


    # xnrn,rn = y,z
    # if len(rn) > len(xnrn):
    #     xnrn, rn = z,y
    #     return False
    #
    # s_d = xnrn.replace(rn, "")
    # if xnrn.endswith(rn) and r_n.match(s_d):
    #     return "xn_xnrn"

    # if (RXR_XRR.match(y, z)):
    #     rules.append( RXR_XRR()

    # if P_XR.match(y, z):
    #     rules.append( P_XR()


    # if S_S.match(y, z):
    #     rules.append( S_S())

    # if S_SR.match(y, z):
    #     rules.append( S_SR())
    # #
    # if SX_S.match(y, z):
    #     rules.append( SX_S())
    #
    # if XS_S.match(y, z):
    #     rules.append(XS_S())
    #
    # if P_NN.match(y, z):
    #     rules.append( P_NN())
    #
    if P_RN.match(y, z):
        rules.append( P_RN())
    #
    # if RRR_RNM.match(y, z):
    #     rules.append(RRR_RNM())
    #
    # if RX_XR.match(y, z):
    #     rules.append( RX_XR())
    #
    # if RM_MR.match(y, z):
    #     rules.append(RM_MR())
    #
    # if RRN_RNR.match(y, z):
    #     rules.append(RRN_RNR())
    #
    #
    # if _R.match(y, z):
    #     rules.append(_R())
    #
    # if _RX.match(y, z):
    #     rules.append(_RX())

    if len(rules) ==0:
        rules.append(NA());
    return rules

def get_incremental_matching_rules(y, z):
    """ Get which rules that match the two string"""


    rules = []

    if S_SR.match(y, z):
        rules.append( S_SR())

    if SX_S.match(y, z):
        rules.append( SX_S())
    #

    if P_RN.match(y, z):
        rules.append( P_RN())

    if len(rules) ==0:
        rules.append(NA());
    return rules

def get_matching_rules(matching_function,nodes):
    return matching_function(*nodes)


def is_less(y, z):
    rules = get_graph_independent_matching_rules(y, z)

    for rule in rules:
     if rule.sign(y, z):
         return True

    return False




def validate_metrics(df, metrics=[], runs=[], M=3,print_na=False, print_violation_only=False,metric_prefix="",one_branch_only=True):
    """
      Evaluate the metrics using the Rule validations
    """

    avg_df = df[df["topic"] == "all"]

    metrics_rules_violation = {}

    G,E = build_fast_graphs(["a", "b"], M)
    g_path_rule = Path(G)
    e_path_rule = Path(E)

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")

    metrics_rules_viloations = {}
    metrics_rules_matches = {}

    rules_names = [SX_S.name,S_SR.name,P_RN.name]+[Path.name,NA.name]
    for m in metrics:
        metrics_rules_viloations[m] = {}
        metrics_rules_matches[m] = {}
        for r in rules_names:
            metrics_rules_viloations[m][r] = 0
            metrics_rules_matches[m][r] = 0

    runs_metrics_scores = {}



    for name, group in avg_df.groupby("run"):
        runs_metrics_scores[name] = {}
        records = pd.DataFrame.to_dict(group,orient='records')
        runs_metrics_scores[name]=  records


    if len(runs) == 0:
            runs = runs_metrics_scores.keys()
    a_runs = [run for run in runs if str(run).startswith("a")]

    if (one_branch_only):
        runs = a_runs


    count=0
    na_count = 0
    for y, z in itertools.combinations(runs,2):
                rules = get_incremental_matching_rules(y, z)

                count+=1
                print (count)
                for rule in rules:
                    if (isinstance(rule,NA)):
                        if (g_path_rule.match(y,z)):
                            rule = g_path_rule
                        # elif (e_path_rule.match(z,y)):
                        #     rule = e_path_rule
                        else:
                            na_count += 1
                            continue
                    for metric in metrics:
                        metrics_rules_matches[metric][rule.name] += 1
                        expect = rule.sign(y, z)
                        yscore = runs_metrics_scores[y][0][metric]
                        zscore = runs_metrics_scores[z][0][metric]
                        actual = yscore <= zscore

                        not_equal = (yscore != zscore)

                        if xor(expect, actual) and not_equal:
                            metrics_rules_viloations[metric][rule.name] += 1
                            print (
                            "{}, {}({} , {}) = {} => ({}'s score: {:.4f}, {}'s score: {:.4f}, expected:{}, actual:{})".format(
                                metric, rule.name, y, z, xor(expect, actual), y, yscore, z, zscore, expect, actual))

    print ("{},{},{},{}".format("metric", "rule", "matches", "violation"))


    for m in metrics:
        for r in rules_names:
            print_line = (not print_violation_only or metrics_rules_viloations[m][r] > 0)
            if print_line:
                line = "{},{},{},{}".format(metric_prefix+m, r, metrics_rules_matches[m][r], metrics_rules_viloations[m][r])
                print (line)

    columns = ["Metric"]
    total = 0
    tab_count = 0
    rules_pretty_names = {NA.name:"NA",
                          S_SR.name:"$\metric{m}(S)\leq\metric{m}(S{\cdot}r)$",
                          SX_S.name:"$\metric{m}(S{\cdot}x)\leq\metric{m}(S)$",
                          P_RN.name:"$\metric{m}(S{\cdot}p) \leq \metric{m}(S{\cdot}n)$",
                          Path.name:"Induction"
                          }
    rules_names.remove(NA.name)
    for name in rules_names:
        tab_count+=1
        total += metrics_rules_matches[metrics[0]][name]
        cell = "\n{} {} ({})".format("\t" * tab_count,rules_pretty_names[name], metrics_rules_matches[metrics[0]][name])
        columns.append(cell)
    tab_count += 1
    columns.append("Total ({})".format(total))


    print (" & ".join(columns))
    for m in metrics:
        violation_total = 0
        rules_v = [metric_prefix+m]
        tab_count = 0
        for r in rules_names:
            tab_count+=1
            violation_total += metrics_rules_viloations[m][r]
            cell = "{} & {}".format("\t" * tab_count, str(metrics_rules_viloations[m][r]))
            rules_v.append(cell)
        tab_count += 1
        cell = "{} & {}".format("\t" * tab_count, str(violation_total))
        rules_v.append(cell)
        print ("\n".join(rules_v) + "\\\\")

    return


def print_coverage_table(a, M, print_na=False):
    A = a + ["x"]

    seen = []
    runs = [""]  # empty set


    # G,E = buildGraphs(a, M)
    G, E = build_fast_graphs(a, M)
    l_path_rule = Path(G)
    r_path_rule = Path(E)
    for m in range(1, M + 1):
        _runs = [r for r in runs]
        for run in _runs:
            for x in A:
                runs.append(run + x)

    runs = set(runs)

    rules_names = get_pre_graph_rules()+[NA.name,Path.name]
    rules_matches = {}
    for r in rules_names:
        rules_matches[r] = []

    for y in runs:
        for z in runs:
            rules = get_graph_independent_matching_rules(y, z)

            for rule in rules:
                if (isinstance(rule,NA)):
                    if (l_path_rule.match(y,z)):
                        rule = l_path_rule
                    elif (r_path_rule.match(y,z)):
                        rule = r_path_rule
                    if isinstance(rule,NA) and print_na:
                        print("NA-> ({}) ({})".format(y, z))

                rules_matches[rule.name].append((y,z))


    for rule in rules_matches:
        print (rule)
        print(rules_matches[rule])

    for rule in rules_matches:
        print (rule +": " + str(len(rules_matches[rule])))

    rules = [rule for rule in rules_matches]

    print (",".join(["rule/rule"]+rules))
    for  r in rules:
        r_line = [r]
        for r2 in rules:
              r_cases  = set(rules_matches[r])
              r2_cases = set(rules_matches[r2])
              if (r ==r2):
                  r_line.append("-")
              else:
                  r_line.append(str(len(r_cases.intersection(r2_cases))))

        print (",".join(r_line))

def buildGraphs(a, M):
   g,e=build_bi_graphs(generate_runs(a,M))
   return g,e

def generate_runs(a,M):
    A = a + ["x"]

    runs = [""]  # empty set

    level_nodes = []
    level_nodes.append([""])  # the first node is an empty node.
    for m in range(1, M + 1):
        level_nodes.append([])
        for node in level_nodes[m - 1]:
            for x in A:
                child = node + x
                level_nodes[m].append(child)
                runs.append(child)
    return runs

def addPrimaryEges(G,E_G,runs):
    edges = []

    na_nodes = []
    count = 0
    seen_y = []
    eqaul_edges = []
    for y in runs:
        if y in seen_y:
            continue
        seen_y.append(y)
        for z in runs:
            rules = get_graph_independent_matching_rules(y, z)
            count += 1
            print (str(count))
            for rule in rules:
                if isinstance(rule, NA):
                    if len(rules) != 1:
                        raise Exception("NA can not be associated with othe rules")

                    na_nodes.append((y,z))
                    continue

                verdict = rule.sign(y, z)
                reversed_verdict = rule.sign(z, y)

                if not xor(verdict, reversed_verdict):
                    if not (isinstance(rule, S_S) or isinstance(rule,P_NN)):
                        raise Exception("A rule can not produce same ouput")
                    else:
                        eqaul_edges.append((y,z))
                        continue

                if verdict:
                    G.add_edge(y, z)
                    if "-".join([y, z]) in edges or "-".join([z, y]) in edges:
                        print("Something went wrong, two edges are added")
                    else:
                        edges.append("-".join([y, z]))

                    break

    E_G = G.copy()
    acyclic_graph = nx.is_directed_acyclic_graph(G)
    acyclic_graph_e_g = nx.is_directed_acyclic_graph(E_G)

    if not (acyclic_graph and acyclic_graph_e_g):
        # G.remove_edge(y,z)
        raise Exception("We can not have cycles")

    return G,E_G,na_nodes


def build_fast_graphs(a,M):

    G = nx.DiGraph()

    A = a + ["x"]

    runs = [""]  # empty set

    level_nodes = []
    level_nodes.append([""])  # the first node is an empty node.
    for m in range(1, M + 1):
        level_nodes.append([])
        for node in level_nodes[m - 1]:
            children = []
            for x in A:
                child = node + x
                level_nodes[m].append(child)
                runs.append(child)
                G.add_edge(node,child)
                children.append(child)

            for y in children:
                for z in children:
                    rules = get_graph_independent_matching_rules(y, z)
                    for rule in rules:
                        if isinstance(rule, NA):
                            if len(rules) != 1:
                                raise Exception("NA can not be associated with othe rules")

                            continue

                        verdict = rule.sign(y, z)
                        reversed_verdict = rule.sign(z, y)

                        if verdict:
                            G.add_edge(y,z)



    # eqaul_edges = []
    # edges = []
    #
    # # total =  len(runs) * (len(runs)-1 )
    # count = 0
    # for y in runs:
    #     for z in runs:
    #         # print ("{} of {} ".format(count,total))
    #         count+=1
    #         rules = get_graph_independent_matching_rules(y, z)
    #         for rule in rules:
    #                 if isinstance(rule, NA):
    #                     if len(rules) != 1:
    #                         raise Exception("NA can not be associated with othe rules")
    #
    #                     continue
    #
    #                 verdict = rule.sign(y, z)
    #                 reversed_verdict = rule.sign(z, y)
    #
    #                 if not xor(verdict, reversed_verdict):
    #                     if not (isinstance(rule, S_S) or isinstance(rule,P_NN) or isinstance(rule,RM_MR)):
    #                         raise Exception("A rule can not produce same ouput")
    #                     elif y!=z and not ((y,z) in eqaul_edges or (z,y) in eqaul_edges) :
    #                         eqaul_edges.append((y,z))
    #
    #                     continue
    #
    #                 if verdict:
    #                     G.add_edge(y, z)
    #                     if "-".join([y, z]) in edges or "-".join([z, y]) in edges:
    #                         print("Something went wrong, two edges are added")
    #                     else:
    #                         edges.append("-".join([y, z]))
    #
    #                     break
    #
    # E = G.copy()
    # for y, z in eqaul_edges:
    #     G.add_edge(y,z)
    #     E.add_edge(z,y)
    #
    #
    # acyclic_graph = nx.is_directed_acyclic_graph(G)
    # acyclic_graph_e_g = nx.is_directed_acyclic_graph(E)
    #
    # if not (acyclic_graph and acyclic_graph_e_g):
    #     # G.remove_edge(y,z)
    #     raise Exception("We can not have cycles")

    return G,G

def build_bi_graphs(runs):
    edges = []
    G = nx.DiGraph()

    eqaul_edges = []
    for y ,z in itertools.combinations(runs,2):
            rules = get_graph_independent_matching_rules(y, z)
            for rule in rules:
                if isinstance(rule, NA):
                    if len(rules) != 1:
                        raise Exception("NA can not be associated with othe rules")

                    continue

                verdict = rule.sign(y, z)
                reversed_verdict = rule.sign(z, y)

                if not xor(verdict, reversed_verdict):
                    if not (isinstance(rule, S_S) or isinstance(rule,P_NN) or isinstance(rule,RM_MR)):
                        raise Exception("A rule can not produce same ouput")
                    elif y!=z and not ((y,z) in eqaul_edges or (z,y) in eqaul_edges) :
                        eqaul_edges.append((y,z))

                    continue

                if verdict:
                    G.add_edge(y, z)
                    if "-".join([y, z]) in edges or "-".join([z, y]) in edges:
                        print("Something went wrong, two edges are added")
                    else:
                        edges.append("-".join([y, z]))

                    break

    E = G.copy()
    for y, z in eqaul_edges:
        G.add_edge(y,z)
        E.add_edge(z,y)


    acyclic_graph = nx.is_directed_acyclic_graph(G)
    acyclic_graph_e_g = nx.is_directed_acyclic_graph(E)

    if not (acyclic_graph and acyclic_graph_e_g):
        # G.remove_edge(y,z)
        raise Exception("We can not have cycles")

    return G,E


def get_graph_based_matching_rules(G,y,z):

    rules = []

    if Nx_Mx.match(y,z):
        rules.append(Nx_Mx)






def addSecondaryEges(G,na_edges):
    edges = []

    na_nodes = []
    for y,z in na_edges:
            rules = get_graph_based_matching_rules(G,y, z)

            for rule in rules:
                if isinstance(rule, NA):
                    if len(rules) != 1:
                        raise Exception("NA can not be associated with othe rules")

                    na_nodes.append((y, z))
                    continue

                verdict = rule.sign(y, z)
                reversed_verdict = rule.sign(z, y)

                if not xor(verdict, reversed_verdict):
                    if not isinstance(rule, S_S):
                        raise Exception("A rule can not produce same ouput")

                if verdict:
                    G.add_edge(y, z)

                    if "-".join([y, z]) in edges or "-".join([z, y]) in edges:
                        print("Something went wrong, two edges are added")
                    else:
                        edges.append("-".join([y, z]))

                    acyclic_graph = nx.is_directed_acyclic_graph(G)

                    if not (acyclic_graph):
                        # G.remove_edge(y,z)
                        raise Exception("We can not have cycles")

def validate_rules(a, M):


    G,E = build_bi_graphs(generate_runs(a,M))
    return G,E

def no_circles(G,nodes):
    for node in nodes:
        has_path = nx.has_path(G,node,node)

        if (has_path):
            print( "Circle exists between " + node)
            print(nx.dijkstra_path(G,node,node))


def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        if parent is not None:
            neighbors.remove(parent)
        for neighbor in neighbors:
            levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        if parent is not None:
            neighbors.remove(parent)
        for neighbor in neighbors:
            pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})
def find_equal_score(df, metrics=[],runs=[]):
    """
      Get all runs that have eqaul scores
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for y in runs:
            for z in runs:
                    yscore = runs_scores[y]
                    zscore = runs_scores[z]

                    equal = (yscore == zscore)

                    if equal:
                        print ("{},{},{},{},{}".format(metric,y,z,yscore,zscore))
    return

def find_violation_same_sequence(df, metrics=[], runs=[]):
    """
      Get all runs that have eqaul scores
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for y in runs:
            for z in runs:
                    yscore = runs_scores[y]
                    zscore = runs_scores[z]

                    equal = (y == z)

                    if equal:
                        yx = y + "x"
                        zx = z + "x"
                        yr = y + "a"
                        zr = z + "a"

                        if not ( runs_scores.has_key(yx) and runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores.has_key(zx)):
                            continue

                        if (yscore > runs_scores[zr]):
                            print ("S S'r: {},{},{},{},{}".format(metric, y, zr, runs_scores[y], runs_scores[zr]))

                        # if (yscore > runs_scores[zr]):
                        #     print ("S S'r: {},{},{},{},{}".format(metric,y,zr,runs_scores[y],runs_scores[zr]))
                        # if (yscore < runs_scores[zx]):
                        #     print ("S S'x: {},{},{},{},{}".format(metric, y, zx, runs_scores[y], runs_scores[zx]))
                        # if (runs_scores[yr] < runs_scores[zx]):
                        #     print ("Sr S'x: {},{},{},{},{}".format(metric, yr, zx, runs_scores[yr], runs_scores[zx]))
                        # if ((runs_scores[yx] < runs_scores[zx])):
                        #     print ("Sx S'x: {},{},{},{},{}".format(metric, yx, zx, runs_scores[yx], runs_scores[zx]))
    return

def find_violation_equal_score(df, metrics=[], runs=[]):
    """
      Get all runs that have eqaul scores
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for y in runs:
            for z in runs:
                    yscore = runs_scores[y]
                    zscore = runs_scores[z]

                    equal = (yscore == zscore)

                    if equal:
                        yx = y + "x"
                        zx = z + "x"
                        yr = y + "a"
                        zr = z + "a"

                        if not ( runs_scores.has_key(yx) and runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores.has_key(zx)):
                            continue


                        if (yscore > runs_scores[zr]):
                            print ("S S'r: {},{},{},{},{}".format(metric,y,zr,runs_scores[y],runs_scores[zr]))
                        if (yscore < runs_scores[zx]):
                            print ("S S'x: {},{},{},{},{}".format(metric, y, zx, runs_scores[y], runs_scores[zx]))
                        if (runs_scores[yr] < runs_scores[zx]):
                            print ("Sr S'x: {},{},{},{},{}".format(metric, yr, zx, runs_scores[yr], runs_scores[zx]))
                        # if ((runs_scores[yx] < runs_scores[zx])):
                        #     print ("Sx S'x: {},{},{},{},{}".format(metric, yx, zx, runs_scores[yx], runs_scores[zx]))
    return


def find_violation_less_score(df,metrics=[], runs=[]):
    """
      Get all runs that have eqaul scores
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for y in runs:
            for z in runs:
                    yscore = runs_scores[y]
                    zscore = runs_scores[z]

                    valid = yscore < zscore

                    if valid:
                        yx = y + "x"
                        zx = z + "x"
                        yr = y + "a"
                        zr = z + "a"

                        # if not ( runs_scores.has_key(yx) and runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores.has_key(zx)):
                        #     continue


                        if (runs_scores.has_key(zr) and yscore > runs_scores[zr]):
                            print ("S S'r: {},{},{},{},{}".format(metric,y,zr,runs_scores[y],runs_scores[zr]))

                        if (runs_scores.has_key(yx) and runs_scores[yx] > runs_scores[z]):
                                print ("Sx S': {},{},{},{},{}".format(metric, yx, z, runs_scores[yx], runs_scores[z]))

                        if (runs_scores.has_key(yx) and runs_scores.has_key(zr) and runs_scores[yx] > runs_scores[zr]):
                            print ("Sx S'r: {},{},{},{},{}".format(metric, yx, zr, runs_scores[yx], runs_scores[zr]))

                        # if (runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores[yr] > runs_scores[zr]):
                            #     print ("Sr S'r: {},{},{},{},{}".format(metric, yr, zr, runs_scores[yr], runs_scores[zr]))

                            # if (runs_scores.has_key(zx) and yscore > runs_scores[zx]):
                        #     print ("S S'x: {},{},{},{},{}".format(metric, y, zx, runs_scores[y], runs_scores[zx]))

                        # if (runs_scores.has_key(yr) and runs_scores.has_key(zx) and runs_scores[yr] > runs_scores[zx]):
                        #     print ("Sr S'x: {},{},{},{},{}".format(metric, yr, zx, runs_scores[yr], runs_scores[zx]))
                        # if (runs_scores.has_key(yx) and  runs_scores.has_key(zx) and runs_scores[yx]  > runs_scores[zx]):
                        #     print ("Sx S'x: {},{},{},{},{}".format(metric, yx, zx, runs_scores[yx], runs_scores[zx]))

    return

def find_violation_less_score(df,metrics=[], runs=[]):
    """
      Get all runs that have eqaul scores
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for y in runs:
            for z in runs:
                    yscore = runs_scores[y]
                    zscore = runs_scores[z]

                    valid = yscore < zscore

                    if valid:
                        yx = y + "x"
                        zx = z + "x"
                        yr = y + "a"
                        zr = z + "a"

                        # if not ( runs_scores.has_key(yx) and runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores.has_key(zx)):
                        #     continue


                        if (runs_scores.has_key(zr) and yscore > runs_scores[zr]):
                            print ("S S'r: {},{},{},{},{}".format(metric,y,zr,runs_scores[y],runs_scores[zr]))

                        if (runs_scores.has_key(yx) and runs_scores[yx] > runs_scores[z]):
                                print ("Sx S': {},{},{},{},{}".format(metric, yx, z, runs_scores[yx], runs_scores[z]))

                        # if (runs_scores.has_key(yx) and runs_scores.has_key(zr) and runs_scores[yx] > runs_scores[zr]):
                        #     print ("Sx S'r: {},{},{},{},{}".format(metric, yx, zr, runs_scores[yx], runs_scores[zr]))

                        if (runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores[yr] > runs_scores[zr]):
                                print ("Sr S'r: {},{},{},{},{}".format(metric, yr, zr, runs_scores[yr], runs_scores[zr]))

                        if (runs_scores.has_key(yr) and runs_scores.has_key(zr) and runs_scores[yr] <
                                    runs_scores[zr]):
                                    print (
                                    "Sr S'r: {},{},{},{},{}".format(metric, yr, zr, runs_scores[yr], runs_scores[zr]))

                            # if (runs_scores.has_key(zx) and yscore > runs_scores[zx]):
                        #     print ("S S'x: {},{},{},{},{}".format(metric, y, zx, runs_scores[y], runs_scores[zx]))

                        # if (runs_scores.has_key(yr) and runs_scores.has_key(zx) and runs_scores[yr] > runs_scores[zx]):
                        #     print ("Sr S'x: {},{},{},{},{}".format(metric, yr, zx, runs_scores[yr], runs_scores[zx]))
                        # if (runs_scores.has_key(yx) and  runs_scores.has_key(zx) and runs_scores[yx]  > runs_scores[zx]):
                        #     print ("Sx S'x: {},{},{},{},{}".format(metric, yx, zx, runs_scores[yx], runs_scores[zx]))

    return

def debug_rule(y,z):
    rules = get_graph_independent_matching_rules(y, z);

    for rule in rules:
        print ("{}({},{})=>{}".format(rule.name, y, z, rule.sign(y, z)))

def main(data_path,tasks,aspects,cutoff):

    os.path.split(data_path)
    A = aspects.split(",")
    adhoc_A = ["a"]
    for m in cutoff:
        for task in tasks:

            if task == "adhoc":
                print (" \n \n ============== {} ===========================".format(m))
                print ("  Ad hoc ")
                dd_result_file = os.path.join(data_path, "{}x-{}-adhoc.csv".format("".join(adhoc_A), m))
                df = pd.read_csv(os.path.expanduser(dd_result_file))
                metrics =   ["recip_rank", "P_5", "infAP", "ndcg", "ndcg_cut_5",
                                 "ndcg_cut_10", "map_cut_5", "success_1",
                                 "success_5"]
                validate_metrics(df, metrics=metrics, M=m, print_na=False, print_violation_only=False)

            if task == "dd":
                print (" \n \n ============== {} ===========================".format(m))


                for l in [10]:

                    dd_result_file = os.path.join(data_path,"{}x-{}-{}-dd.csv".format("".join(A),m,str(l)))
                    df = pd.read_csv(os.path.expanduser(dd_result_file))
                    metrics = ["ct", "act","nct","EU","nEU","sDCG","nsDCG"]

                    for ct in [1]:
                        print ("  DD Lenght : {} ==== Iteration: {} ".format(str(l),str(ct)))
                        ct_df = df[df["iteration"]==ct]
                        validate_metrics(df, metrics=metrics, M=m, print_na=False, print_violation_only=False)

            if task == "web":
                print (" \n \n Web Depth")

                metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
                    m).split(",")

                metrics_names = ""
                metrics = []
                for i in [5,10,20]:
                    metrics = metrics + "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(i).split(",")

                metrics = list(set(metrics))
                web_result_file = os.path.join(data_path,"{}x-{}-web.csv".format("".join(A),m))
                df = pd.read_csv(os.path.expanduser(web_result_file))
                validate_metrics(df, metrics=metrics, M=m, print_na=False, print_violation_only=False)


if __name__ == '__main__':
    a, M = ["a", "b"], 5

    data_path = "D:\\ameer\\txlabs\\clients\\rmit\\data\\results-csv"
    aspects = "a,b"
    cutoff = ["5"]

    # for m in [5]:
    #     # print_coverage_table(a, m, True)
    #     print (" \n \n ============== {} ===========================".format(m))
    #     print ("  DD ")
    #     dd_result_file = "ax-{}-dd.csv".format(m)
    #     df = pd.read_csv(os.path.expanduser(dd_result_file))
    #     metrics = ["avg_ct", "ct", "p"]
    #     #find_equal_score(df, metrics);
    #     # find_violation_equal_score(df, metrics)
    #     find_violation_less_score(df, metrics)
    #
    #     print (" \n \n Web Depth")
    #     metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
    #         m).split(",")
    #     web_result_file = "ax-{}-web.csv".format(m)
    #     t_web_result_file = "ax-{}-tweb.csv".format(m)
    #     df = pd.read_csv(os.path.expanduser(web_result_file))
    #     find_violation_equal_score(df, metrics)
    #     # find_violation_less_score(df,metrics)
    #
    #     print (" \n \n Truncated Depth")
    #     metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
    #         5).split(",")
    #     df = pd.read_csv(os.path.expanduser(t_web_result_file))
    #     # find_violation_equal_score(df, metrics)
    #     print ("Finish m " + (str(m)))
    # RM_MR.match("xa","ax")
    # RRN_RNR.match("aaab","aaba")
    # RRN_RNR.match("aaabb", "aabab")
    # RRN_RNR.match("aba", "baa")
    # debug_rule('bbba', 'abbb')
    # debug_rule('bbbaa', 'aabbb')
    # debug_rule("baaab", "bbaaa")
    # debug_rule("bbaaa", "baaab")
    # debug_rule('abaa','aaba')
    # debug_rule('aaba', 'abaa')
    # debug_rule('abbb', 'bbba')

    # debug_rule("aaab","aaba")
    # debug_rule("aaabb", "aabab")
    # debug_rule("aba", "baa")


    # RM_MR.match('abba','baab')
    # RRR_RNM.match("aaa", "abb")
    # M=3
    # validate_rules(a,M)
    # # M=4
    # # validate_rules(a,M)
    # # M=5
    # # validate_rules(a,M)
    #
    # G,E=buildGraphs(a,M)
    # if (nx.has_path(G,"abxx" , "bbba")):
    #     print(nx.shortest_path(G,"abxx" , "bbba"))
    # elif nx.has_path(E,"abxx" , "bbba"):
    #     print(nx.shortest_path(E,"abxx" , "bbba"))

    # debug_rule("bbbbx","")
    # debug_rule("","bbbbx")


    # debug_rule("aaab", "aaba")
    # debug_rule("aaba", "aaab")
    # debug_rule("aaab", "abaa")
    # debug_rule("aaba", "abaa")
    # debug_rule("aaba", "baaa")
    # debug_rule("", "a")
    # debug_rule("a", "")
    # debug_rule("","bbbba")
    # debug_rule("bbbba","")
    # debug_rule("aaxxa","axaax")
    # debug_rule("aaxxa", "xaaax")
    #
    # debug_rule("xaaax", "aaxxa")
    # debug_rule("aaa","abb")
    # debug_rule("abb", "aaa")
    # debug_rule("x","xx")



    # print_coverage_table(["a", "b"], 3)
    # for m in [3,5]:
    #     print_coverage_table(["a","b"],m)
    # # print (is_less("aaa","aa"))
    # # print (is_less("aa", "aaa"))
    # # print (is_less("aab", "aaba"))
    # # print (is_less("aaba", "aab"))
    # # print (is_less("ab", "abx"))
    # # print (is_less("abx", "abxx"))
    # # print (is_less("abxx", "abx"))
    # # print (is_less("aaab", "aaba"))
    # # print (is_less("aaba","aaab"))
    # # print (is_less("aax", "aa"))
    # # print (is_less("aa", "aax"))
    #
    # # print (is_less("ax", "aax"))
    # # print (is_less("aa", "aax"))
    # # print (is_less("aax", "aa"))
    # print (get_matching_rule("", "a").name)
    # print (get_matching_rule("a", "").name)
    # print (get_matching_rule("a", "aa").name)
    # print (get_matching_rule("aa", "a").name)
    # print (get_matching_rule("aa", "aaa").name)
    # print (get_matching_rule("aaa", "aa").name)
    #
    # print ("\n")
    # print (get_matching_rule("aaaaaaax", "aa").name)
    # print ("\n")
    #
    # print (get_matching_rule("aax", "aa").name)
    # print (get_matching_rule("aax", "aaa").name)
    # print (get_matching_rule("aaa", "aax").name)
    # print (get_matching_rule("a", "x").name)
    # print (get_matching_rule("x", "a").name)
    # print (get_matching_rule("ax", "ab").name)
    # print (get_matching_rule("ab", "ax").name)
    # print ("\n")
    #
    # print (get_matching_rule("ab", "aa").name)
    # print (get_matching_rule("aa", "ab").name)
    # print ("\n")
    #
    # print (get_matching_rule("xa", "xb").name)
    # print (get_matching_rule("xb", "xa").name)
    # print (get_matching_rule("xc", "xa").name)
    # print (get_matching_rule("xa", "xc").name)
    # print("\n")
    #
    # print (get_matching_rule("xa", "ax").name)
    # print (get_matching_rule("xxa", "aax").name)
    # print (get_matching_rule("ab", "abx").name)

    # print (get_matching_rule("xa", "ax").name)
    # print (get_matching_rule("x", "").name)
    # print (get_matching_rule("", "x").name)
    # print (get_matching_rule("", "xa").name)
    # print (get_matching_rule("", "aa").name)
    # print (get_matching_rule("", "aax").name)
    # print (get_matching_rule("x", "aax").name)
    # print (get_matching_rule("xaa", "aax").name)
    # print (get_matching_rule("xaa", "aax").name)
    # print (get_matching_rule("xa", "ax").name)
    # print (get_matching_rule("xb", "ax").name)

    # debug_rule("xb", "ax")
    # debug_rule("","")
    # debug_rule("x","x")
    # debug_rule("x", "x")
    #
    # debug_rule("xbb", "bxb")
    # debug_rule("xxb", "xbx")
    # debug_rule("xxx", "xx")
    # debug_rule("xx", "xxx")
    #
    # debug_rule("aaa", "abb")


    # G = buildGraph(a, M)
    # print(nx.has_path(G,"aa","aaa"))
    # print(nx.has_path(G, "aaa", "aa"))
    # print(nx.has_path(G, "a", "aaa"))
    # print(nx.has_path(G, "aaa", "a"))
    # print(nx.has_path(G, "xxx", "aaa"))
    # print(nx.has_path(G, "xxa", "axx"))
    # print(nx.has_path(G, "aaa", "aab"))
    # print(nx.has_path(G, "aab", "aaa"))
    # print(nx.has_path(G, "aa", "xxx"))
    # print(nx.has_path(G, "xxx", "aa"))
    # print(nx.has_path(G, "xa", "aa"))
    # print(nx.has_path(G, "aa", "xb"))
    # print(nx.has_path(G, "xb", "ax"))
    # print(nx.has_path(G, "xb", "aa"))
    # print(nx.has_path(G, "xbx", "xxb"))
    # print(nx.has_path(G, "xa", "aa"))
    # print(nx.has_path(G, "xaa", "aax"))
    # print(nx.has_path(G, "axa", "bxb"))
    # print(nx.has_path(G, "xx", "xx"))




    # print_coverage_table(a, 5, True)




    # # pd.read_csv
    # df = pd.read_csv(os.path.expanduser("abx-3-dd.csv"))
    # # metrics= ["avg_ct", "ct", "alpha-nDCG", "nERR-IA", "AVG_anDCG", "AVG_nERRIA", "nsdcg", "p"]
    # metrics = ["avg_ct", "ct"  ,"p"]
    # validate_metrics(df, metrics, print_na=False)
    #
    # print (" \n \nTruncated  @3 Depth")
    # metrics = ["alpha-DCG@3","alpha-nDCG@3","ERR-IA@3","nERR-IA@3","NRBP","nNRBP","P-IA@3","strec@3","MAP-IA"]
    # df = pd.read_csv(os.path.expanduser("abx-3-tweb.csv"))
    # validate_metrics(df, metrics=metrics, print_na=False)
    #
    # print (" \n \n Web @3 Depth")
    # df = pd.read_csv(os.path.expanduser("abx-3-web.csv"))
    # validate_metrics(df, metrics=metrics, print_na=False)

    for m in [5]:

        print_coverage_table(a, m, False)
        print (" \n \n ============== {} ===========================".format(m))
        print ("  DD ")
        dd_result_file = "abx-{}-5-dd.csv".format(m)
        # dd_result_file = "abx-{}-dd.csv".format(m)
        # metrics = ["ct","act"]
        df = pd.read_csv(os.path.join(data_path,dd_result_file))
        metrics = ["ct", "nct","act","EU","nEU","sDCG","nsDCG"]
        # it_df = df
        it_df = df[df["iteration"] == 1]
        validate_metrics(it_df, metrics,M=m, print_na=False, print_violation_only=False,one_branch_only=False)

        print (" \n \n Web Depth")
        metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
            m).split(",")
        web_result_file = os.path.join(data_path,"abx-{}-web.csv".format(m))
        t_web_result_file = "abx-{}-tweb.csv".format(m)
        df = pd.read_csv(os.path.expanduser(web_result_file))
        validate_metrics(df, metrics=metrics, M=m,print_na=False, print_violation_only=False,one_branch_only=True)

        # print(" \n \n Ad hoc Depth")
        #
        # print(" \n \n ============== {} ===========================".format(m))
        # print("  Ad hoc ")
        # adhoc_A = ["a"]
        # dd_result_file = os.path.join(data_path, "{}x-{}-adhoc.csv".format("".join(adhoc_A), m))
        # df = pd.read_csv(os.path.expanduser(dd_result_file))
        # metrics = ["recip_rank", "infAP", "ndcg",
        #            "ndcg_cut_10", "map_cut_5", "map_cut_10"]
        # adhoc_A = ["a"]
        # validate_metrics(df, metrics=metrics, M=m, print_na=False, print_violation_only=False, one_branch_only=False)



