import os
import re

import itertools

import pandas as pd
import networkx as nx

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

    """
    Represent a relation that is not model by the current rules.
    """

    def sign(self, y, z):
        return "-"


class Path(Rule):
    """
    A induction rule that uses a graph to check violations of metrics
    """


    name = "Induction"


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

        return is_match and verdict




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



class RuleValidator(object):


    def __init__(self,rules):
        self.rules=rules
        self.graph = None

    def get_matching_rules(self, y, z):
        """ Get which rules that match the two string"""

        matched_rules = []
        for rule in self.rules:
            if rule.match(y, z):
                matched_rules.append(rule)

        return matched_rules

    def validate_metrics(self,df, A=[],metrics=[], runs=[], m=3,print_violation=False):
        """
          Evaluate the metrics using the Rule validations
        """

        avg_df = df[df["topic"] == "all"]

        metrics_rules_violation = {}

        path_rule = Path(self.graph)

        if len(metrics) == 0:
            values = df.columns.values

            for v in values:
                metrics.append(v)

            metrics.remove("topic")
            metrics.remove("iteration")
            metrics.remove("run")

        metrics_rules_viloations = {}
        metrics_rules_matches = {}

        rules_names = [r.name for r in self.rules]+[Path.name,NA.name]
        for metric in metrics:
            metrics_rules_viloations[metric] = {}
            metrics_rules_matches[metric] = {}
            for r in rules_names:
                metrics_rules_viloations[metric][r] = 0
                metrics_rules_matches[metric][r] = 0

        runs_metrics_scores = {}


        records = pd.DataFrame.to_dict(avg_df, orient='records')

        for r in records:
            name = r["run"]
            runs_metrics_scores[name] = r


        if len(runs) == 0:
                runs =self.generate_runs(A,m)


        self.check_count_violation(metrics, metrics_rules_matches, metrics_rules_viloations, path_rule,runs_metrics_scores,print_violation)

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

        metrics_pretty= {
            "recip_rank":"\metric{RR}",
            "P_10":"\metricat{P}{10}",
            "infAP":"\metric{infAP}",
            "ndcg_cut_10":"\\ndcgat{10}",
            "ndcg": "\\ndcg",
            "map_cut_10":"\metric{AP}",
            "success_10":"\metricat{Success}{10}",
            "ct":"\metric{CT}",
            "nct":"\metric{nCT}",
            "act":"\metric{ACT}",
            "EU":"\metric{EU}",
            "nEU":"\metric{nEU}",
            "sDCG":"\metric{sDCG}",
            "nsDCG":"\metric{nsDCG}",
             "P-IA@10":"\metricat{P-IA}{10}",
             "alpha-DCG@10":"\metricat{$\\alpha$\metric{-DCG}}{10}",
            "nERR-IA@10":"\metricat{nERR-IA}{10}",
            "strec@10":"\metricat{STREC}{10}",
            "NRBP":"\\nrbp",
            "ERR-IA@10":"\metricat{ERR-IA}{10}",
            "alpha-nDCG@10":"\metricat{$\\alpha$\metric{-NDCG}}{10}",
            "nNRBP":"\metric{nNRBP}",
            "MAP-IA":"\metric{AP-IA}",
            "rbu":"\metricat{RBU}{10}",
            "rbu-0.990-0.000":"\metricat{RBU_{p=0.990,e=0}}{10}",
            "rbu-0.990-0.001": "\metricat{RBU_{p=0.990,e=0.001}}{10}",
        }


        print (" & ".join(columns))
        for m in metrics:
            violation_total = 0
            rules_v = [metrics_pretty[m]]
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

    def check_count_violation(self, metrics, metrics_rules_matches, metrics_rules_viloations, path_rule,runs_metrics_scores,print_violation=False):
        def reachable(start):
            # the set of reachable nodes
            reachable = set()

            # recursive function to add all reachable nodes to `reachable`
            def finder(node):
                reachable.add(node)
                for me, other in self.graph.out_edges(node):
                    finder(other)

            # add everything we can reach from here
            finder(start)
            return reachable

        count = 0
        for r in self.graph.nodes_iter():
            count += 1
            reachables = reachable(r)

            y = r
            for z in reachables:
                if (z == '' or y == ''):
                    continue
                rules = self.get_matching_rules(y, z)

                # for the graph, we new there is path, check if metric detect them right
                for metric in metrics:
                    metrics_rules_matches[metric][path_rule.name] += 1
                    expect = True
                    yscore = runs_metrics_scores[y][metric]
                    zscore = runs_metrics_scores[z][metric]
                    actual = yscore <= zscore

                    not_equal = (yscore != zscore)

                    if xor(expect, actual) and not_equal:
                        metrics_rules_viloations[metric][path_rule.name] += 1

                for rule in rules:
                    if (isinstance(rule, NA)):
                        continue

                    for metric in metrics:

                        metrics_rules_matches[metric][rule.name] += 1
                        metrics_rules_matches[metric][path_rule.name] += 1
                        expect = rule.sign(y, z)
                        yscore = runs_metrics_scores[y][metric]
                        zscore = runs_metrics_scores[z][metric]
                        actual = yscore <= zscore

                        not_equal = (yscore != zscore)

                        if xor(expect, actual) and not_equal:
                            metrics_rules_viloations[metric][rule.name] += 1
                            if print_violation:
                                print(
                                "{}, {}({} , {}) = {} => ({}'s score: {:.4f}, {}'s score: {:.4f}, expected:{}, actual:{})".format(
                                    metric, rule.name, y, z, xor(expect, actual), y, yscore, z, zscore, expect, actual))

    def generate_runs(self,a,m):
        A = a + ["x"]

        runs = [""]  # empty set

        level_nodes = []
        level_nodes.append([""])  # the first node is an empty node.
        for m in range(1, m + 1):
            level_nodes.append([])
            for node in level_nodes[m - 1]:
                for x in A:
                    child = node + x
                    level_nodes[m].append(child)
                    runs.append(child)
        return runs


    def build_graph(self, A, m):
        """

        Builds a relation graph using rules for all truncated and diversfied raunkings of maximums size of M
        and query aspect A

        for instance if we want to build graph of 2-document ranking for two aspects, then :
        m = 2 and A = {"a","b"}

        :param A:
               a list of alphavet the represnet the diversfication aspects such a, b, c
        :param m:
               the size of the a ranking
        :return:
           return nx.graph object
        """

        G = nx.DiGraph()

        A_X = A + ["x"]

        runs = [""]  # empty set

        level_nodes = []
        level_nodes.append([""])  # the first node is an empty node.
        for m in range(1, m + 1):
            level_nodes.append([])
            for node in level_nodes[m - 1]:
                children = []
                for x in A_X:
                    child = node + x
                    level_nodes[m].append(child)

                    runs.append(child)

                    if (x == "x"):
                      G.add_edge(child,node)
                    else:
                        G.add_edge(node,child)
                    children.append(child)

                for y in children:
                    for z in children:
                            rule = P_RN()
                            verdict = rule.sign(y, z)
                            reversed_verdict = rule.sign(z, y)

                            if verdict:
                                G.add_edge(y,z)

        return G

def adcs2018_experiment(data_path):
        M = [10]
        A= ["a","b"]
        rules = [SX_S(), S_SR(), P_RN()]
        validator = RuleValidator(rules=rules)
        dd_metrics = ["ct","nct","act","EU","nEU","sDCG","nsDCG"]

        adhoc_metrics = ["recip_rank", "P_10", "infAP", "ndcg", "ndcg_cut_10", "map_cut_10", "success_10"]

        for m in M:
            print (" \n \n Building tree relationship with ".format(m) + " A: " + ",".join(A))
            validator.graph =   validator.build_graph(A=A, m=m)

            print ("  Dynamic Domain Metrics")
            dd_result_file = "abx-{}-dd.csv".format(m)
            dd_df = pd.read_csv(os.path.join(data_path, dd_result_file))
            validator.validate_metrics(dd_df, A=A, metrics=dd_metrics, m=m)

            print (" \n \n Web Depth")

            web_d_metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
                m).split(",")
            web_result_file = os.path.join(data_path, "abx-{}-web.csv".format(m))
            wd_df = pd.read_csv(os.path.expanduser(web_result_file))

            validator.validate_metrics(wd_df, A=A, metrics=web_d_metrics, m=m)

            print(" \n \n Ad hoc Depth")

            print(" \n \n ============== {} ===========================".format(m))
            print("  Ad hoc ")
            adhoc_A = ["a"]
            dd_result_file = os.path.join(data_path, "{}x-{}-adhoc.csv".format("".join(adhoc_A), m))
            df = pd.read_csv(os.path.expanduser(dd_result_file))


            adhoc_A = ["a"]
            validator.graph = validator.build_graph(A=adhoc_A,m=m)
            validator.validate_metrics(df, A=adhoc_A,metrics=adhoc_metrics, m=m)


if __name__ == '__main__':


    adcs2018_experiment(data_path = "data")






