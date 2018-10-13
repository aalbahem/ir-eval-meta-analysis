import os
import re

import itertools

import click
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def xor(a, b):
    return (a and not b) or (not a and b)

def eval(s,runs_scores):
    return runs_scores[s]


def violation_summary(metrics,metrics_matches,metrics_violations):
    print ("{},{},{}".format("metric", "matches", "violation"))
    for m in metrics:
        line = "{},{},{}".format(m, metrics_matches[m], metrics_violations[m])
        print (line)

#
# def validate_axiom_1(df, A=["a","b"],size=5,metrics=[],runs=[]):
#     """
#       Check metrics against axiom 1
#       m(Sx) <= m(S) <= m(Sr) <=m(Sd)
#     """
#
#     avg_df = df[df["topic"] == "all"]
#     runs_scores = {}
#     metrics_violations = {}
#     metrics_matches = {}
#
#     if len(metrics) == 0:
#         values = df.columns.values
#
#         for v in values:
#             metrics.append(v)
#
#         metrics.remove("topic")
#         metrics.remove("iteration")
#         metrics.remove("run")
#
#
#     for metric in metrics:
#         metrics_violations[metric]=0
#         metrics_matches[metric] = 0
#
#     for metric in metrics:
#
#         for name, group in avg_df.groupby("run"):
#             runs_scores[name] = group.iloc[0][metric]
#
#         if len(runs) == 0:
#             runs = runs_scores.keys()
#
#         for S in runs:
#             if len(S)==size:
#                 continue
#
#             metrics_matches[metric] +=1
#             Sx = S+"x"
#             Sr = S + "a"
#             Sd = None
#
#             for a in A:
#                 if S.find(a) >-1:
#                   Sr = S+a
#                 else:
#                   Sd = S+a
#             violaltion = False
#             if not(eval(Sx,runs_scores) <= eval(S,runs_scores)):
#                 violaltion = True
#
#                 print ("{}: Sx <= S, Sx={},S={},m(Sx)={},m(S)={}".format(metric,Sx,S,eval(Sx,runs_scores),eval(S,runs_scores)))
#             if not (eval(S, runs_scores) <= eval(Sr, runs_scores)):
#                 violaltion = True
#                 print ( "{}: S <= Sr, S={},Sr={},m(S)={},m(Sr)={}".format(metric, S, Sr, eval(S, runs_scores), eval(Sr, runs_scores)))
#             if Sd != None and not (eval(Sr, runs_scores) <= eval(Sd, runs_scores)):
#                 violaltion = True
#                 print ("{}: Sr <= Sd, Sr={},Sd={},m(Sr)={},m(Sd)={}".format(metric, Sr, Sd, eval(Sr, runs_scores), eval(Sd, runs_scores)))
#
#             if violaltion:
#                 metrics_violations[metric] += 1
#
#     violation_summary(metrics,metrics_matches,metrics_violations)
#
#     return (metrics_matches,metrics_violations)


def validate_s_sx(df, A=["a","b"],size=5,metrics=[],runs=[],print_cases=False):
    """
      Check metrics against axiom 1
      m(Sx) <= m(S) <= m(Sr) <=m(Sd)
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}
    metrics_violations = {}
    metrics_matches = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")


    for metric in metrics:
        metrics_violations[metric]=0
        metrics_matches[metric] = 0

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for S in runs:
            if len(S)==size:
                continue

            metrics_matches[metric] +=1
            Sx = S+"x"
            Sr = S + "a"
            Sd = None

            for a in A:
                if S.find(a) >-1:
                  Sr = S+a
                else:
                  Sd = S+a
            violaltion = False
            if not(eval(Sx,runs_scores) <= eval(S,runs_scores)):
                violaltion = True

                if print_cases:
                    print ("{}: Sx <= S, Sx={},S={},m(Sx)={},m(S)={}".format(metric,Sx,S,eval(Sx,runs_scores),eval(S,runs_scores)))
            if violaltion:
                metrics_violations[metric] += 1

    # violation_summary(metrics,metrics_matches,metrics_violations)

    return (metrics_matches,metrics_violations)

def validate_s_sr(df, A=["a","b"],size=5,metrics=[],runs=[],print_cases=False):
    """
      Check metrics against axiom 1
      m(Sx) <= m(S) <= m(Sr) <=m(Sd)
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}
    metrics_violations = {}
    metrics_matches = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")


    for metric in metrics:
        metrics_violations[metric]=0
        metrics_matches[metric] = 0

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for S in runs:
            if len(S)==size:
                continue

            metrics_matches[metric] +=1
            Sx = S+"x"
            Sr = S + "a"
            Sd = None

            for a in A:
                if S.find(a) >-1:
                  Sr = S+a
                else:
                  Sd = S+a
            violaltion = False

            if not (eval(S, runs_scores) <= eval(Sr, runs_scores)):
                violaltion = True

                if print_cases:
                    print ( "{}: S <= Sr, S={},Sr={},m(S)={},m(Sr)={}".format(metric, S, Sr, eval(S, runs_scores), eval(Sr, runs_scores)))

            if violaltion:
                metrics_violations[metric] += 1

    # violation_summary(metrics,metrics_matches,metrics_violations)

    return (metrics_matches,metrics_violations)


def validate_sr_sn(df, A=["a","b"],size=5,metrics=[],runs=[],print_cases=False):
    """
      Check metrics against axiom 1
      m(Sx) <= m(S) <= m(Sr) <=m(Sd)
    """

    avg_df = df[df["topic"] == "all"]
    runs_scores = {}
    metrics_violations = {}
    metrics_matches = {}

    if len(metrics) == 0:
        values = df.columns.values

        for v in values:
            metrics.append(v)

        metrics.remove("topic")
        metrics.remove("iteration")
        metrics.remove("run")


    for metric in metrics:
        metrics_violations[metric]=0
        metrics_matches[metric] = 0

    for metric in metrics:

        for name, group in avg_df.groupby("run"):
            runs_scores[name] = group.iloc[0][metric]

        if len(runs) == 0:
            runs = runs_scores.keys()

        for S in runs:
            if len(S)==size:
                continue

            metrics_matches[metric] +=1
            Sx = S+"x"
            Sr = S + "a"
            Sd = None

            for a in A:
                if S.find(a) >-1:
                  Sr = S+a
                else:
                  Sd = S+a
            violaltion = False

            if Sd != None and not (eval(Sr, runs_scores) <= eval(Sd, runs_scores)):
                violaltion = True
                if print_cases:
                    print ("{}: Sr <= Sd, S={},Sr={},Sd={},m(S)={},m(Sr)={},m(Sd)={}".format(metric, S,Sr, Sd,eval(S, runs_scores), eval(Sr, runs_scores), eval(Sd, runs_scores)))

            if violaltion:
                metrics_violations[metric] += 1

    # violation_summary(metrics,metrics_matches,metrics_violations)

    return (metrics_matches,metrics_violations)

#
# def validate_axiom_2(df, A=["a","b"],size=5,metrics=[],runs=[]):
#     """
#       Check metrics against axiom 1
#       m(Sx) <= m(S) <= m(Sr) <=m(Sd)
#     """
#
#     avg_df = df[df["topic"] == "all"]
#     runs_scores = {}
#     metrics_violations = {}
#     metrics_matches = {}
#
#     if len(metrics) == 0:
#         values = df.columns.values
#
#         for v in values:
#             metrics.append(v)
#
#         metrics.remove("topic")
#         metrics.remove("iteration")
#         metrics.remove("run")
#
#
#     for metric in metrics:
#         metrics_violations[metric]=0
#         metrics_matches[metric]=0
#
#     for metric in metrics:
#
#         for name, group in avg_df.groupby("run"):
#             runs_scores[name] = group.iloc[0][metric]
#
#         if len(runs) == 0:
#             runs = runs_scores.keys()
#
#         for S in runs:
#             for S_ in runs:
#                 if len(S)==size:
#                     continue
#
#                 if eval(S,runs_scores) !=eval(S_,runs_scores):
#                     continue
#                 metrics_matches[metric] +=1
#                 Sx = S+"x"
#                 Sr = S + "a"
#                 Sd = None
#
#                 for a in A:
#                     if S.find(a) >-1:
#                       Sr = S+a
#                     else:
#                       Sd = S+a
#
#                 violation = False
#                 if not(eval(Sx,runs_scores) <= eval(S_,runs_scores)):
#                     violation = True
#                     print ("{}: m(S)=m(S') and Sx <= S', S'={},S={},Sx={},m(S')={},m(S)={},m(Sx)".format(metric,
#                                                                                                     S_,S,Sx,
#                                                                                                     eval(S_,runs_scores),
#                                                                                                     eval(S,runs_scores),
#                                                                                                     eval(S,runs_scores)))
#                 if not (eval(S_, runs_scores) <= eval(Sr, runs_scores)):
#                     violation = True
#
#                     print ( "{}: m(S)=m(S') and S' <= Sr, S'={},S={},Sr={},m(S')={},m(Sr)={}".format(metric, S_,S, Sr, eval(S_, runs_scores), eval(Sr, runs_scores)))
#                 if Sd != None and not (eval(S_, runs_scores) <= eval(Sd, runs_scores)):
#                     violation = True
#                     # print ("{}: m(S)=m(S') and S' <= Sd, S'={},S={},Sd={},m(Sr)={},m(Sd)={}".format(metric, Sr, Sd, eval(Sr, runs_scores), eval(Sd, runs_scores)))
#
#                 if violation:
#                     metrics_violations[metric] += 1
#
#
#
#     violation_summary(metrics,metrics_matches,metrics_violations)
#
#     return (metrics_matches,metrics_violations)
#
# def validate_axiom_3(df, A=["a","b"],size=5,metrics=[],runs=[]):
#     """
#       Check metrics against axiom 1
#       m(Sx) <= m(S) <= m(Sr) <=m(Sd)
#     """
#
#     avg_df = df[df["topic"] == "all"]
#     runs_scores = {}
#     metrics_violations = {}
#
#     if len(metrics) == 0:
#         values = df.columns.values
#
#         for v in values:
#             metrics.append(v)
#
#         metrics.remove("topic")
#         metrics.remove("iteration")
#         metrics.remove("run")
#
#     metrics_matches = {}
#     for metric in metrics:
#         metrics_violations[metric]=0
#         metrics_matches[metric] = 0
#
#     for metric in metrics:
#
#         for name, group in avg_df.groupby("run"):
#             runs_scores[name] = group.iloc[0][metric]
#
#         if len(runs) == 0:
#             runs = runs_scores.keys()
#
#         for S in runs:
#             for S_ in runs:
#                 if len(S)==size:
#                     continue
#
#                 if eval(S_,runs_scores) < eval(S,runs_scores):
#                     metrics_matches[metric] +=1
#
#                     Sr = S + "a"
#                     S_x = S_ + "x"
#                     Sd = None
#
#                     for a in A:
#                         if S.find(a) >-1:
#                           Sr = S+a
#                         else:
#                           Sd = S+a
#
#                     violation = False
#                     if runs_scores.has_key(S_x) and not(eval(S_x,runs_scores) < eval(S,runs_scores)):
#                         violation = True
#                         print ("{}: m(S') < m(S), S'x < S: S={},S'={},S'x={},m(S)={},m(S')={},m(S'x)={}".format(metric,
#                                                                                                                 S,S_,S_x,
#                                                                                                                 eval(S,runs_scores),
#                                                                                                                 eval(S_,runs_scores),
#                                                                                                                 eval(S_x,runs_scores)))
#                     if Sd !=None and runs_scores.has_key(S_x) and not (eval(S_x, runs_scores) < eval(Sd, runs_scores)):
#                         violation = True
#                         print ("{}: m(S') < m(S), S'x < Sd: S={},S'={},S'x={},Sd={},m(S)={},m(S')={},m(S'x)={},m(Sd)={}".format(metric,
#                                                                                                                 S, S_,S_x,Sd,
#                                                                                                                 eval(S,runs_scores),
#                                                                                                                 eval(S_,runs_scores),
#                                                                                                                 eval(S_x,runs_scores),
#                                                                                                                 eval(Sd,runs_scores)))
#
#                     if not (eval(S_, runs_scores) < eval(Sr, runs_scores)):
#                         violation = True
#                         metrics_violations[metric] += 1
#                         print ( "{}: m(S') < m(S), S' < Sr, S'={},S={},Sr={},m(S')={},m(S)={},m(Sr)={}".format(metric,
#                                                                                                                S_,S, Sr,
#                                                                                                                eval(S_, runs_scores),
#                                                                                                                eval(S, runs_scores),
#                                                                                                                eval(Sr, runs_scores)))
#
#                     if Sd !=None and not (eval(S_, runs_scores) < eval(Sd, runs_scores)):
#                         violation = True
#                         print ("{}: m(S') < m(S), S' < Sd, S'={},S={},Sd={},m(S')={},m(S)={},m(Sd)={}".format(metric, S_, S, Sd,
#                                                                                                               eval(S_, runs_scores),
#                                                                                                               eval(S, runs_scores),
#                                                                                                               eval(Sd, runs_scores)))
#                     if violation:
#                         metrics_violations[metric] += 1
#
#
#     violation_summary(metrics,metrics_matches,metrics_violations)
#
#     return (metrics_matches,metrics_violations)

def validate_axioms(df, A=["a","b"],size=5,metrics=[],runs=[]):
    axioms = {}
    # axioms["Axiom 1"] = validate_axiom_1(df,A,size,metrics,runs)
    # axioms["Axiom 2"] = validate_axiom_2(df, A, size, metrics, runs)
    # axioms["Axiom 3"] = validate_axiom_3(df, A, size, metrics, runs)
    # keys = ["Axiom 1","Axiom 2","Axiom 3"]

    axioms["S leq S_r"] = validate_s_sr(df,A,size,metrics,runs)
    axioms["S_x leq S"] = validate_s_sx(df, A, size, metrics, runs)
    axioms["S_r leq S_n"] = validate_sr_sn(df, A, size, metrics, runs)
    keys = ["S leq S_r","S_x leq S","S_r leq S_n"]

    print("Metric,"+",".join(keys))
    for metric in metrics:
        cells = [metric]
        for axiom in keys:
            matches,violations = axioms[axiom][0],axioms[axiom][1]
            cells.append ("{}({})".format(matches[metric],violations[metric]))

        print (",".join(cells))




# @click.command()
# @click.option('-data_path', default="H:\\projects\\trec-dd\\metric-evalutions\\results-csv")
# @click.option('-aspects', default="a", help="comman separated list of aspects. use a to evalute a single or adhoc mode")
# @click.option('-cutoff', default="5",multiple=True)


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
                validate_axioms(df, A=adhoc_A, size=int(m), metrics=metrics, runs=[])

            if task == "dd":
                print (" \n \n ============== {} ===========================".format(m))


                for l in [5]:

                    dd_result_file = os.path.join(data_path,"{}x-{}-{}-dd.csv".format("".join(A),m,str(l)))
                    df = pd.read_csv(os.path.expanduser(dd_result_file))
                    metrics = ["ct", "act","nct","EU","nEU","sDCG","nsDCG"]

                    for ct in [1]:
                        print ("  DD Lenght : {} ==== Iteration: {} ".format(str(l),str(ct)))
                        ct_df = df[df["iteration"]==ct]
                        validate_axioms(ct_df, A=A, size=int(m), metrics=metrics, runs=[])

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
                validate_axioms(df, A=A, size=int(m), metrics=metrics, runs=[])

if __name__ == '__main__':
    # data_path="H:\\projects\\trec-dd\\metric-evalutions\\results-csv"
    data_path = "D:\\ameer\\txlabs\\clients\\rmit\\data\\results-csv"
    aspects = "a,b"
    cutoff = ["5"]

    # tasks = ["dd","adhoc","web"]
    tasks = ["dd"]
    main(data_path,tasks,aspects,cutoff)

    # for m in [5]:
    #     # print_coverage_table(a, m, True)
    #     print (" \n \n ============== {} ===========================".format(m))
    #     print ("  DD ")
    #     dd_result_file = "abx-{}-dd.csv".format(m)
    #     df = pd.read_csv(os.path.expanduser(dd_result_file))
    #     metrics = ["avg_ct", "ct", "p"]
    #     validate_axioms(df, A=a, size=M, metrics=metrics, runs=[])

        # print (" \n \n Web Depth")
        # metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
        #     m).split(",")
        # web_result_file = "abx-{}-web.csv".format(m)
        # df = pd.read_csv(os.path.expanduser(web_result_file))
        # validate_axioms(df, A=a, size=M, metrics=metrics, runs=[])

        # print (" \n \n Truncated Depth")
        # t_web_result_file = "abx-{}-tweb.csv".format(m)
        # metrics = "alpha-DCG@{0},alpha-nDCG@{0},ERR-IA@{0},nERR-IA@{0},NRBP,nNRBP,P-IA@{0},strec@{0},MAP-IA".format(
        #     5).split(",")
        # df = pd.read_csv(os.path.expanduser(t_web_result_file))
        # validate_axioms(df, A=a, size=M, metrics=metrics, runs=[])
        # print ("Finish m " + (str(m)))


