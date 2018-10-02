import numpy
import pandas as pd
import os
import csv

import itertools

import pandas
import scipy
import scipy.stats as stats
import sys

from mu import MU
metrics=[]
def to_dict(csvfile, by=["run","topic","iteration"]):
    with open(csvfile, 'rb') as csvfile:
        all_ = list(csv.DictReader(csvfile))

        d = {}

        for row in all_:
            run = row[by[0]]
            topic = row[by[1]]
            iteration = row[by[2]]

            if run not in d.keys():
                d[run] = {}

            if topic not in d[run].keys():
                d[run][topic]={}

            if iteration not in d[run][topic].keys():
                d[run][topic][iteration] = {}

            for m in row.keys() :
                if m not in by:
                    d[run][topic][iteration][m]=row[m]

        return d

def generate_mu_simple_metrics(eval_path, year, metrics, mu_analysis_file):
    I_S_T_M = to_dict(os.path.join(eval_path, "trec-all-{}-evals.csv".format(year)), ["iteration", "run", "topic"])
    csv_writer = csv.DictWriter(open(mu_analysis_file,mode="wb"),fieldnames=["iteration","dimensions","metric","mu","m_ij","set_ij","m_set_ij"])
    csv_writer.writeheader()



    assert(len(I_S_T_M.keys()) ==10)
    for it in I_S_T_M.keys():
        S_M = I_S_T_M[str(it)]
        R = S_M.keys()
        T = []
        for r in R:
            T.extend(S_M[r].keys())
        T = list(set(T))
        T.remove("all")

        M_GS = [
            ["p"],
            ["strec"],

            # ["rtime"],
            # ["ntime"],
            ["ttime"],

            ["strec", "p"],
            # ["strec", "rtime"],
            # ["strec", "ntime"],
            ["strec", "ttime"],
            # ["p", "rtime"],
            # ["p", "ntime"],
            # ["p", "ttime"],

            # ["strec", "p", "rtime"],
            # ["strec", "p", "ntime"],
            ["strec", "p", "ttime"],
        ]

        for MG in M_GS:
            for m in metrics:
                M = [m] + MG
                mu = MU(T, R, M, S_M)
                pmis = mu.mu()


                print ("{}\t{}\t{}\t{}".format(str(it), "+".join(MG), m, pmis[m]))
                pmis[m]["iteration"]=it
                pmis[m]["metric"] = m
                pmis[m]["dimensions"] = "+".join(MG)
                csv_writer.writerow(pmis[m])


def generate_mu_complex_metrics(eval_path, year, metrics, mu_analysis_file):
    I_S_T_M = to_dict(os.path.join(eval_path, "trec-all-{}-evals.csv".format(year)), ["iteration", "run", "topic"])



    with open(mu_analysis_file, mode="w") as mu_out:
        for it in I_S_T_M.keys():
            S_M = I_S_T_M[it]
            R = S_M.keys()
            T = []
            for r in R:
                T.extend(S_M[r].keys())
            T = list(set(T))
            T.remove("all")

            M = metrics
            mu = MU(T, R, M, S_M)
            pmis = mu.mu()

            for m in metrics:

                print ("{}\t{}\t{}\t{}".format(it, "complex-metrics", m, pmis[m]))
                mu_out.write("{},{},{},{}\n".format(it, "complex-metrics", m, pmis[m]))



if __name__=="__main__":
    years = ["2015","2016","2017"]
    simple_metrics = ["strec", "p", "ttime"]
    dd_metrics = ["act", "nct", "nEU", "nsDCG"]
    wd_metrics = "alpha-nDCG,nERR-IA".split(",")



    # Ps = [0.800, 0.900, 0.990]
    # Es = [0.001, .050, 0.1, 0.5]
    Ps = [0.990]
    # Es = [.050]
    Es = [.050]


    # rbus_metrics = ["rbu_0.800_0.001","rbu_0.800_0.050","rbu_0.800_0.1","rbu_0.800_0.5",
    #                 "rbu_0.900_0.001","rbu_0.900_0.050","rbu_0.900_0.1","rbu_0.900_0.5",
    #                 "rbu_0.990_0.001","rbu_0.990_0.050","rbu_0.990_0.1","rbu_0.990_0.5"]
    rbus_metrics = ["rbu_{0:.3f}_{1:.3f}".format(p,e) for p,e in itertools.product(Ps,Es)]
    complex_metrics = dd_metrics+wd_metrics + rbus_metrics
    metrics = complex_metrics

    all_metrics = dd_metrics+wd_metrics+simple_metrics

    metrics_pretty_header ={"act":"\\act","nct":"\\nct","nEU":"\\neu","nsDCG":"\\nsdcg","alpha-nDCG":"\\alphandcg","nERR-IA":"\\nerria"}
    tie_data_rank = "dense"
    for m in rbus_metrics:
        metrics_pretty_header[m] = "\\rbu".format(m.split("_")[1],m.split("_")[2])
    M_GS = [
        ["p"],
        ["strec"],

        # ["rtime"],
        # ["ntime"],
        ["ttime"],

        ["strec", "p"],
        # ["strec", "rtime"],
        # ["strec", "ntime"],
        # ["strec", "ttime"],
        # ["p", "rtime"],
        # ["p", "ntime"],
        # ["p", "ttime"],
        #
        # ["strec", "p", "rtime"],
        # ["strec", "p", "ntime"],
        ["strec", "p", "ttime"],
    ]

    M_GS_Labels = ["+".join(m_s) for m_s in M_GS]

    class Mu_TREC_Analysis:

        def get_year_iteration_mu(self, year, mu_csv_file, metrics_sets):
            metrics_set_scores = {}
            for metrics_set in metrics_sets:

                iteration_scores = {}
                mu_analysis_file = os.path.join("..","data", "evals", year, mu_csv_file.format(year))

                mu_map = to_dict(mu_analysis_file, by=["iteration", "dimensions", "metric"])

                for i in range(1,11):
                    iteration_scores[str(i)] = {}
                    iteration_mu = mu_map[str(i)]
                    mu_gold = metrics_set

                    gold_mu_scores = iteration_mu[mu_gold]

                    for m in complex_metrics:
                        iteration_scores[str(i)][m] = {"mu": float(gold_mu_scores[m]["mu"]),
                                                       "m_ij":float(gold_mu_scores[m]["m_ij"]),
                                                       "set_ij":float(gold_mu_scores[m]["set_ij"]),
                                                       "m_set_ij":float(gold_mu_scores[m]["m_set_ij"])
                                                       }


                metrics_set_scores[metrics_set] = iteration_scores

            return metrics_set_scores;

        def calculate_iteration_mu_cd_agreement(self,metric_set,iteration_mu,iteration_it):
            same_count = 0
            count = 0
            seen = []
            for m in complex_metrics:
                for n in complex_metrics:
                    if n == m:
                        continue

                    if (m, n) in seen or (n, m) in seen:
                        continue
                    else:
                        seen.append((m, n))

                    # for gold in ["p","strec","ttime","strec+p","strec+p+ttime"]:
                    for gold in [metric_set]:
                        mu_gold = "complex-metrics" if gold not in iteration_mu else gold

                        gold_mu_scores = iteration_mu[mu_gold]
                        gold_cd_scores = iteration_it[gold]

                        m_it_mu = gold_mu_scores[m]
                        n_it_mu = gold_mu_scores[n]
                        mu_diff = float(m_it_mu["mu"]) - float(n_it_mu["mu"])
                        cd_diff = float(gold_cd_scores[m + "-" + n]["dis_correct_lef"]) - float(
                            gold_cd_scores[m + "-" + n]["dis_correct_right"])

                        if (mu_diff * cd_diff > 0) or (mu_diff == 0 and cd_diff == 0):
                            same_count += 1

                        count += 1
            return ({"count":count,"agree_count":same_count,"agree_prob":float(same_count)/count})


        def calculate_pair_mu_cd_agreement(self,m,n,pair_mu,pair_it):
            same_count = 0
            count = 0
            for i in range(1,11):
                for gold in ["strec+p+ttime"]:
                    mu_gold = "complex-metrics" if gold not in pair_mu[m] else gold
                    m_mu_scores = pair_mu[m][mu_gold][str(i)]
                    n_mu_scores = pair_mu[n][mu_gold][str(i)]

                    m_it_scores = pair_it[gold][str(i)]
                    n_it_scores = pair_it[gold][str(i)]


                    mu_diff = float(m_mu_scores["mu"]) - float(n_mu_scores["mu"])
                    cd_diff = float(m_it_scores["dis_correct_lef"]) - float(n_it_scores["dis_correct_right"])

                    if (mu_diff * cd_diff > 0) or (mu_diff == 0 and cd_diff == 0):
                        same_count += 1

                    count += 1

            return ({"count":count,"agree_count":same_count,"agree_prob":float(same_count)/count})




        def generate_pair_mu_cd_agreement(self,metrics,year,mu_csv_file,mettrics_sets ):
            mu_analysis_file = os.path.join("data", "evals", year, mu_csv_file.format(year,mettrics_set))
            condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))

            mu_map = to_dict(mu_analysis_file, by=["metric","dimensions","iteration"])
            cd_map = to_dict(condorcment_file, by=["pair","gold","iteration"])

            print("\\midrule " + year + "\\\\\\midrule")
            pair_scores = {}
            seen = []
            for m in metrics:
                for n in metrics:
                    if (m==n):
                        continue

                    if any([(m,n) in seen,  (n,m) in seen]):
                        continue
                    else:
                        seen.append((m,n))

                    pair_mu = {m:mu_map[m],n:mu_map[n]}
                    pair_it = cd_map[m+"-"+n]
                    pair_scores[m+"-"+n]= self.calculate_pair_mu_cd_agreement(m=m,n=n,pair_mu=pair_mu,pair_it=pair_it);


            for pair,scores in pair_scores.iteritems():
                print("&".join([pair, "{:.4f}".format(scores["agree_prob"])]) + "\\\\")

        def get_iteration_mu_cd_agreements(self, year, mu_csv_file, metrics_sets):
            metrics_set_scores = {}
            for metrics_set in metrics_sets:
                iteration_scores = {}
                mu_analysis_file = os.path.join("data", "evals", year, mu_csv_file.format(year))
                condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))

                mu_map = to_dict(mu_analysis_file, by=["iteration", "dimensions", "metric"])
                cd_map = to_dict(condorcment_file, by=["iteration", "gold", "pair"])

                for i in range(1,11):
                    iteration_mu = mu_map[str(i)]
                    iteration_it = cd_map[str(i)]
                    iteration_scores[str(i)]= self.calculate_iteration_mu_cd_agreement(metrics_set,iteration_mu=iteration_mu,iteration_it=iteration_it);

                metrics_set_scores[metrics_set]=iteration_scores

            return metrics_set_scores;

        def get_iteration_cd_ranking(self, year, metrics_sets,iterations):
            metrics_set_rankings = {}

            condorcment_file = os.path.join("..","data", "evals", year, "metric-condorcent-{}.csv".format(year))
            cd_map = to_dict(condorcment_file, by=["iteration", "gold", "pair"])
            for metrics_set in metrics_sets:
                iterations_rankings = {}
                for i in iterations:
                    iteration_it = cd_map[str(i)]
                    itr_gold_it = iteration_it[metrics_set]

                    remainding = list(complex_metrics)
                    iteration_ranking = []
                    limit = len(complex_metrics)
                    r = 1
                    while r <=limit and len(remainding) >0:
                        best = remainding[0]
                        for m in remainding:
                            if m ==best:
                                continue

                            if ("-".join([m,best]) not in itr_gold_it.keys()):
                                continue
                            pair_cd = itr_gold_it["-".join([m,best])]
                            if float(pair_cd["dis_correct_left"])>float(pair_cd["dis_correct_right"]):
                                best = m
                            else:
                                best = best
                        remainding.remove(best)
                        if (len(iteration_ranking) ==0):
                            iteration_ranking.append( ( best, r ))
                            r+=1
                            continue



                        for prev in iteration_ranking:
                            if ("-".join([prev[0], best]) not in itr_gold_it.keys()):
                                continue
                            pair_cd = itr_gold_it["-".join([prev[0], best])]
                            if float(pair_cd["dis_correct_left"]) < float(pair_cd["dis_correct_right"]):
                                print("Can not happend ")

                        if  float(pair_cd["dis_correct_left"]) == float(pair_cd["dis_correct_right"]):
                            iteration_ranking.append((best,r-1))
                        else:
                            iteration_ranking.append((best, r))
                            r += 1


                    erro = False
                    for m,n in itertools.combinations(iteration_ranking,2):
                        if "-".join([m[0], n[0]]) not in itr_gold_it.keys():
                            continue

                        pair_cd = itr_gold_it["-".join([m[0], n[0]])]

                        if float(pair_cd["dis_correct_left"])< float(pair_cd["dis_correct_right"]) and m[1]< n[1]:
                            erro=True
                        if float(pair_cd["dis_correct_left"]) > float(pair_cd["dis_correct_right"]) and m[1] > n[1]:
                            erro=True
                        if float(pair_cd["dis_correct_left"]) == float(pair_cd["dis_correct_right"]) and m[1] != n[1]:
                            erro=True
                    if (erro):
                        print("Ranking is not right")

                    ranking = {}

                    for data in iteration_ranking:
                        ranking[data[0]]={"metric":data[0],"it_rank":data[1]}

                    iterations_rankings[str(i)] = ranking

                metrics_set_rankings[metrics_set]=iterations_rankings

            return metrics_set_rankings;


        def get_iteration_signficant_cd_ranking(self, year, metrics_sets,iterations):
            metrics_set_rankings = {}

            condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))
            cd_map = to_dict(condorcment_file, by=["iteration", "gold", "pair"])
            for metrics_set in metrics_sets:
                iterations_rankings = {}
                for i in iterations:
                    iteration_it = cd_map[str(i)]
                    itr_gold_it = iteration_it[metrics_set]

                    remainding = list(complex_metrics)
                    iteration_ranking = []
                    limit = len(complex_metrics)
                    r = 1
                    while r < limit:
                        best = remainding[0]
                        for m in remainding:
                            if m ==best:
                                continue
                            pair_cd = itr_gold_it["-".join([m,best])]
                            if float(pair_cd["dis_correct_lef"])>float(pair_cd["dis_correct_right"]) and float(pair_cd["left_signficant"])<0.05:
                                best = m
                            else:
                                best = best
                        remainding.remove(best)
                        if (len(iteration_ranking) ==0):
                            iteration_ranking.append( ( best, r ))
                            r+=1
                            continue



                        for prev in iteration_ranking:
                            pair_cd = itr_gold_it["-".join([prev[0], best])]
                            if float(pair_cd["dis_correct_lef"]) < float(pair_cd["dis_correct_right"]) and float(pair_cd["right_signficant"])>=0.05:
                                print("Can not happend ")

                        if  float(pair_cd["dis_correct_lef"]) == float(pair_cd["dis_correct_right"]):
                            iteration_ranking.append((best,r-1))
                        else:
                            iteration_ranking.append((best, r))
                            r += 1


                    erro = False
                    for m,n in itertools.combinations(iteration_ranking,2):
                        pair_cd = itr_gold_it["-".join([m[0], n[0]])]

                        if float(pair_cd["dis_correct_lef"])< float(pair_cd["dis_correct_right"]) and m[1]< n[1]:
                            erro=True
                        if float(pair_cd["dis_correct_lef"]) > float(pair_cd["dis_correct_right"]) and m[1] > n[1]:
                            erro=True
                        if float(pair_cd["dis_correct_lef"]) == float(pair_cd["dis_correct_right"]) and m[1] != n[1]:
                            erro=True
                    if (erro):
                        print("Ranking is not right")

                    iterations_rankings[str(i)] = iteration_ranking

                metrics_set_rankings[metrics_set]=iterations_rankings

            return metrics_set_rankings;


        def get_iteration_mu_agre(self, year, mu_csv_file, metrics_sets):
            metrics_set_scores = {}
            for metrics_set in metrics_sets:
                iteration_scores = {}
                mu_analysis_file = os.path.join("data", "evals", year, mu_csv_file.format(year, metrics_set))
                condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))

                mu_map = to_dict(mu_analysis_file, by=["iteration", "dimensions", "metric"])
                cd_map = to_dict(condorcment_file, by=["iteration", "gold", "pair"])

                for i in range(1,11):
                    iteration_mu = mu_map[str(i)]
                    iteration_it = cd_map[str(i)]
                    iteration_scores[str(i)]= self.calculate_iteration_mu_cd_agreement(iteration_mu=iteration_mu,iteration_it=iteration_it);

                metrics_set_scores[metrics_set]=iteration_scores

            return metrics_set_scores;

        def generate_iteration_mu_cd_agreements_latex_report(self, years, mu_csv_file, metrics_sets,iterations):

            years_scores = {}

            for year in years:
                years_scores[year] = self.get_iteration_mu_cd_agreements(year, mu_csv_file, metrics_sets)


            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets)  + 1)))

            # colums = ["itr"]
            # for year in years:
            #     colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(metrics_sets), year))
            # print("\\toprule")
            # print ("& ".join(colums)+"\\\\")
            # print("\midrule")

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = ["Iteration"]
            for metric_set in metrics_sets:
                colums.append(set_pretty_header[metric_set])

            print ("& ".join(colums)+"\\\\")
            print("\midrule")
            colums = [" "]

            for year in years:
                print("\n\n \midrule "+year+ "\\\\"+" \midrule" )
                colums = [" "]
                for i in iterations:
                    colums = [str(i)]

                    for metric_set in metrics_sets:
                        colums.append("{0:.4f}".format(years_scores[year][metric_set][str(i)]["agree_prob"]))

                    print ("& ".join(colums) + "\\\\")


            print("\\end{tabular}\n")


        def generate_iteration_cd_ranking_latex_report(self, years, metrics_sets,iterations,sig=False):

            years_scores = {}

            for year in years:
                if sig:
                  years_scores[year] = self.get_iteration_signficant_cd_ranking(year, metrics_sets,iterations)
                else:
                    years_scores[year] = self.get_iteration_cd_ranking(year, metrics_sets, iterations)


            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(iterations)  + 1)))

            # colums = ["itr"]
            # for year in years:
            #     colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(metrics_sets), year))
            # print("\\toprule")
            # print ("& ".join(colums)+"\\\\")
            # print("\midrule")

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = [" "]
            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(iterations),set_pretty_header[metric_set]))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")

            colums = ["Metric/Iteration"]

            iterations = [str(i) for i in iterations]
            for metric_set in metrics_sets:
                for iteration in iterations:
                    colums.append("{{\it {}}}".format(iteration))

            print ("& ".join(colums) + "\\\\")
            print("\n")
            total_rankings = {}
            for year in years:
                total_rankings[year] = {}

                for mettrics_set in metrics_sets:
                    total_rankings[year][mettrics_set] = {}

                    for m in complex_metrics:
                        total_rankings[year][mettrics_set][m] = []

            for year in years:
                print("\n\midrule {} \\\\ \midrule".format(year))

                year_set_metrics_ranks = years_scores[year]

                for m in complex_metrics:
                    colums = [metrics_pretty_header[m]]
                    for metric_set in metrics_sets:
                        for iteration in iterations:
                            metric_ranks = year_set_metrics_ranks[metric_set][iteration]
                            for d in metric_ranks:
                                if (d[0] == m):
                                    colums.append(str(d[1]))
                    print ("& ".join(colums) + "\\\\")

            print("\\bottomrule")
            print("\\end{tabular}\n")




        def generate_iteration_cd_wins_latex_report(self, years, metrics_sets,iterations,sig=False):

            years_scores = {}

            for year in years:
                if sig:
                  years_scores[year] = self.get_iteration_signficant_cd_wins(year, metrics_sets,iterations)
                else:
                    years_scores[year] = self.get_iteration_signficant_cd_wins(year, metrics_sets, iterations)


            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(iterations)  + 1)))

            # colums = ["itr"]
            # for year in years:
            #     colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(metrics_sets), year))
            # print("\\toprule")
            # print ("& ".join(colums)+"\\\\")
            # print("\midrule")

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = [" "]
            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(iterations),set_pretty_header[metric_set]))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")

            colums = ["Metric/Iteration"]

            iterations = [str(i) for i in iterations]
            for metric_set in metrics_sets:
                for iteration in iterations:
                    colums.append("{{\it {}}}".format(iteration))

            print ("& ".join(colums) + "\\\\")
            print("\n")
            total_rankings = {}
            for year in years:
                total_rankings[year] = {}

                for mettrics_set in metrics_sets:
                    total_rankings[year][mettrics_set] = {}

                    for m in complex_metrics:
                        total_rankings[year][mettrics_set][m] = []
            year_wins_sums = {}

            for metric_set in metrics_sets:
                year_wins_sums[metric_set] = {}

                for iteration in iterations:
                    year_wins_sums[metric_set][iteration] ={}

                    for m in complex_metrics:
                        year_wins_sums[metric_set][iteration][m] = 0


            for year in years:
                print("\n\midrule {} \\\\ \midrule".format(year))

                year_set_metrics_ranks = years_scores[year]

                for m in complex_metrics:
                    year_wins_sums
                    colums = [metrics_pretty_header[m]]
                    for metric_set in metrics_sets:
                        for iteration in iterations:
                            metric_ranks = year_set_metrics_ranks[metric_set][iteration]
                            for d,wins in metric_ranks.items():
                                if (d == m):
                                    colums.append(str(wins))
                                    year_wins_sums[metric_set][iteration][m]+=wins
                    print ("& ".join(colums) + "\\\\")


            print("\n\midrule All years \\\\ \midrule")
            for m in complex_metrics:
                colums = [metrics_pretty_header[m]]
                for metric_set in metrics_sets:
                    for iteration in iterations:
                        for d, wins in year_wins_sums[metric_set][iteration].items():
                            if (d == m):
                                colums.append(str(wins))
                print ("& ".join(colums) + "\\\\")


            print("\\bottomrule")
            print("\\end{tabular}\n")


        def generate_iteration_cd_wins_ranking_latex_report(self, years, metrics_sets,iterations,sig=False):

            years_scores = {}

            for year in years:
                if sig:
                  years_scores[year] = self.get_iteration_signficant_cd_wins_rankings(year, metrics_sets,iterations)
                else:
                    years_scores[year] = self.get_iteration_signficant_cd_wins_rankings(year, metrics_sets, iterations)


            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(iterations)  + 1)))

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = [" "]
            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(iterations),set_pretty_header[metric_set]))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")

            colums = ["Metric/Iteration"]

            iterations = [str(i) for i in iterations]
            for metric_set in metrics_sets:
                for iteration in iterations:
                    colums.append("{{\it {}}}".format(iteration))

            print ("& ".join(colums) + "\\\\")
            print("\n")
            total_rankings = {}
            for year in years:
                total_rankings[year] = {}

                for metric_set in metrics_sets:
                    total_rankings[year][metric_set] = {}

                    for m in complex_metrics:
                        total_rankings[year][metric_set][m] = []
            year_rank_sum = {}

            for metric_set in metrics_sets:
                year_rank_sum[metric_set] = {}

                for iteration in iterations:
                    year_rank_sum[metric_set][iteration] ={}

                    for m in complex_metrics:
                        year_rank_sum[metric_set][iteration][m] = 0


            for year in years:
                print("\n\midrule \multicolumn{{{}}}{{{{c}}}}{{{}}} \\\\ \midrule".format(len(metrics_sets) * len(iterations) + 1,"TREC Dynamic Domain " + year))

                year_set_metrics_ranks = years_scores[year]

                for m in complex_metrics:
                    year_rank_sum
                    colums = [metrics_pretty_header[m]]
                    for metric_set in metrics_sets:
                        for iteration in iterations:
                            d = year_set_metrics_ranks[metric_set][iteration][m]
                            colums.append("{0:g}".format(d["sig_wins_rank"]))
                            year_rank_sum[metric_set][iteration][m]+=d["sig_wins_rank"]
                    print ("& ".join(colums) + "\\\\")


            print("\n\midrule All years \\\\ \midrule")
            for m in complex_metrics:
                colums = [metrics_pretty_header[m]]
                for metric_set in metrics_sets:
                    for iteration in iterations:
                        for d, wins in year_rank_sum[metric_set][iteration].items():
                            if (d == m):
                                colums.append(str(wins))
                print ("& ".join(colums) + "\\\\")


            print("\\bottomrule")
            print("\\end{tabular}\n")



        def print_rank_correlation(self, left_field,right_field,years, metrics_sets, iterations, mu_csv_file,print_all,print_all_years):
            iterations = [str(i) for i in iterations]
            all_analysis = self.get_all_analysis(metrics_sets, mu_csv_file, years)

            all_analysis_1eft = self.add_sum_rankings(all_analysis,left_field,print_all,print_all_years)
            all_analysis_right = self.add_sum_rankings(all_analysis, right_field, print_all, print_all_years)
            data = all_analysis_1eft.merge(all_analysis_right,on=["year","iteration","metric_set","metric"])

            print("\\begin{{tabular}}{{{}}}\n".format("c" * (len(metrics_sets)  + 1)))

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = ["{Iteration / Dimensions}"]
            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(1,set_pretty_header[metric_set]))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")





            for year in years + ["all"]:
                print("\n\midrule \multicolumn{{{}}}{{{{c}}}}{{{}}} \\\\ \midrule".format(
                    len(metrics_sets) + 1, "TREC Dynamic Domain " + year))

                for iteration in iterations + ["all"]:
                    colums = [iteration]
                    for metric_set in metrics_sets:
                        it_df = data[ (data["iteration"]==iteration) & (data["year"]==year) & (data["metric_set"]==metric_set)]
                        # it_df = data[ (data["iteration"]==iteration) & (data["metric_set"]==metric_set)]

                        left_field_ranks = list(it_df[left_field+"_rank"].values)
                        right_field_ranks = list(it_df[right_field+"_rank"].values);

                        left_field_ranks = list(it_df[left_field].values)
                        right_field_ranks = list(it_df[right_field].values);

                        left_field_ranking = list(it_df.sort_values(by=[left_field+"_rank"],ascending=True)["metric"].values)
                        right_field_ranking = list(it_df.sort_values(by=[right_field+"_rank"],ascending=True)["metric"].values)
                        tau,p = stats.kendalltau(left_field_ranking,right_field_ranking)


                        speanr, p = stats.spearmanr(left_field_ranks, right_field_ranks)

                        # cell = "{0:.2f}".format(tau)
                        cell="{0:.2f}".format(speanr)
                        if (p < 0.05):
                            cell+="$^*$"
                        colums.append(cell)

                    print ("& ".join(colums) + "\\\\")


            print("\\bottomrule")
            print("\\end{tabular}\n")

        def print_analysis_latex_report(self,field,years, metrics_sets,iterations,mu_csv_file,print_all=False,print_all_years=False):

            all_analysis = self.get_all_analysis(metrics_sets, mu_csv_file, years)
            all_analysis = self.add_sum_rankings(all_analysis,field,print_all,print_all_years)

            iterations = [str(i) for i in iterations]
            single_pretty_headers = {"strec": "Diversity (Div)", "p": "Topical Relevance (Rel.)",
                                     "ttime": "User Effort"}
            dimensin_pretty_short = {"strec": "Div.", "p": "Rel.", "ttime": "User effort"}

            set_pretty_header = {}

            iterations = iterations + ["all"] if print_all else iterations
            years = years + ["all"] if print_all_years else years
            print("\\begin{{tabular}}{{l{}}}\n".format("c" * (len(metrics_sets) * (len(iterations)))))
            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set] = single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set] = " and ".join(sorted(header))

            colums = [" "]
            scount = 1
            for metric_set in metrics_sets:
                scount+=1
                colums.append(
                    "{}\multicolumn{{{}}}{{{{c}}}}{{{}}}".format("\t"*scount,len(iterations), set_pretty_header[metric_set]))

            print("\\toprule")
            print ("\n& ".join(colums) + "\\\\")
            print("\midrule")

            colums = ["{\it Metric/Iteration}"]



            scount=0
            for metric_set in metrics_sets:
                scount+=1
                scolumns = []
                for iteration in iterations:
                    scolumns.append("{{\it {} }}".format(iteration))

                colums.append("\n"+("\t"*scount)+"& ".join(scolumns))

            print ("& ".join(colums) + "\\\\")
            print("\n")


            for year in years:
                print("\n\midrule \multicolumn{{{}}}{{{{c}}}}{{{}}} \\\\ \midrule".format(
                    len(metrics_sets) * (len(iterations))+1, "TREC Dynamic Domain " + year))


                for m in complex_metrics:
                    colums = [metrics_pretty_header[m]]
                    scount = 0

                    for metric_set in metrics_sets:
                        scount+=1
                        metric_set_df = all_analysis[(all_analysis["year"] == year) & (all_analysis["metric_set"] == metric_set) & (all_analysis["metric"] == m)]
                        mscoumns = []
                        for k, group in metric_set_df.groupby(["iteration"]):
                            if str(k).replace(".0","") in iterations:
                                v = group[field+"_rank"].values[0]
                                mscoumns.append("{0:g}".format(v))

                        colums.append(("\t"*scount)+" & ".join(mscoumns))
                    print ("\n & ".join(colums) +"\\\\")

            print("\\bottomrule")
            print("\end{tabular}")

        def get_all_analysis(self, metrics_sets, mu_csv_file, years):
            iterations = [str(i) for i in range(1,11)]
            years_cd_sig_wins_ranking = {}

            years_cd_ranking = {}
            years_mu_rankings = {}
            for year in years:
                years_cd_sig_wins_ranking[year] = self.get_iteration_signficant_cd_wins_rankings(year, metrics_sets,
                                                                                                 iterations)
                years_cd_ranking[year] = self.get_iteration_cd_ranking(year, metrics_sets, iterations)
                years_mu_rankings[year] = self.get_year_iteration_mu(year, mu_csv_file, metrics_sets)

                years_mu_rankings[year] = self.calculate_mu_wins_rank(iterations, metrics_sets, year, years_mu_rankings)
            all_analysis = []
            for year in years:
                for metric_set in metrics_sets:
                    for iteration in iterations:
                        for m in complex_metrics:
                            data = {"metric": m,
                                    "sig_wins_rank": years_cd_sig_wins_ranking[year][metric_set][iteration][m][
                                        "sig_wins_rank"],
                                    "sig_wins": years_cd_sig_wins_ranking[year][metric_set][iteration][m][
                                        "sig_wins"],
                                    "it_rank": years_cd_ranking[year][metric_set][iteration][m]["it_rank"],
                                    "mu_rank": years_mu_rankings[year][metric_set][iteration][m]["mu_rank"],
                                    "year": year,
                                    "iteration": iteration,
                                    "mu_wins_rank": years_mu_rankings[year][metric_set][iteration][m]["mu_wins_rank"],
                                    "mu_wins": years_mu_rankings[year][metric_set][iteration][m]["mu_wins"],
                                    "metric_set": metric_set,
                                    }
                            all_analysis.append(data)
            return all_analysis

        def get_iteration_mu_ranking(self,year,mu_csv_file,metrics_sets,iterations):
            years_scores = {}

            for year in years:
                years_scores[year] = self.get_year_iteration_mu(year, mu_csv_file, metrics_sets)

            for i in iterations:
                for year in years:
                    for metric_set in metrics_sets:
                        metrics_mus = {}
                        for m in complex_metrics:
                            metrics_mus[m] = years_scores[year][metric_set][str(i)][m]["mu"]

                        metrics_mus = sorted(metrics_mus.iteritems(), key=lambda (k, v): (v, k), reverse=True)


        def generate_iteration_mu_metric_ranking_latex_report(self, years, mu_csv_file, metrics_sets,iterations):
            years_scores = {}

            for year in years:
                years_scores[year] = self.get_year_iteration_mu(year, mu_csv_file, metrics_sets)

            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(iterations)  + 1)))

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = [" "]
            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(iterations),set_pretty_header[metric_set]))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")

            colums = ["Metric/Iteration"]

            iterations = [str(i) for i in iterations]
            for metric_set in metrics_sets:
                for iteration in iterations:
                    colums.append("{{\it {}}}".format(iteration))

            print ("& ".join(colums) + "\\\\")
            print("\n")
            total_rankings = {}
            for year in years:
                total_rankings[year] = {}

                for metric_set in metrics_sets:
                    total_rankings[year][metric_set] = {}

                    for m in complex_metrics:
                        total_rankings[year][metric_set][m] = []
            year_rank_sum = {}

            for metric_set in metrics_sets:
                year_rank_sum[metric_set] = {}

                for iteration in iterations:
                    year_rank_sum[metric_set][iteration] ={}

                    for m in complex_metrics:
                        year_rank_sum[metric_set][iteration][m] = 0


            for year in years:
                print("\n\midrule \multicolumn{{{}}}{{{{c}}}}{{{}}} \\\\ \midrule".format(len(metrics_sets) * len(iterations) + 1,"TREC Dynamic Domain " + year))

                year_set_metrics_ranks = self.calculate_metrics_rank(iterations,metrics_sets,year,years_scores)

                for m in complex_metrics:
                    year_rank_sum
                    colums = [metrics_pretty_header[m]]
                    for metric_set in metrics_sets:
                        for iteration in iterations:
                            d = year_set_metrics_ranks[metric_set][iteration][m]
                            colums.append("{0:g}".format(d["mu_rank"]))
                            year_rank_sum[metric_set][iteration][m]+=d["mu_rank"]
                    print ("& ".join(colums) + "\\\\")



        def generate_iteration_mu_wins_ranking_latex_report(self, years, mu_csv_file, metrics_sets,iterations):
            years_scores = {}

            for year in years:
                years_scores[year] = self.get_year_iteration_mu(year, mu_csv_file, metrics_sets)

            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(iterations)  + 1)))

            single_pretty_headers = {"strec":"Diversity (Div)","p":"Topical Relevance (Rel.)","ttime":"User Effort"}
            dimensin_pretty_short = {"strec":"Div.","p":"Rel.","ttime":"User effort"}

            set_pretty_header = {}

            for metric_set in metrics_sets:
                if metric_set in single_pretty_headers:
                    set_pretty_header[metric_set]=single_pretty_headers[metric_set]
                else:
                    header = []
                    for dim in metric_set.split("+"):
                        header.append(dimensin_pretty_short[dim])

                    set_pretty_header[metric_set]=" and ".join(sorted(header))


            colums = [" "]
            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{{{c}}}}{{{}}}".format(len(iterations),set_pretty_header[metric_set]))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")

            colums = ["Metric/Iteration"]

            iterations = [str(i) for i in iterations]
            for metric_set in metrics_sets:
                for iteration in iterations:
                    colums.append("{{\it {}}}".format(iteration))

            print ("& ".join(colums) + "\\\\")
            print("\n")
            total_rankings = {}
            for year in years:
                total_rankings[year] = {}

                for metric_set in metrics_sets:
                    total_rankings[year][metric_set] = {}

                    for m in complex_metrics:
                        total_rankings[year][metric_set][m] = []
            year_rank_sum = {}

            for metric_set in metrics_sets:
                year_rank_sum[metric_set] = {}

                for iteration in iterations:
                    year_rank_sum[metric_set][iteration] ={}

                    for m in complex_metrics:
                        year_rank_sum[metric_set][iteration][m] = 0


            for year in years:
                print("\n\midrule \multicolumn{{{}}}{{{{c}}}}{{{}}} \\\\ \midrule".format(len(metrics_sets) * len(iterations) + 1,"TREC Dynamic Domain " + year))

                year_set_metrics_ranks = self.calculate_mu_wins_rank(iterations,metrics_sets,year,years_scores)

                for m in complex_metrics:
                    year_rank_sum
                    colums = [metrics_pretty_header[m]]
                    for metric_set in metrics_sets:
                        for iteration in iterations:
                            d = year_set_metrics_ranks[metric_set][iteration][m]
                            colums.append("{0:.2f}".format(d["mu_wins_rank"]))
                            year_rank_sum[metric_set][iteration][m]+=d["mu_wins_rank"]
                    print ("& ".join(colums) + "\\\\")



        def generate_iteration_mu_metric_summarized_ranking_latex_report(self, years,iterations=[], mu_csv_file="", metrics_sets=[]):
            years_scores = {}

            for year in years:
                years_scores[year] = self.get_year_iteration_mu(year, mu_csv_file, metrics_sets)

            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(iterations) + 1)))

            colums = [" "]

            for metric_set in metrics_sets:
                colums.append("\multicolumn{{{}}}{{c}}{{{}}}".format(len(iterations),metric_set))

            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")

            colums = ["Metric/Iteration"]


            for metrics_set in metrics_sets:
                for iteration in iterations:
                    colums.append(iteration)

            print ("& ".join(colums) + "\\\\")

            total_rankings = {}
            for year in years:
                total_rankings[year] = {}

                for mettrics_set in metrics_sets:
                    total_rankings[year][mettrics_set]={}

                    for m in complex_metrics:
                        total_rankings[year][mettrics_set][m] = []

            for year in years:
                print("\n\n \midrule {} \\\\\midrule".format(year))

                year_set_metrics_ranks = self.calculate_metrics_rank(iterations, metrics_sets, year, years_scores)

                for m in complex_metrics:
                    colums = [m]
                    for metric_set in metrics_sets:
                       for iteration in iterations:
                          metric_ranks = year_set_metrics_ranks["-".join([year, metric_set, iteration])]
                          for d in metric_ranks:
                              if (d["metric"]==m):
                                colums.append("{0:g}".format(d["rank"]))
                    print ("& ".join(colums)+"\\\\")

            # colums = ["avg"]
            #
            # for year in years:
            #     for metric_set in metrics_sets:
            #         avg_ranking = {}
            #         for m in complex_metrics:
            #             avg_ranking[m]=sum(total_rankings[year][metric_set][m])/float(len(total_rankings[year][mettrics_set][m]))
            #
            #         metrics_mus = sorted(avg_ranking.iteritems(), key=lambda (k, v): (v, k), reverse=False)
            #         ranking = ">".join([k for k, v in metrics_mus])
            #         colums.append(ranking)
            #
            #
            # print ("& ".join(colums) + "\\\\")
            # print("\midrule \midrule")
            #
            print("\\bottomrule")
            print("\\end{tabular}\n")

        def calculate_metrics_rank(self, iterations, metrics_sets, year, years_scores):
            year_set_metrics_ranks = {}
            for metric_set in metrics_sets:
                metric_set_mu_rank = {}
                for iteration in iterations:
                    metrics_mus = {}

                    for m in complex_metrics:
                        metrics_mus[m] = years_scores[year][metric_set][iteration][m]["mu"]

                    mus_df = pd.DataFrame.from_dict(
                        {"metric": list(metrics_mus.keys()), "mu": list(metrics_mus.values())})
                    # metrics_mus = sorted(metrics_mus.iteritems(), key=lambda (k,v): (v,k),reverse=True)
                    mus_df["mu_rank"] = mus_df["mu"].rank(method=tie_data_rank, ascending=False)
                    mus_df = mus_df.sort_values(by=["mu_rank"])
                    # print(mus_df)

                    metrics_mus_ranks = mus_df.to_dict(orient="records")

                    final_ranking = {}
                    for data in metrics_mus_ranks:
                        final_ranking[data["metric"]]=data
                    metric_set_mu_rank[iteration]= final_ranking

                year_set_metrics_ranks [metric_set] = metric_set_mu_rank
            return year_set_metrics_ranks


        def calculate_mu_wins_rank(self, iterations, metrics_sets, year, years_scores,sign_wins=False):
            year_totals = {"2015":(118 * 31 * 30),
                           "2016":(53 * 21 * 20),
                           "2017":(60 * 11 * 10)}
            year_set_metrics_ranks = {}
            for metric_set in metrics_sets:
                metric_set_mu_rank = {}
                for iteration in iterations:
                    metrics_mus = {}

                    for m in complex_metrics:
                        metrics_mus[m] = years_scores[year][metric_set][iteration][m]

                    metrics_wins = {}
                    for m in complex_metrics:
                        metrics_wins[m] = 0

                    for m,n in itertools.combinations(metrics_mus.keys(),2):

                        if (metrics_mus[m]["mu"] > metrics_mus[n]["mu"]):
                            pvalue=scipy.stats.binom_test(x=metrics_mus[m]["m_set_ij"],n=year_totals[year],p=metrics_mus[m]["m_set_ij"]/year_totals[year],alternative="greater")

                            # if (pvalue < 0.05):
                            metrics_wins[m]+=1
                        elif(metrics_mus[m]["mu"] < metrics_mus[n]["mu"]):
                            pvalue = scipy.stats.binom_test(x=metrics_mus[n]["m_set_ij"], n=year_totals[year],
                                                            p=metrics_mus[n]["m_set_ij"] / year_totals[year],
                                                            alternative="greater")
                            # if (pvalue < 0.05):
                            metrics_wins[n]+=1

                    metric_df = pd.DataFrame.from_dict(
                        {"metric": list(metrics_mus.keys()), "mu": list(metrics_mus.values())})
                    metric_df["mu_rank"] = metric_df["mu"].rank(method=tie_data_rank, ascending=False)

                    mus_df = pd.DataFrame.from_dict(
                        {"metric": list(metrics_wins.keys()), "wins": list(metrics_wins.values())})
                    metric_df["mu_wins_rank"] = mus_df["wins"].rank(method=tie_data_rank, ascending=False)
                    metric_df["mu_wins"] = mus_df["wins"]

                    metric_df = metric_df.sort_values(by=["mu_rank"])
                    metrics_mus_ranks = metric_df.to_dict(orient="records")

                    final_ranking = {}
                    for data in metrics_mus_ranks:
                        final_ranking[data["metric"]]=data
                    metric_set_mu_rank[iteration]= final_ranking

                year_set_metrics_ranks [metric_set] = metric_set_mu_rank
            return year_set_metrics_ranks



        def generate_iteration_mu_latex_report(self, years, mu_csv_file, metrics_sets):

            years_scores = {}

            for year in years:
                years_scores[year] = self.get_year_iteration_mu(year, mu_csv_file, metrics_sets)


            print("\\begin{{tabular}}{{{}}}\n".format("l" * (len(metrics_sets) * len(years) + 2)))


            colums = ["itr","metric"]


            for year in years:
                colums.append("\multicolumn{{{}}}{{c}}{{{}}}".format(len(metrics_sets), year))
            print("\\toprule")
            print ("& ".join(colums)+"\\\\")
            print("\midrule")
            colums = [" "," "]
            for year in years:
                for mettrics_set in metrics_sets:
                    colums.append(mettrics_set.replace("-metrics",""))

            print ("& ".join(colums) + "\\\\")
            print("\midrule \midrule")

            for i in [1,2,5,8,10]:
                colums = [str(i)]
                for m in complex_metrics:
                    colums = [str(i),m]
                    for year in years:
                        for metric_set in metrics_sets:
                            colums.append("{0:.4f}".format(years_scores[year][metric_set][str(i)][m]["mu"]))

                    print ("& ".join(colums)+"\\\\")
                print("\midrule \midrule")

            print("\\end{tabular}\n")






        def pair_mu_cd_agreement_simple_metrics(self):
            for year in years:
                mu_analysis_file = os.path.join("data", "evals", year, "trec-{}-mu-simple-metrics.csv".format(year))
                condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))

                mu_map = to_dict(mu_analysis_file, by=["iteration","dimensions", "metric"])
                cd_map = to_dict(condorcment_file, by=["iteration","gold", "pair" ])

                print("\\midrule " + year + "\\\\\\midrule")
                seen = []

                for m in complex_metrics:
                    for n in complex_metrics:
                        if n == m:
                            continue

                        if (m, n) in seen or (n, m) in seen:
                            continue
                        else:
                            seen.append((m, n))

                        same_count = 0
                        count = 0

                        for i in range(1,11):
                            iteration_mu = mu_map[str(i)]
                            iteration_it = cd_map[str(i)]


                                    # for gold in ["p","strec","ttime","strec+p","strec+p+ttime"]:
                            for gold in ["strec+p+ttime"]:
                                gold_mu_scores = iteration_mu[gold]
                                gold_cd_scores = iteration_it[gold]


                                m_it_mu = gold_mu_scores[m]
                                n_it_mu = gold_mu_scores[n]
                                mu_diff = float(m_it_mu["mu"]) - float(n_it_mu["mu"])
                                cd_diff = float(gold_cd_scores[m + "-" + n]["dis_correct_lef"]) - float(gold_cd_scores[m + "-" + n]["dis_correct_right"])

                                if (mu_diff * cd_diff > 0) or (mu_diff ==0 and cd_diff ==0):
                                    same_count+=1

                                count+=1

                        print("&".join([m+","+n, "{:.4f}".format(float(same_count)/count)]) + "\\\\")



        def mu_metric_correlation_simple_metrics(self,):
            for year in years:
                mu_analysis_file = os.path.join("data", "evals", year, "trec-{}-mu-simple-metrics.csv".format(year))
                condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))

                mu_map = to_dict(mu_analysis_file,by=["dimensions","metric","iteration"])
                cd_map = to_dict(condorcment_file,by=["gold","pair","iteration"])
                seen = []
                print("\\midrule " + year + "\\\\\\midrule")
                for m in complex_metrics:
                    for n in complex_metrics:
                        if n==m:
                            continue

                        if (m,n) in seen or (n,m) in seen:
                            continue
                        else:
                            seen.append((m,n))

                        # for gold in ["p","strec","ttime","strec+p","strec+p+ttime"]:
                        for gold in ["strec+p+ttime"]:
                            gold_mu_scores = mu_map[gold]

                            pair_mu_scores = []

                            m_it_mu = gold_mu_scores[m]
                            n_it_mu = gold_mu_scores[n]

                            for i in range(1,11):
                                m_mu = m_it_mu[str(i)]
                                n_mu = n_it_mu[str(i)]
                                pair_mu_scores.append(float(m_mu["mu"])-float(n_mu["mu"]))



                            gold_cd_scores = cd_map[gold]
                            pair_cd_scores=[]
                            pair_score = gold_cd_scores[m+"-"+n]

                            for i in range(1, 11):

                                pair_cd_scores.append(float(pair_score[str(i)]["dis_correct_diff"]))

                            correlation,pvalue= stats.spearmanr(pair_cd_scores,pair_mu_scores)

                            print("&".join([m+","+n,"{:.4f}".format(float(correlation)),"{:.4f}".format(float(pvalue))])+"\\\\")




        def mu_metric_correlation_complex_metrics(self,):
            for year in years:
                # mu_simple_metrics(eval_path=os.path.join("data", "evals", year), year=year,
                #                   metrics=metrics,mu_analysis_file=os.path.join("data", "evals", year,"trec-{}-mu-simple-metrics.csv".format(year)))
                # mu_complex_metrics(eval_path=os.path.join("data", "evals", year), year=year,
                #                    metrics=complex_metrics,mu_analysis_file=os.path.join("data", "evals", year,"trec-{}-mu-complex-metrics.csv".format(year)))
                # mu_complex_metrics(eval_path=os.path.join("data", "evals", year), year=year,
                #                    metrics=all_metrics,mu_analysis_file=os.path.join("data", "evals", year,"trec-{}-mu-all-metrics.csv".format(year)))

                mu_analysis_file = os.path.join("data", "evals", year, "trec-{}-mu-complex-metrics.csv".format(year))
                condorcment_file = os.path.join("data", "evals", year, "metric-condorcent-{}.csv".format(year))

                mu_map = to_dict(mu_analysis_file, by=["dimensions", "metric", "iteration"])
                cd_map = to_dict(condorcment_file, by=["gold", "pair", "iteration"])
                seen = []
                print("\\midrule " + year + "\\\\\\midrule")
                for m in complex_metrics:
                    for n in complex_metrics:
                        if n == m:
                            continue

                        if (m, n) in seen or (n, m) in seen:
                            continue
                        else:
                            seen.append((m, n))

                        # for gold in ["p","strec","ttime","strec+p","strec+p+ttime"]:
                        for gold in ["strec+p+ttime"]:
                            gold_mu_scores = mu_map["complex-metrics"]

                            pair_mu_scores = []

                            m_it_mu = gold_mu_scores[m]
                            n_it_mu = gold_mu_scores[n]

                            for i in range(1, 11):
                                m_mu = m_it_mu[str(i)]
                                n_mu = n_it_mu[str(i)]
                                pair_mu_scores.append(float(m_mu["mu"]) - float(n_mu["mu"]))

                            gold_cd_scores = cd_map[gold]
                            pair_cd_scores = []
                            pair_score = gold_cd_scores[m + "-" + n]

                            for i in range(1, 11):
                                pair_cd_scores.append(float(pair_score[str(i)]["dis_correct_diff"]))

                            correlation, pvalue = stats.spearmanr(pair_cd_scores, pair_mu_scores)

                            print("&".join([m + ", " + n, "{:.4f}".format(float(correlation)),
                                            "{:.4f}".format(float(pvalue))]) + "\\\\")

        def get_iteration_signficant_cd_wins(self, year, metrics_sets, iterations):
            metrics_set_rankings = {}

            condorcment_file = os.path.join("..","data", "evals", year, "metric-condorcent-{}.csv".format(year))
            cd_map = to_dict(condorcment_file, by=["iteration", "gold", "pair"])
            for metrics_set in metrics_sets:
                iterations_rankings = {}
                for i in iterations:
                    iteration_it = cd_map[str(i)]
                    itr_gold_it = iteration_it[metrics_set]
                    win_counts = {}
                    for m in complex_metrics:
                        win_counts[m] = 0

                    for left, right in itertools.combinations(complex_metrics,2):
                            if ("-".join([left,right]) not in itr_gold_it.keys()):
                              print("Missing  pair " + "-".join([left,right]) + " " + metrics_set + " it: " + str(i))
                              continue
                            pair_cd = itr_gold_it["-".join([left,right])]
                            if float(pair_cd["dis_correct_left"]) > float(pair_cd["dis_correct_right"]) and float(pair_cd["left_signficant"]) < 0.05:
                                win_counts[left]+=1
                            elif float(pair_cd["dis_correct_left"]) < float(pair_cd["dis_correct_right"]) and float(pair_cd["right_signficant"]) < 0.05:
                                win_counts[right]+=1

                    iterations_rankings[str(i)] = win_counts

                metrics_set_rankings[metrics_set] = iterations_rankings

            return metrics_set_rankings;


        def get_iteration_signficant_cd_wins_rankings(self, year, metrics_sets, iterations):
            metrics_set_ranking = {}

            condorcment_file = os.path.join("..","data", "evals", year, "metric-condorcent-{}.csv".format(year))
            cd_map = to_dict(condorcment_file, by=["iteration", "gold", "pair"])
            for metrics_set in metrics_sets:
                iterations_rankings = {}
                for i in iterations:
                    iteration_it = cd_map[str(i)]
                    itr_gold_it = iteration_it[metrics_set]
                    win_counts = {}
                    for m in complex_metrics:
                        win_counts[m] = 0

                    for left, right in itertools.combinations(complex_metrics,2):
                            if "-".join([left,right]) not in itr_gold_it.keys():
                                sys.stderr.write("Missing " + "-".join([left, right]) +" Year "+year+" Iteration " +str(i) + " "+metrics_set+ "\n")
                                continue
                            pair_cd = itr_gold_it["-".join([left,right])]
                            if float(pair_cd["dis_correct_left"]) > float(pair_cd["dis_correct_right"]) and float(pair_cd["left_signficant"]) < 0.05:
                                win_counts[left]+=1
                            elif float(pair_cd["dis_correct_left"]) < float(pair_cd["dis_correct_right"]) and float(pair_cd["right_signficant"]) < 0.05:
                                win_counts[right]+=1

                    iteration_wins_ranking = []
                    for m, wins  in win_counts.items():
                        iteration_wins_ranking.append({"metric":m,"sig_wins":wins})

                    ranking_df = pd.DataFrame.from_dict(iteration_wins_ranking)
                    ranking_df["sig_wins_rank"] = ranking_df["sig_wins"].rank(method="dense",ascending=False)
                    ranking_df= ranking_df.sort_values(by="sig_wins_rank")


                    records = ranking_df.to_dict(orient="records")
                    final_ranking = {}

                    for r in records:
                        final_ranking[r["metric"]]=r;

                    iterations_rankings[str(i)] = final_ranking

                metrics_set_ranking[metrics_set] = iterations_rankings

            return metrics_set_ranking;

        def add_sum_rankings(self,all_analysis,field,print_all, print_all_years):
            filterd = []
            columns = ["year", "iteration", "metric_set", "metric",field,field+"_rank"]


            for row in all_analysis:
                d = {}
                for column in columns:
                    d[column] = row[column]

                filterd.append(d)

            all_analysis = pd.DataFrame(filterd)

            for k, group in all_analysis.groupby(["iteration", "metric_set"]):
                metric_sum = []
                for m, ggroup in group.groupby(["metric"]):
                    assert len(ggroup[field].values)==3
                    sum = ggroup[field].sum()
                    r = {"year": "all", "iteration": k[0], "metric_set": k[1], "metric": m, field: sum}
                    metric_sum.append(r)

                metric_sum_rank_df = pd.DataFrame(metric_sum)
                metric_sum_rank_df[field + "_rank"] = metric_sum_rank_df[field].rank(method=tie_data_rank,
                                                                                     ascending=False)
                metric_sum = metric_sum_rank_df.to_dict(orient="records")

                for d in metric_sum:
                    if (print_all_years):
                        filterd.append(d)

            all_analysis = pd.DataFrame(filterd)
            for k, group in all_analysis.groupby(["year", "metric_set"]):
                metric_sum = []
                for m, ggroup in group.groupby(["metric"]):
                    assert len(ggroup[field].values) == 10
                    sum = ggroup[field].sum()
                    metric_sum.append({"year": k[0], "metric_set": k[1], "iteration": "all", "metric": m, field: sum})

                metric_sum_rank_df = pd.DataFrame(metric_sum)
                metric_sum_rank_df[field + "_rank"] = metric_sum_rank_df[field].rank(method=tie_data_rank,
                                                                                     ascending=False)
                metric_sum = metric_sum_rank_df.to_dict(orient="records")

                for d in metric_sum:
                    if (print_all):
                        filterd.append(d)
            all_analysis = pd.DataFrame(filterd)
            return all_analysis


    # for year in ["2015"]:
    # for year in ["2016"]:
    # for year in ["2017"]:
    # # for year in ["2017-extra"]:
    #     generate_mu_simple_metrics(eval_path=os.path.join("..","data", "evals", year), year=year,
    #                       metrics=metrics,mu_analysis_file=os.path.join("..","data", "evals", year,"trec-{}-mu-simple-metrics.csv".format(year)))

    trec_analysis = Mu_TREC_Analysis()



    # for year in years:
    #     trec_analysis.generate_iteration_mu_cd_agreement(year=year,mu_csv_file="trec-{}-mu-{}.csv",mettrics_set=["simple-metrics","complex-metrics"]);
    #
    #
    # for year in years:
    #     trec_analysis.generate_iteration_mu_cd_agreement(year, "trec-{}-mu-{}.csv", "complex-metrics");

    # for year in years:
    #     trec_analysis.generate_pair_mu_cd_agreement(year, "trec-{}-mu-{}.csv", "simple-metrics",metrics);
    # #
    # for year in years:
    #     trec_analysis.generate_pair_mu_cd_agreement(year, "trec-{}-mu-{}.csv", "simple-metrics");

    # trec_analysis.generate_iteration_mu_cd_agreements_latex_report(years=years,metrics_sets=M_GS_Labels,mu_csv_file="trec-{}-mu-simple-metrics.csv",iterations=[1,2,5,8,10])
    # trec_analysis.generate_iteration_cd_ranking_latex_report(years=["2015","2016","2017"],metrics_sets=M_GS_Labels,iterations=[1,2,5,8,10],sig=False)
    # trec_analysis.generate_iteration_cd_wins_latex_report(years=["2016"], metrics_sets=M_GS_Labels,iterations=[1, 2, 5, 8, 10], sig=False)
    # trec_analysis.generate_iteration_cd_wins_ranking_latex_report(years=["2015","2016","2017"], metrics_sets=M_GS_Labels,iterations=[1, 2, 5, 8, 10], sig=False)
    #
    # trec_analysis.generate_iteration_mu_latex_report(years=years,metrics_sets=["simple-metrics"], mu_csv_file="trec-{}-mu-{}.csv")
    # trec_analysis.generate_iteration_mu_wins_ranking_latex_report(years=["2016"], metrics_sets=M_GS_Labels,mu_csv_file="trec-{}-mu-simple-metrics.csv",iterations=[str(i) for i in [1,2,5,8,10]])
    # trec_analysis.generate_iteration_mu_metric_summarized_ranking_latex_report(years=["2015","2016","2017"], metrics_sets=M_GS_Labels,mu_csv_file="trec-{}-mu-simple-metrics.csv",iterations=["1","2","5","8","10"])
    # trec_analysis.print_analysis_latex_report(field="mu_wins",years=["2016", "2017"], metrics_sets=M_GS_Labels,mu_csv_file="trec-{}-mu-simple-metrics.csv", iterations=[1, 2, 5, 8, 10],print_all=False)
    # trec_analysis.print_analysis_latex_report(field="mu_wins", years=["2015","2016", "2017"], metrics_sets=M_GS_Labels,mu_csv_file="trec-{}-mu-simple-metrics.csv", iterations=[1, 3, 7, 10],print_all_years=True,print_all=True)
    # trec_analysis.print_analysis_latex_report(field="sig_wins", years=["2015", "2016", "2017"], metrics_sets=M_GS_Labels,
    #                                           mu_csv_file="trec-{}-mu-simple-metrics.csv", iterations=[1, 3, 10],
    #                                           print_all_years=True, print_all=True)

    trec_analysis.print_analysis_latex_report(field="mu_wins", years=["2015", "2016", "2017"],
                                              metrics_sets=M_GS_Labels,
                                              mu_csv_file="trec-{}-mu-simple-metrics.csv", iterations=[1, 3, 10],
                                              print_all_years=True, print_all=True)

    # trec_analysis.print_rank_correlation(left_field="mu_wins", right_field = "sig_wins",years=["2015", "2016", "2017"], metrics_sets=M_GS_Labels,
    #                                      mu_csv_file="trec-{}-mu-simple-metrics.csv", iterations=[1, 3, 10],print_all=True,print_all_years=True)

    # data = [{"wins":5,"metric":"a"},{"wins":4,"metric":"b"},{"wins":4,"metric":"c"},{"wins":3,"metric":"d"}]
    # data_df = pd.DataFrame(data)
    # data_df["rank_avg"] = data_df["wins"].rank(method="average",ascending=False)
    # data_df["rank_min"] = data_df["wins"].rank(method="min", ascending=False)
    # data_df["rank_max"] = data_df["wins"].rank(method="max", ascending=False)
    # data_df["rank_dense"] = data_df["wins"].rank(method="dense", ascending=False)
    # print (data_df)