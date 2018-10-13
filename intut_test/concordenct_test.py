import itertools

import operator
import os

import pandas as pd
from docutils.nodes import option_string
from scipy import stats

from analysis import to_dict
import matplotlib.pyplot as plt


class CondorcentTestReport:
    def comp_no_zero(X):
        verdict = all(x >0 for x in X) or all(x < 0 for x in X)
        return verdict

    def comp_with_zero(X):
        verdict = all(x >=0 for x in X) or all(x < 0 for x in X)
        return verdict


    #
    def same_no_tie(self,y,X):
        verdict = all(y * x > 0 for x in X)
        return verdict

    def same_with_tie(self,y,X):
        verdict = all(y * x >=0 for x in X)
        return verdict

    def get_condorcnet_line(self,Mg,c,M1,M2,disagreements,total,dis_correct_1,dis_correct_2,agreements,agree_wrong_1,agree_wrong_2, left_sig_level,right_sig_level):

       line = "{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{:.4f},{:.4f},{:.4f},{:.4f}".format(Mg, str(c), M1 + "-" + M2, M1, M2,
                       disagreements,
                      (float(disagreements) / total),
                      (dis_correct_1 / float(disagreements)),
                      (dis_correct_2 / float(disagreements)),
                      (dis_correct_1 / float(disagreements)) - (dis_correct_2 / float(disagreements)),
                      left_sig_level,
                      right_sig_level,
                       agreements,
                      (float(agreements) / total),
                       (agree_wrong_1 / float(agreements)),
                       (agree_wrong_2 / float(agreements)),
                       (agree_wrong_1 / float(agreements)) - (agree_wrong_2 / float(agreements))
               )
       return line



    def condorcent_test_nmg(self,condorcment_file,csfile,MG,metrics,topics,runs):
        metrics_df = to_dict(csfile)


        print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format("gold", "iteration", "pair", "left", "right",
                                                                          "disagreement_count",
                                                                          "disagreement_per", "dis_correct_left",
                                                                          "dis_correct_right", "dis_correct_diff",
                                                                          "left_signficant",
                                                                          "right_signficant",
                                                                          "agreement_count",
                                                                          "agreement_per", "agree_wrong_lef",
                                                                          "agree_wrong_right",
                                                                          "agree_wrong_diff"
                                                                          ))


        cd_file = open(condorcment_file, mode="w")
        cd_file.write(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format("gold", "iteration", "pair", "left", "right",
                                                                      "disagreement_count",
                                                                      "disagreement_per", "dis_correct_left",
                                                                      "dis_correct_right", "dis_correct_diff",
                                                                      "left_signficant",
                                                                      "right_signficant",
                                                                      "agreement_count",
                                                                      "agreement_per", "agree_wrong_lef",
                                                                      "agree_wrong_right",
                                                                      "agree_wrong_diff"
                                                                      ))


        for M1 in metrics:
            for M2 in metrics:
                if M1 == M2:
                    continue

                for c in [1,2,3,4,5,6,7,8,9,10]:
                    for Mg in MG:
                        total = 0
                        ties = 0
                        disagreements, dis_correct_1, dis_correct_2 = 0, 0, 0

                        agreements, agree_wrong_1, agree_wrong_2 = 0, 0, 0

                        for r_1, r_2 in itertools.combinations(runs,2):
                                    if r_1 == r_2:
                                        continue

                                    for t in topics:
                                        if t == "all":
                                            continue

                                        if t not in metrics_df[r_1].keys() or t not in metrics_df[r_2].keys():
                                            continue

                                        total += 1
                                        M1_r1 = metrics_df[r_1][t][str(c)][M1]
                                        M2_r1 = metrics_df[r_1][t][str(c)][M2]

                                        if (str(c) not in metrics_df[r_2][t].keys() ):
                                            print ("missing iteration")
                                            continue
                                        M1_r2 = metrics_df[r_2][t][str(c)][M1]
                                        M2_r2 = metrics_df[r_2][t][str(c)][M2]


                                        M1_d = float(M1_r1) - float(M1_r2)
                                        M2_d = float(M2_r1) - float(M2_r2)


                                        MG_Ds = []

                                        for mg in Mg:
                                            Mg_r1 = metrics_df[r_1][t][str(c)][mg]
                                            Mg_r2 = metrics_df[r_2][t][str(c)][mg]
                                            mg_d = float(Mg_r1) - float(Mg_r2)

                                            MG_Ds.append(mg_d)

                                        if (M1_d * M2_d < 0):
                                            disagreements += 1

                                            # if comp_no_zero([M1_d] + MG_Ds):
                                            if self.same_no_tie(M1_d,MG_Ds):
                                                dis_correct_1 += 1

                                            # if comp_no_zero([M2_d]+ MG_Ds):
                                            if self.same_no_tie(M2_d,MG_Ds):
                                                dis_correct_2 += 1

                                            # if comp_with_zero([M1_d,M2_d]+MG_Ds):
                                            if self.same_with_tie(M1_d,MG_Ds) and self.same_with_tie(M2_d,MG_Ds):
                                                ties += 1

                                        else:
                                            agreements += 1

                                            if not self.same_with_tie(M1_d, MG_Ds):
                                                agree_wrong_1 += 1

                                            if not self.same_with_tie(M2_d,MG_Ds):
                                                agree_wrong_2 += 1


                        if disagreements > 0:
                                right_sig_level = 1
                                left_sig_level = 1

                                assert ((dis_correct_1 + dis_correct_2 ) <=disagreements)
                                dis = dis_correct_2 + dis_correct_1
                                if dis_correct_1 > dis_correct_2:
                                    left_sig_level = stats.binom_test(dis_correct_1,dis)

                                if dis_correct_2 > dis_correct_1:
                                    right_sig_level = stats.binom_test(dis_correct_2,dis)


                                if dis != disagreements:
                                    dummpy=0
                                line = self.get_condorcnet_line("+".join(Mg),c,M1,M2,disagreements,total,dis_correct_1,dis_correct_2,agreements,agree_wrong_1,agree_wrong_2, left_sig_level,right_sig_level)
                                print (line)
                                cd_file.write(line+"\n")

        cd_file.close()



    def condorcent_test_simple_disagreements(self,csfile,MG,metrics,topics,runs):
        seen = []
        metrics_df = to_dict(csfile)

        cmp_with_tie = lambda x, y: y * x >= 0
        cmp_mo_tie = lambda x, y: y * x > 0

        print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format("gold", "iteration", "pair", "left", "right",
                                                                          "disagreement_count",
                                                                          "disagreement_per", "dis_correct_lef",
                                                                          "dis_correct_right", "dis_correct_diff",
                                                                          "left_signficant",
                                                                          "right_signficant",
                                                                          "agreement_count",
                                                                          "agreement_per", "agree_wrong_lef",
                                                                          "agree_wrong_right",
                                                                          "agree_wrong_diff"
                                                                          ))


        cd_file = open("metric-simple-dis-condorcent.csv", mode="w")
        cd_file.write(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format("gold", "iteration", "pair", "left", "right",
                                                                      "disagreement_count",
                                                                      "disagreement_per", "dis_correct_lef",
                                                                      "dis_correct_right", "dis_correct_diff",
                                                                      "left_signficant",
                                                                      "right_signficant",
                                                                      "agreement_count",
                                                                      "agreement_per", "agree_wrong_lef",
                                                                      "agree_wrong_right",
                                                                      "agree_wrong_diff"
                                                                      ))


        for M1 in metrics:
            for M2 in metrics:
                if M1 == M2:
                    continue
                # if M1 + M2 in seen or M2+M1 in seen:
                #     continue
                # else:
                #     seen.append(M1 + M2)

                for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    for Mg in MG:
                        total = 0
                        ties = 0
                        disagreements, dis_correct_1, dis_correct_2 = 0, 0, 0

                        agreements, agree_wrong_1, agree_wrong_2 = 0, 0, 0

                        for r_1 in runs:
                                for r_2 in runs:
                                    if r_1 == r_2:
                                        continue

                                    t_m1_correct = 0
                                    t_m2_correct = 0

                                    for t in topics:
                                        if t == "all":
                                            continue

                                        if t not in metrics_df[r_1].keys() or t not in metrics_df[r_2].keys():
                                            continue

                                        total += 1
                                        M1_r1 = metrics_df[r_1][t][str(c)][M1]
                                        M2_r1 = metrics_df[r_1][t][str(c)][M2]

                                        M1_r2 = metrics_df[r_2][t][str(c)][M1]
                                        M2_r2 = metrics_df[r_2][t][str(c)][M2]


                                        M1_d = float(M1_r1) - float(M1_r2)
                                        M2_d = float(M2_r1) - float(M2_r2)


                                        MG_Ds = []

                                        for mg in Mg:
                                            Mg_r1 = metrics_df[r_1][t][str(c)][mg]
                                            Mg_r2 = metrics_df[r_2][t][str(c)][mg]
                                            mg_d = float(Mg_r1) - float(Mg_r2)

                                            MG_Ds.append(mg_d)

                                        if (M1_d * M2_d < 0):
                                            disagreements += 1

                                            # if comp_no_zero([M1_d] + MG_Ds):
                                            if self.same_no_tie(M1_d,MG_Ds):
                                                dis_correct_1 += 1

                                            # if comp_no_zero([M2_d]+ MG_Ds):
                                            if self.same_no_tie(M2_d,MG_Ds):
                                                dis_correct_2 += 1

                                            # if comp_with_zero([M1_d,M2_d]+MG_Ds):
                                            if self.same_with_tie(M1_d,MG_Ds) and self.same_with_tie(M2_d,MG_Ds):
                                                ties += 1

                                        else:
                                            agreements += 1

                                            if not self.same_with_tie(M1_d, MG_Ds):
                                                agree_wrong_1 += 1

                                            if not self.same_with_tie(M2_d,MG_Ds):
                                                agree_wrong_2 += 1


                        if disagreements > 0:
                                right_sig_level = 1
                                left_sig_level = 1

                                assert ((dis_correct_1 + dis_correct_2 ) <=disagreements)
                                dis = dis_correct_2 + dis_correct_1 + ties/2.0
                                if dis_correct_1 > dis_correct_2:
                                    left_sig_level = stats.binom_test(dis_correct_1,dis)

                                if dis_correct_2 > dis_correct_1:
                                    right_sig_level = stats.binom_test(dis_correct_2,dis)


                                if dis != disagreements:
                                    dummpy=0
                                # line = get_condorcnet_line("+".join(Mg),c,M1,M2,disagreements,total,dis_correct_1,dis_correct_2,agreements,agree_wrong_1,agree_wrong_2, left_sig_level,right_sig_level)
                                line = self.get_condorcnet_line("+".join(Mg), c, M1, M2, dis, total, dis_correct_1,
                                                           dis_correct_2, agreements, agree_wrong_1, agree_wrong_2,
                                                           left_sig_level, right_sig_level)
                                print (line)
                                cd_file.write(line+"\n")

        cd_file.close()


    def plot_condorcent_test(self,csv_file="metric-condorcent.csv",MG=["strec","p"],label_size=40):
        condorcent_df = pd.read_csv(csv_file)
        markers_cycle = itertools.cycle(['>', 'o', '^', 's', '*', 'd', '<', 'v', 'h', '8', 'D', '.', '1', "x"])
        lines_cycle = itertools.cycle(["-", "--", ":"])
        metrics_prety = {"ct": "CT", "act": "ACT", "nct": "nCT", "nEU": "nEU", "nsDCG": "nsDCG",
                         "alpha-nDCG": r"$\alpha$-nDCG", "nERR-IA": r"$nERR-IA$", "nNRBP": "nNRBP"}
        font = {'family': 'serif',
                'weight': 'normal',
                'size': label_size,
                }


        plt.rc('font', **font)

        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)
        plt.rc('axes', labelsize='x-large')
        for mg in MG:
            mg_df = condorcent_df[(condorcent_df["gold"] == mg) & (condorcent_df["disagreement_count"] >0)]

            for key, group in mg_df.groupby(by="left"):
                plt.rc('text', usetex=True)
                lfig, laxs = plt.subplots(figsize=(9, 4))
                # laxs.set_axis_bgcolor('w')
                # group.plot(kind="scatter", x="iteration", y="disagreement", ax=axs, s=marker_size, c=runs_colors[name], marker=markers.next(),
                #            label=name)
                for l, lgroup in group.groupby(by="right"):
                    # lgroup.plot(kind="scatter", x="iteration", y="dis_correct_diff", ax=laxs, s=100,label=l, marker=markers_cycle.next())
                    lgroup.plot(kind="line", x="iteration", y="dis_correct_diff", ax=laxs, style=lines_cycle.next(),
                                label=metrics_prety[l], ms=15)

                laxs.legend(loc="upper right", title="$M_2$")
                # plt.ylim(0, 14)
                laxs.set_ylabel("Intuitiveness($M_1$ )- Intuitiveness($M_2$)")

                plt.xlim(0, 11)

                plt.title("$M_1 = $" + metrics_prety[key])
                # plt.show()
                plt.savefig("./{}-{}-diff-metrics.pdf".format(key.lower(), mg), bbox_inches='tight', format='pdf', dpi=1000)
                print("Finished showing condorcent")


class CondorcentTestAnalysis2:
    #
    # def plot_condorcent_test_as_heat_map(csv_file="metric-condorcent.csv",MG=["strec","p"],cutoffs=[],label_size=40):
    #     condorcent_df = pd.read_csv(csv_file)
    #     markers_cycle = itertools.cycle(['>', 'o', '^', 's', '*', 'd', '<', 'v', 'h', '8', 'D', '.', '1', "x"])
    #     lines_cycle = itertools.cycle(["-", "--", ":"])
    #     metrics_prety = {"ct": "CT", "act": "ACT", "nct": "nCT", "nEU": "nEU", "nsDCG": "nsDCG",
    #                      "alpha-nDCG": r"$\alpha$-nDCG", "nERR-IA": r"$nERR-IA$", "nNRBP": "nNRBP"}
    #     font = {'family': 'serif',
    #             'weight': 'normal',
    #             'size': label_size,
    #             }
    #
    #
    #     plt.rc('font', **font)
    #
    #     plt.rc('xtick', labelsize=label_size)
    #     plt.rc('ytick', labelsize=label_size)
    #     plt.rc('axes', labelsize='x-large')
    #
    #     map_ticks = []
    #
    #     for m in metrics:
    #         for c in cutoffs:
    #             map_ticks.append(m+"@"+str(c))
    #
    #     for mg in MG:
    #
    #         for cutoff in cutoffs:
    #             mg_df = condorcent_df[condorcent_df["gold"] == mg]
    #             seen = []
    #
    #             for left in metrics:
    #                 for right in metrics:
    #                     if right == left or (left + right in seen) or right + left in seen:
    #                         continue
    #                     seen.append(right + left)
    #
    #                     data = mg_df[(mg_df["left"] == left) & (mg_df["right"] == right) & (mg_df["iteration"] == cutoff)]
    #                     sig_left = data["left_signficant"].values[0]
    #                     sig_right = data["right_signficant"].values[0]
    #
    #                     if sig_left < 0.05:
    #                         wins_counts[left] += 1
    #                         sig_relation[left + right] = "$>$"
    #                     elif sig_right < 0.05:
    #                         wins_counts[right] += 1
    #                         sig_relation[right + left] = "$>$"
    #                     else:
    #                         # pass
    #                         if not (wins_only):
    #                             wins_counts[right] += 1
    #                             wins_counts[left] += 1
    #
    #                         sig_relation[right + left] = ","
    #                         symbol = ","
    #
    #             # for key, value in wins_counts.iteritems():
    #             #     wins_counts[key] = value/float(len(metrics))
    #
    #             # for key, value in wins_counts.iteritems():
    #             #       # wins_counts[key] = value/float(len(metrics))
    #             #     print (",".join([mg,str(cutoff),key,str(value)]))
    #
    #             sorted_x = sorted(wins_counts.items(), key=operator.itemgetter(1), reverse=True)
    #             keys = [item[0] for item in sorted_x]
    #
    #             items = []
    #             added_metrics = []
    #             for i in range(len(sorted_x) - 1):
    #                 if not metrics_prety[sorted_x[i][0]] in items:
    #                     items.append(metrics_prety[sorted_x[i][0]])
    #
    #                 symbol = ","
    #                 if sig_relation.has_key(sorted_x[i][0] + sorted_x[i + 1][0]):
    #                     symbol = sig_relation[sorted_x[i][0] + sorted_x[i + 1][0]]
    #                 else:
    #                     symbol = sig_relation[sorted_x[i + 1][0] + sorted_x[i][0]]
    #
    #                 items.append(symbol)
    #
    #                 if not metrics_prety[sorted_x[i + 1][0]] in items:
    #                     items.append(metrics_prety[sorted_x[i + 1][0]])
    #             mg_metrics_rankings[str(mg) + "@" + str(cutoff)] = "".join(items)
    #             mg_metrics_wins[str(mg) + "@" + str(cutoff)] = wins_counts
    #             print (str(mg) + "&" + str(cutoff) + "&" + "".join(items) + "\\\\")


    def plot_condorcent_test_wins(csv_file="metrics-it-wins.csv", MG=[], label_size=40):
        condorcent_df = pd.read_csv(csv_file)
        markers_cycle = itertools.cycle(['>', 'o', '^', 's', '*', 'd', '<', 'v', 'h', '8', 'D', '.', '1', "x"])
        lines_cycle = itertools.cycle(["-", "--", ":"])
        metrics_prety = {"ct": "CT", "act": "ACT", "nct": "nCT", "nEU": "nEU", "nsDCG": "nsDCG",
                         "alpha-nDCG": "$\\alpha$-nDCG", "nERR-IA": "nERR-IA", "nNRBP": "nNRBP"}

        font = {'family': 'serif',
                'weight': 'normal',
                'size': label_size,
                }

        plt.rc('font', **font)

        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)
        plt.rc('axes', labelsize='x-large')
        for mg in MG:
            markers_cycle = itertools.cycle(['>', 'o', '^', 's', '*', 'd', '<', 'v', 'h', '8', 'D', '.', '1', "x"])
            lines_cycle = itertools.cycle(["-", "--", ":"])
            mg_df = condorcent_df[(condorcent_df["dimensions"] == mg)]

            mg_fig, mg_axs = plt.subplots(figsize=(9, 4))
            for key, group in mg_df.groupby(by="metric"):
                plt.rc('text', usetex=True)

                group.plot(kind="line", x="iteration", y="wins", ax=mg_axs,
                           style=lines_cycle.next() + markers_cycle.next(),
                           label=metrics_prety[key], ms=15)

            mg_axs.legend(loc="upper right", title="$Metrics$", prop={'size': 18})
            mg_axs.set_ylabel("Wins")

            plt.xlim(0, 14)
            plt.ylim(0, 6)
            # plt.show()
            plt.savefig("./{}-it-wins-metrics.pdf".format(mg.lower()), bbox_inches='tight', format='pdf', dpi=1000)
            print("Finished showing condorcent wins")

    def generate_condorcent_table(self,csv_file="metric-condorcent.csv", MG=["strec", "p", "rtime"], metrics=None,
                                  cutoffs=[], score_only=True):
        condorcent_df = pd.read_csv(csv_file)

        metrics_prety = {"ct": "CT", "act": "ACT", "nct": "nCT", "nEU": "nEU", "nsDCG": "nsDCG",
                         "alpha-nDCG": r"$\alpha$-nDCG", "nERR-IA": r"$nERR-IA$", "nNRBP": "nNRBP"}

        prety_header = []

        for m in metrics:
            prety_header.append(metrics_prety[m])

        print ("&".join(["iteration"] + prety_header[1:]) + "\\\\")

        for mg in MG:
            print("\\hline &{}\\\\\\hline".format(mg))

            for cutoff in cutoffs:
                mg_df = condorcent_df[condorcent_df["gold"] == mg]
                seen = []
                for left in metrics[:-1]:
                    line = [str(cutoff), metrics_prety[left]]
                    for right in metrics[1:]:
                        if left == right:
                            line.append("-")
                            continue

                        if left + right in seen or right + left in seen:
                            line.append("-")
                        else:
                            seen.append(left + right)
                            data = mg_df[
                                (mg_df["left"] == left) & (mg_df["right"] == right) & (mg_df["iteration"] == cutoff)]

                            # assert (len(data) > 1)

                            disagreements = data["disagreement_per"].values[0]
                            correct_1 = data["dis_correct_lef"].values[0]
                            correct_2 = data["dis_correct_right"].values[0]

                            sig_left = data["left_signficant"].values[0]
                            sig_right = data["right_signficant"].values[0]

                            sig_left_symbol = ""
                            if sig_left < 0.05:
                                sig_left_symbol = "$^+$"

                            sig_right_symbol = ""
                            if sig_right < 0.05:
                                sig_right_symbol = "$^+$"
                            line.append(
                                "{0:.4f}{1}/{2:.4f}{3} ({4:.2f}\%)".format((correct_1), sig_left_symbol, (correct_2),
                                                                           sig_right_symbol, (100 * disagreements)))

                    print (" &".join(line) + "\\\\")

                print("\\hline")

    def generate_condorecent_test_summary_table(csv_file="metric-condorcent.csv", MG=["strec", "p", "rtime"],
                                                metrics=None, cutoffs=[]):
        condorcent_df = pd.read_csv(csv_file)

        metrics_prety = {"ct": "CT", "act": "ACT", "nct": "nCT", "nEU": "nEU", "nsDCG": "nsDCG",
                         "alpha-nDCG": r"$\alpha$-nDCG", "nERR-IA": r"$nERR-IA$", "nNRBP": "nNRBP"}

        prety_header = []

        for m in metrics:
            prety_header.append(metrics_prety[m])

        mg_metrics_rankings = {}
        print ("dimensions,iteration,metric,wins")
        for mg in MG:

            for cutoff in cutoffs:
                mg_df = condorcent_df[condorcent_df["gold"] == mg]
                wins_counts = {}
                seen = []
                for m in metrics:
                    wins_counts[m] = 0

                for left in metrics:
                    for right in metrics:
                        if right == left or (left + right in seen) or right + left in seen:
                            continue
                        seen.append(right + left)

                        data = mg_df[
                            (mg_df["left"] == left) & (mg_df["right"] == right) & (mg_df["iteration"] == cutoff)]
                        sig_left = data["left_signficant"].values[0]
                        sig_right = data["right_signficant"].values[0]

                        if sig_left < 0.05:
                            wins_counts[left] += 1
                        elif sig_right < 0.05:
                            wins_counts[right] += 1
                        else:
                            # pass
                            wins_counts[right] += 1
                            wins_counts[left] += 1

                # for key, value in wins_counts.iteritems():
                #     wins_counts[key] = value/float(len(metrics))

                # for key, value in wins_counts.iteritems():
                #       # wins_counts[key] = value/float(len(metrics))
                #     print (",".join([mg,str(cutoff),key,str(value)]))

                sorted_x = sorted(wins_counts.items(), key=operator.itemgetter(1), reverse=True)
                keys = [item[0] for item in sorted_x]
                # print (str(mg) + "@" +str(cutoff)+ ">>".join(keys))

    def get_win_count(mg, cutoff, condorcent_df, wins_only=False):
        mg_df = condorcent_df[condorcent_df["gold"] == mg]
        wins_counts = {}
        seen = []

        for m in metrics:
            wins_counts[m] = 0

        for left in metrics:
            for right in metrics:
                if right == left or (left + right in seen) or right + left in seen:
                    continue
                seen.append(right + left)

                data = mg_df[(mg_df["left"] == left) & (mg_df["right"] == right) & (mg_df["iteration"] == cutoff)]
                sig_left = data["left_signficant"].values[0]
                sig_right = data["right_signficant"].values[0]

                if sig_left < 0.05:
                    wins_counts[left] += 1
                elif sig_right < 0.05:
                    wins_counts[right] += 1
                else:
                    # pass
                    if not wins_only:
                        wins_counts[right] += 1
                        wins_counts[left] += 1

        return wins_counts

    def generate_condorecent_test_wins_table(self,csv_file="metric-condorcent.csv", MG=["strec", "p", "rtime"], metrics=None,
                                             cutoffs=[]):
        condorcent_df = pd.read_csv(csv_file)

        metrics_prety = {"ct": "CT", "act": "\\act", "nct": "\\nct", "nEU": "\\neu", "nsDCG": "\\nsdcg",
                         "alpha-nDCG": "\\alphandcg", "nERR-IA": "\\nerria", "nNRBP": "nNRBP"}

        prety_header = {}

        for m in metrics:
            prety_header[m] = metrics_prety[m]

        mg_metrics_wins = {}
        headers = []

        for mg in MG:
            for cutoff in cutoffs:
                headers.append(str(cutoff))
                wins_counts = self.get_win_count(mg, cutoff, condorcent_df, wins_only=True)
                sorted_x = sorted(wins_counts.items(), key=operator.itemgetter(1), reverse=True)
                mg_metrics_wins[mg + "@" + str(cutoff)] = wins_counts
        print ("\\begin{{tabular}}{{c{}}}".format("c".join(["" for i in range(len(cutoffs) * len(MG))])))
        print ("&".join([""] + ["\multicolumn{{{}}}{{c}}{{{}}}".format(len(cutoffs), mg) for mg in MG]) + "\\\\")
        print ("&".join(["metric/iteration"] + headers) + "\\\\")
        for m in metrics:
            formated_cells = [prety_header[m] + "\n\t"]

            mg_count = 0
            for mg in MG:
                mg_count += 1
                row_cells = []

                for cutoff in cutoffs:
                    wins_counts = mg_metrics_wins[mg + "@" + str(cutoff)]
                    row_cells.append(" " + str(wins_counts[m]) + " ")

                formated_cells.append("\n\t" + (mg_count * (" ")) + "&".join(row_cells))

            print ("&".join(formated_cells) + "\\\\")
        print("\end{tabular}")

    def generate_concordence_test_rankings(csv_file="metric-condorcent.csv", MG=["strec", "p", "rtime"], metrics=None,
                                           cutoffs=[], wins_only=False):
        condorcent_df = pd.read_csv(csv_file)

        # metrics_prety = {"ct": "CT", "act": "ACT", "nct": "nCT", "nEU": "nEU", "nsDCG": "nsDCG",
        #                  "alpha-nDCG": r"$\alpha$-nDCG", "nERR-IA": r"$nERR-IA$", "nNRBP": "nNRBP"}

        metrics_prety = {"ct": "\\ct", "act": "\\act", "nct": "\\nct", "nEU": "\\neu", "nsDCG": "\\nsdcg",
                         "alpha-nDCG": "\\alphandcg", "nERR-IA": "\\nerria", "nNRBP": "\\nnrbp"}

        prety_header = []

        for m in metrics:
            prety_header.append(metrics_prety[m])

        mg_metrics_rankings = {}
        mg_metrics_wins = {}
        print ("dimensions,iteration,metric,rank")
        for mg in MG:

            for cutoff in cutoffs:
                sig_relation = {}
                mg_df = condorcent_df[condorcent_df["gold"] == mg]
                wins_counts = {}
                seen = []
                for m in metrics:
                    wins_counts[m] = 0

                for left in metrics:
                    for right in metrics:
                        if right == left or (left + right in seen) or right + left in seen:
                            continue
                        seen.append(right + left)

                        data = mg_df[
                            (mg_df["left"] == left) & (mg_df["right"] == right) & (mg_df["iteration"] == cutoff)]
                        sig_left = data["left_signficant"].values[0]
                        sig_right = data["right_signficant"].values[0]

                        if sig_left < 0.05:
                            wins_counts[left] += 1
                            sig_relation[left + right] = "$>$"
                        elif sig_right < 0.05:
                            wins_counts[right] += 1
                            sig_relation[right + left] = "$>$"
                        else:
                            # pass
                            if not (wins_only):
                                wins_counts[right] += 1
                                wins_counts[left] += 1

                            sig_relation[right + left] = ","
                            symbol = ","

                # for key, value in wins_counts.iteritems():
                #     wins_counts[key] = value/float(len(metrics))

                # for key, value in wins_counts.iteritems():
                #       # wins_counts[key] = value/float(len(metrics))
                #     print (",".join([mg,str(cutoff),key,str(value)]))

                sorted_x = sorted(wins_counts.items(), key=operator.itemgetter(1), reverse=True)
                keys = [item[0] for item in sorted_x]

                items = []
                added_metrics = []
                for i in range(len(sorted_x) - 1):
                    if not metrics_prety[sorted_x[i][0]] in items:
                        items.append(metrics_prety[sorted_x[i][0]])

                    symbol = ","
                    if sig_relation.has_key(sorted_x[i][0] + sorted_x[i + 1][0]):
                        symbol = sig_relation[sorted_x[i][0] + sorted_x[i + 1][0]]
                    else:
                        symbol = sig_relation[sorted_x[i + 1][0] + sorted_x[i][0]]

                    items.append(symbol)

                    if not metrics_prety[sorted_x[i + 1][0]] in items:
                        items.append(metrics_prety[sorted_x[i + 1][0]])
                mg_metrics_rankings[str(mg) + "@" + str(cutoff)] = "".join(items)
                mg_metrics_wins[str(mg) + "@" + str(cutoff)] = wins_counts
                print (str(mg) + "&" + str(cutoff) + "&" + "".join(items) + "\\\\")
                # print (str(mg) + "@" +str(cutoff)+ ">>".join(keys))
                # print ("dimensions,iteration,metric,rank")
                # for mg, ranking in mg_metrics_rankings.iteritems():
                #     buckets = ranking.split(">")
                #     for i in range(len(buckets)):
                #         b_metrics = buckets[i].split(",")
                #         gold,cutoff = mg.split("@")
                #         for m in b_metrics:
                #             print (",".join([gold,cutoff,m.replace("$","").replace("\\",""),str(1./(i+1))]))

                # print ("\n\n\n\n")
                # print ("dimensions,iteration,metric,wins")
                # for mg, wins_map in mg_metrics_wins.iteritems():
                #
                #     for key,count in wins_map.iteritems():
                #         gold, cutoff = mg.split("@")
                #         print (",".join([gold, cutoff, key.replace("$", "").replace("\\", ""), str(count)]))


if __name__ == "__main__":
    MG = ["strec", "p", "rtime"]
    dd_metrics = ["act", "nct","nEU","nsDCG"]
    wd_metrics = "alpha-nDCG,nERR-IA".split(",")
    rbu_metrics = []

    Es = [0.001, .050, 0.100, 0.500]
    Ps = [0.800, 0.900, 0.990]

    rbus_metrics = ["rbu_{0:.3f}_{1:.3f}".format(p, e) for p, e in itertools.product(Ps, Es)]
    metrics = dd_metrics + wd_metrics + rbus_metrics

    M_GS = [
        ["p"],
        ["strec"],


        ["rtime"],
        ["ntime"],
        ["ttime"],

        ["strec", "p"],
        ["strec", "rtime"],
        ["strec", "ntime"],
        ["strec", "ttime"],
        ["p", "rtime"],
        ["p", "ntime"],
        ["p", "ttime"],

        ["strec", "p", "rtime"],
        ["strec", "p", "ntime"],
        ["strec", "p", "ttime"],
    ]

    M_GS_label = []
    for m in M_GS:
        M_GS_label.append("+".join(m))

    condorcnetTest = CondorcentTestReport()
    analaysis =CondorcentTestAnalysis2()

    def merge_year_evals(year,topic_prefix):
        data_dir = os.path.join("..", "data", "evals", "trec-dd",str(year))
        wd_result_file = os.path.join(data_dir,"trec-wd-{}-evals.csv".format(year))
        dd_result_file = os.path.join(data_dir, "trec-dd-{}-evals.csv".format(year))

        wd_df = pd.read_csv(wd_result_file)
        dd_df = pd.read_csv(dd_result_file)
        dd_df["topic"] = dd_df["topic"].str.replace(topic_prefix,"")
        metrics_df = wd_df.merge(dd_df,on=["run","iteration","topic"])


        rbu_result_file = os.path.join(data_dir, "rbu-trec-all-{}-evals.csv".format(year))
        all_result_file = os.path.join(data_dir, "trec-all-{}-evals.csv".format(year))
        rbu_df = pd.read_csv(rbu_result_file)
        metrics_df = rbu_df.merge(metrics_df, on=["run", "iteration", "topic"])
        # metrics_df.drop(metrics_df.columns[metrics_df.columns.str.contains('unnamed', case=False)], axis=1)
        metrics_df.to_csv(all_result_file)


    def run_year_condorcment_test(year):
        data_dir = os.path.join("..", "data", "evals", "trec-dd", str(year))
        eval_file = os.path.join(data_dir,"trec-all-{}-evals.csv".format(year))
        condorcment_file = os.path.join(data_dir,"metric-condorcent-{}.csv".format(year))
        metrics_df = pd.read_csv(eval_file)
        topics = list(set(metrics_df["topic"].values))
        runs = set(metrics_df["run"].values)
        condorcnetTest.condorcent_test_nmg(condorcment_file=condorcment_file,csfile=eval_file, MG=M_GS, metrics=metrics, topics=topics, runs=runs)


    def run_year_generate_condorcent_table(year):
        data_dir = os.path.join("..", "data", "evals", "trec-dd", str(year))
        condorcment_file = os.path.join(data_dir, "metric-condorcent-{}.csv".format(year))
        # analaysis.generate_condorcent_table(csv_file=condorcment_file,MG=M_GS_label, metrics=metrics, cutoffs=[1,10])
        analaysis.generate_condorecent_test_wins_table(csv_file=condorcment_file,MG=M_GS_label, metrics=metrics, cutoffs=[1,2,5,8,10])


    merge_year_evals(year=2015,topic_prefix="DD15-")
    run_year_condorcment_test(year=2015)
    # run_year_generate_condorcent_table(year=2015)


    # merge_year_evals(year=2016,topic_prefix="DD16-")
    # run_year_condorcment_test(year=2016)
    # run_year_generate_condorcent_table(year=2016)

    # merge_year_evals(year=2017, topic_prefix="dd17-")
    # run_year_condorcment_test(year=2017)
    # run_year_generate_condorcent_table(year=2017)


    # merge_year_evals(year="2017-extra",topic_prefix="dd17-")
    # run_year_condorcment_test(year="2017-extra")
    # run_year_generate_condorcent_table(year="2017-extra")
    #
    # merge_year_evals(year="rmit-2017", topic_prefix="dd17-")
    # run_year_condorcment_test(year="rmit-2017")