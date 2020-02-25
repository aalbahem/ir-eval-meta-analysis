import csv
import itertools
import os
import pandas as pd
from intut_test.IntutivnessTest import IntutivnessTest, IntutivnessTestResult


def to_dict(csvfile, by=["iteration","run","topic"]):
    with open(csvfile, 'rb') as csvfile:
        all_ = list(csv.DictReader(csvfile))

        d = {}

        for row in all_:
            first_key = row[by[0]]
            second_key = row[by[1]]
            third_key = row[by[2]]

            if first_key not in d.keys():
                d[first_key] = {}

            if second_key not in d[first_key].keys():
                d[first_key][second_key]={}

            if third_key not in d[first_key][second_key].keys():
                d[first_key][second_key][third_key] = {}

            for m in row.keys() :
                if m not in by:
                    d[first_key][second_key][third_key][m]=row[m]

        return d


class TRECDD_IntutivnessTest_Analysis(object):

    def analysis(self, analysis_file_path, csfile, MG=[], metrics=[], topics=[], runs=[]):
        trec_dd_data = to_dict(csfile)

        columns = ["iteration","simple-metrics","pair","left","right","total","disagree_count","left_disagree_wins","right_disagree_wins","left_intutiveness_score","right_intutiveness_score"]
        print(",".join(columns))

        file_out = open(analysis_file_path,mode="wb")
        file_out.write(",".join(columns)+"\n")
        for left in metrics:
            for right in metrics:
                if left == right:
                    continue
                for c in range(1, 11):
                    for ms in MG:
                        intutivness_test = IntutivnessTest(left=left, right=right, MS=ms, runs=runs, topics=topics,
                                                       data=trec_dd_data[str(c)])
                    result = intutivness_test.compute()
                    line = str(c) + "," + result.to_csv()
                    print (line)
                    file_out.write(line+"\n")


        file_out.close()

        return

    def generate_intutivness_test_table(self,csv_file=None, MG=["strec", "p", "ttime"], metrics=None,
                                  cutoffs=[], score_only=True):
        analysis_df = pd.read_csv(csv_file)

        metrics_prety = {"ct": "\\ct", "act": "\\act", "nct": "\\nct", "nEU": "\\neu", "nsDCG": "\\nsdcg",
                         "alpha-nDCG": "\\alphandcg", "nERR-IA": "\\nerria", "nNRBP": "\\nnrbp",

                         "rbu_0.990_0.050":"\\rbu{0.990}{0.050}"}

        Es = [0.001, .050, 0.100, 0.500]
        Ps = [0.800, 0.900, 0.990]

        for p, e in itertools.product(Ps, Es):
            metrics_prety["rbu_{0:.3f}_{1:.3f}".format(p, e)] = "\\rbu{{{0:.3f}}}{{{1:.3f}}}".format(p,e)

        prety_header = []

        for m in metrics:
            prety_header.append(metrics_prety[m])

        print ("&".join(["iteration"] + prety_header[1:]) + "\\\\")

        for mg in MG:
            print("\\hline &{}\\\\\\hline".format(mg))

            for cutoff in cutoffs:
                mg_df = analysis_df[(analysis_df["simple-metrics"] == mg) & (analysis_df["iteration"] == cutoff)]
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
                                (mg_df["left"] == left) & (mg_df["right"] == right)]

                            # assert (len(data) > 1)

                            total = data["total"].values[0]
                            disagreements = data["disagree_count"].values[0]
                            ratio = disagreements/float(total)
                            correct_1 = data["left_disagree_wins"].values[0]
                            correct_2 = data["right_disagree_wins"].values[0]
                            left_int = data["left_intutiveness_score"].values[0]
                            right_int = data["right_intutiveness_score"].values[0]

                            intutivnessTestResult = IntutivnessTestResult(mg,left,right,total,disagreements,0,correct_1,correct_2,left_int,right_int)
                            sig_tests = intutivnessTestResult.compute_sig_level()

                            sig_left = sig_tests[0]
                            sig_right = sig_tests[1]

                            sig_left_symbol = ""
                            if sig_left < 0.05:
                                sig_left_symbol = "$^+$"

                            sig_right_symbol = ""
                            if sig_right < 0.05:
                                sig_right_symbol = "$^+$"
                            line.append(
                                "{0:.4f}{1}/{2:.4f}{3} ({4:.2f}\%)".format((left_int), sig_left_symbol, (right_int),
                                                                           sig_right_symbol, (100 * ratio)))

                    print (" &".join(line) + "\\\\")

                print("\\hline")


#
if __name__ == "__main__":
    MG = ["strec", "p", "rtime"]
    dd_metrics = ["act", "nct", "nEU", "nsDCG"]
    wd_metrics = "alpha-nDCG,nERR-IA".split(",")
    rbu_metrics = []

    Es = [0.050]
    Ps = [0.990]

    rbus_metrics = ["rbu_{0:.3f}_{1:.3f}".format(p, e) for p, e in itertools.product(Ps, Es)]
    metrics = dd_metrics + wd_metrics + rbus_metrics

    M_GS = [
        # ["p"],
        # ["strec"],
        #
        # ["rtime"],
        # ["ntime"],
        # ["ttime"],
        #
        # ["strec", "p"],
        # ["strec", "rtime"],
        # ["strec", "ntime"],
        # ["strec", "ttime"],
        # ["p", "rtime"],
        # ["p", "ntime"],
        # ["p", "ttime"],

        # ["strec", "p", "rtime"],
        # ["strec", "p", "ntime"],
        ["strec", "p", "ttime"],
    ]

    M_GS_label = []
    for m in M_GS:
        M_GS_label.append("+".join(m))


    def run_year_analysis(year):
        data_dir = os.path.abspath(os.path.join( "data", "evals","trec-dd",  str(year)))
        eval_file = os.path.join(data_dir, "trec-all-{}-evals.csv".format(year))
        analysis_file = os.path.join(data_dir, "metric-intuitiveness-test-{}.csv".format(year))
        metrics_df = pd.read_csv(eval_file)
        topics = list(set(metrics_df["topic"].values))
        runs = set(metrics_df["run"].values)

        analysis = TRECDD_IntutivnessTest_Analysis()

        analysis.analysis(analysis_file_path=analysis_file, csfile=eval_file, MG=M_GS, metrics=metrics, topics=topics, runs=runs)


    def merge_all_years_evals(years):
        all_data = None
        total = 0
        for year in years:
            data_dir = os.path.abspath(os.path.join( "data", "evals","trec-dd",  str(year)))
            eval_file = os.path.join(data_dir, "trec-all-{}-evals.csv".format(year))
            metrics_df = pd.read_csv(eval_file)
            total+= metrics_df.size
            if all_data is None:
                all_data = pd.DataFrame(metrics_df)
            else:
                all_data = all_data.append(metrics_df)

        assert total == all_data.size
        data_dir = os.path.abspath(os.path.join("data", "evals", "trec-dd", "all"))
        eval_file = os.path.join(data_dir, "trec-all-{}-evals.csv".format("all"))
        all_data.to_csv(eval_file)


    def run_year_generate_intutivness_table(year):
        analysis = TRECDD_IntutivnessTest_Analysis()

        data_dir = os.path.join("data", "evals", "trec-dd", str(year))
        analysis_file = os.path.join(data_dir, "metric-intuitiveness-test-{}.csv".format(year))
        analysis.generate_intutivness_test_table(csv_file=analysis_file,MG=M_GS_label, metrics=metrics, cutoffs=[1,10])


    # merge_all_years_evals(["2015", "2016", "2017"])
    # run_year_analysis("2015")
    # run_year_analysis("2016")
    # run_year_analysis("2017")
    # run_year_analysis("all")
    run_year_generate_intutivness_table("2016")