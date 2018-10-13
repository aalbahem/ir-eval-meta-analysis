import csv
import itertools
from itertools import chain
import scipy.stats as stats

import matplotlib
import numpy
import pandas as pd
from os import listdir
from subprocess import call
import subprocess

from pandas.tools.plotting import boxplot_frame_groupby
from scipy.stats import ttest_ind

from guiqwt.pyplot import savefig
import matplotlib.pyplot as plt
from scipy.stats.stats import ttest_rel

import trec_analysis

def generate_cubetest_result(path,cube_test_command,qrel,cutoff):
    """
    Generate a run file for each file found in path
    :param path:
    :param qrel:
    :return:
    """

    for run in listdir(path):
        print run
        call(["perl.exe",cube_test_command, qrel,path+"\\"+run,cutoff],stdout=open(run.replace(".txt",""),"w"))


def generate_topic_summary(runs,title="performance",xlabel=None,ylabel=None,run_dfs=None,relative=False,baseline=None):
    """
    Loads runs from <b>runs</b> and then generate summary figure using xlabel, ylabel and title
    :param runs:
           runs names.
    :param title:
           the title of the figure, used to save figure as well.
    :param xlabel:
            the x axis label
    :param ylabel:
            the y axis label
    :return:
    """

    if run_dfs==None:
        run_dfs = {};

        for run in runs:
            run_dfs[run] = pd.read_csv(run)
            print len(run_dfs[run])

    #data frame for drawing.
    _df = pd.DataFrame();
    topics = run_dfs[baseline][xlabel].values
    topics = sorted(topics)
    _df[xlabel] = topics

    for run in runs:
        _df[run]=0



    for x  in range(len(_df[xlabel])):

        for run in runs:
          print (_df)
          run_df = run_dfs[run]
          t = _df.iloc[x][xlabel]

          run_f = run_df[run_df[xlabel] ==t ]
          values = run_f["avg_ct"].values
          y = values[0]

          # _df.set_value(x, run, y)
          _df.ix[x,run]=y

    # _df.plot(kind="bar",x=xlabel,label = ylabel,title=title)

    #
    # for run in runs:
    #     if (relative):
    #         _df[run]=run_dfs[run][ylabel]-run_dfs[run][ylabel];
    #     else:
    #        _df[run] = run_dfs[run][ylabel]
    #
    # if (relative):
    #     del _df[baseline]
    #
    #
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.07)
    fig=_df.plot.bar(x=xlabel,label = ylabel,title=title)
    fig.set_axis_bgcolor('w')
    print("find")

def generate_ct_by_topic(path,cube_test_command,qrel,cutoff,x="topic",by="run_id",metric="ct",relative=False,baseline=None):
    """
    Loads runs from <b>runs</b> and then generate summary figure using xlabel, ylabel and title
    :param runs:
           runs names.
    :param title:
           the title of the figure, used to save figure as well.
    :param xlabel:
            the x axis label
    :param ylabel:
            the y axis label
    :return:
    """

    run_dfs = {};
    runs = []


    for run in listdir(path):
            params  = ["perl.exe", cube_test_command, qrel,
                 runs_path + "\\" + run,
                 str(cutoff)]
            lines = (subprocess.check_output(params).strip().split("\n"))
            lines = [line.replace(runs_path + "\\", "").replace(".txt", "")+"\n" for line in lines]

            run_name = run.replace(".txt","");
            runs.append(run_name)
            run_file = open(run_name+".csv","w")

            for line in lines:
                run_file.write(line);
            run_file.close()

            run_df=pd.read_csv(run_name+".csv")
            run_dfs[run_name]=run_df

    generate_topic_summary(runs,run_dfs,x=x,by=by,y=str(metric)+"@"+str(cutoff),title="Title",relative=relative,baseline=baseline)






    #
    # for run in runs:
    #     if (relative):
    #         _df[run]=run_dfs[run][ylabel]-run_dfs[run][ylabel];
    #     else:
    #        _df[run] = run_dfs[run][ylabel]
    #
    # if (relative):
    #     del _df[baseline]
    #
    #
    # fig = _df.plot(kind="bar",x=xlabel,title=title)
    print("find")


def generate_relevance_statistics(runs_path,out):
    """
    Iterates through runs in runs_path and generate a csv value with the following details:
      run_id, topic,iteration,r,nr,n_subtopics,avg_subtopic_doc,avg_rating
    :param runs_path:
    :return:
    """

    o = open(out,"w");
    o.write("run, topic,iteration,relevant,non-relevant,n_subtopics,avg_sutopics,iteration_total_rating\n");

    for run in listdir(runs_path):
        run_file =  open(runs_path+"\\"+run);
        iteration_r, iteration_nr, iteration_n_subtopics, iteration_total_rate = 0,0,0,0

        prev_topic,prev_iteration=None,0
        for line in run_file.readlines():

            parts = line.strip().split("\t");
            #DD16-1 0         docid  4.418071 1           DD16-1.1:3|DD16-1.1:2|DD16-1.3:2|DD16-1.2:3
            topic,  iteration,doc_id,score,   on_topic = parts[0:5];

            iteration = int(iteration)

            if (iteration != prev_iteration and topic != prev_topic):
                # run_id, topic,iteration,r,nr,n_subtopics,avg_subtopic_doc,avg_rating
                o.write(
                    "{0},{1},{2},{3},{4},{5},{6},{7}\n".format(run.replace(".txt",""), prev_topic, prev_iteration+1, iteration_r, iteration_nr,
                                                               iteration_n_subtopics,
                                                               iteration_n_subtopics / max(iteration_r, 1),
                                                               iteration_total_rate))
                prev_iteration = iteration
                iteration_r, iteration_nr, iteration_n_subtopics, iteration_total_rate = 0, 0, 0, 0

            if (topic != prev_topic):
                prev_topic = topic
                iteration_r, iteration_nr, iteration_n_subtopics, iteration_total_rate = 0, 0, 0, 0



            if (iteration != prev_iteration and topic == prev_topic):
                # run_id, topic,iteration,r,nr,n_subtopics,avg_subtopic_doc,avg_rating
                o.write(
                    "{0},{1},{2},{3},{4},{5},{6},{7}\n".format(run.replace(".txt",""), topic, prev_iteration+1, iteration_r, iteration_nr,
                                                               iteration_n_subtopics,
                                                               iteration_n_subtopics / max(iteration_r, 1),
                                                               iteration_total_rate))
                prev_iteration = iteration
                iteration_r, iteration_nr, iteration_n_subtopics, iteration_total_rate = 0, 0, 0, 0




            details =None;

            if (on_topic=="1"):
                details=parts[-1]
            else:
                iteration_nr = iteration_nr + 1

            if (details !=None):
                iteration_r = iteration_r + 1;
                subtopics = details.split("|")
                iteration_n_subtopics = iteration_n_subtopics+len(subtopics)


                for subtopic in subtopics:
                    subtopic_id,rate = subtopic.split(":");
                    iteration_total_rate = iteration_total_rate+int(rate)





def summarize_cubetest_result(path,cube_test_command,qrel,cutoffs):
    """
    Generate a run file for each file found in path
    :param path:
    :param qrel:
    :return:
    """

    for run in listdir(path):
        for c in cutoffs:
            line=(str(c) + "," + subprocess.check_output(
                ["perl.exe", cube_test_command, qrel, runs_path + "\\" + run,
                 str(c)]).strip().split("\n")[-1])
            line = line.replace(runs_path+"\\","").replace(".txt","").replace(",all","")
            print line


def ttest(path="evalution.csv",runs=["lm"],baseline="lm",metric="ct",cutoffs=[]):
    evals  = pd.read_csv(path)

    for cutoff in cutoffs:
        evals_cutoff = evals[evals['iteration'] == cutoff]
        baseline_evals = evals_cutoff[evals_cutoff['run'] == baseline]
        baseline_evals = baseline_evals[baseline_evals['topic'] != "all"]
        # baseline_scores = baseline_scores[baseline_scores['iteration'] == cutoff]
        # baseline_metric_scores = baseline_scores[metric]


        # data frame for drawing.
        _df = pd.DataFrame();
        topics = baseline_evals["topic"].values
        topics = sorted(topics)
        _df["topic"] = topics
        xlabel="topic"

        for run in runs:
            _df[run] = 0

        for x in range(len(_df["topic"])):

            for run in runs:
                # print (_df)
                run_df = evals_cutoff[evals_cutoff['run'] == run]
                t = _df.iloc[x][xlabel]

                run_f = run_df[run_df[xlabel] == t]
                values = run_f[metric].values
                y = values[0]

                # _df.set_value(x, run, y)
                _df.ix[x, run] = y

        for run in runs:

            run_metric_scores = _df[run].values
            baseline_metric_scores = _df[baseline].values

            win,loses,tie = 0,0,0
            for b,r in zip(baseline_metric_scores,run_metric_scores):
                if b>r:
                    loses=loses+1
                elif b==r:
                    tie= tie +1
                else:
                    win = win+1

            s,p_value=ttest_rel(baseline_metric_scores, run_metric_scores)
            # print (s,p_value)
            print "{},{},{},{},{},{}\n".format(run,baseline,cutoff,metric,",".join([str(win),str(loses),str(tie)]),str(p_value<0.05))


def delta_ttest(path="evalution.csv",runs=["lm"],baseline="lm",metric="ct",cutoffs=[]):
    evals  = pd.read_csv(path)
    cutoffs_evals = dict()

    for cutoff in cutoffs:


        evals_cutoff = evals[evals['iteration'] == cutoff]
        baseline_evals = evals_cutoff[evals_cutoff['run'] == baseline]
        baseline_evals = baseline_evals[baseline_evals['topic'] != "all"]
        # baseline_scores = baseline_scores[baseline_scores['iteration'] == cutoff]
        # baseline_metric_scores = baseline_scores[metric]


        # data frame for drawing.
        _df = pd.DataFrame();
        topics = baseline_evals["topic"].values
        topics = sorted(topics)
        _df["topic"] = topics
        xlabel="topic"

        for run in runs:
            _df[run] = 0

        for x in range(len(_df["topic"])):

            for run in runs:
                # print (_df)

                if cutoff > 1:
                    run_df = evals_cutoff[evals_cutoff['run'] == run]
                run_df = evals_cutoff[evals_cutoff['run'] == run]
                t = _df.iloc[x][xlabel]

                run_f = run_df[run_df[xlabel] == t]
                values = run_f[metric].values
                y = values[0]

                # _df.set_value(x, run, y)
                _df.ix[x, run] = y

        cutoffs_evals[str(cutoff)] = _df


        for run in runs:
            run_metric_scores = _df[run].values
            baseline_metric_scores = _df[baseline].values


            run_metric_scores_previous = numpy.zeros(len(run_metric_scores))
            baseline_metric_scores_previous = numpy.zeros(len(baseline_metric_scores))

            if (cutoff > 1):
                run_metric_scores_previous = cutoffs_evals[str(cutoff - 1)][run].values
                baseline_metric_scores_previous = cutoffs_evals[str(cutoff - 1)][baseline].values

            run_metric_scores_diff = [y-x for x,y in zip(run_metric_scores_previous,run_metric_scores)]
            baseline_metric_scores_diff = [y-x for x,y in zip(baseline_metric_scores_previous,baseline_metric_scores)]

            win,loses,tie = 0,0,0
            for b,r in zip(baseline_metric_scores_diff,run_metric_scores_diff):
                if b>r:
                    loses=loses+1
                elif b==r:
                    tie= tie +1
                else:
                    win = win+1

            s,p_value=ttest_rel(baseline_metric_scores, run_metric_scores)
            baselin_score, run_score = numpy.mean(baseline_metric_scores),numpy.mean(run_metric_scores),
            # print (s,p_value)
            print "metric: {},{},{}:{},({}),{}\n".format(cutoff,run,metric,run_score,",".join([str(win),str(loses),str(tie)]),str(p_value<0.05))

            s, p_value = ttest_rel(baseline_metric_scores_diff, run_metric_scores_diff)
            baselin_score, run_score = numpy.mean(baseline_metric_scores_diff), numpy.mean(run_metric_scores_diff),
            print "delta: {},{},{}:{},({}),{}\n".format(cutoff, run, metric, run_score, ",".join([str(win), str(loses), str(tie)]),str(p_value < 0.05))

def delta_m(path="results.csv",baseline="lm",runs=["lm"],metrics=["ct"],cutoffs=[]):
    evals  = pd.read_csv(path)
    cutoffs_evals = dict()

    delta_metrics = dict()

    for cutoff in cutoffs:
        for run in runs:
            delta_metrics[":".join([str(cutoff),run])]=dict()

    for cutoff in cutoffs:
        for metric in metrics:
            evals_cutoff = evals[evals['iteration'] == cutoff]
            baseline_evals = evals_cutoff[evals_cutoff['run'] == baseline]
            baseline_evals = baseline_evals[baseline_evals['topic'] != "all"]

            # data frame for drawing.
            _df = pd.DataFrame();
            topics = baseline_evals["topic"].values
            topics = sorted(topics)
            _df["topic"] = topics
            xlabel="topic"

            for run in runs:
                _df[run] = 0

            for x in range(len(_df["topic"])):

                for run in runs:
                    if cutoff > 1:
                        run_df = evals_cutoff[evals_cutoff['run'] == run]
                    run_df = evals_cutoff[evals_cutoff['run'] == run]
                    t = _df.iloc[x][xlabel]

                    run_f = run_df[run_df[xlabel] == t]
                    values = run_f[metric].values
                    y = values[0]

                    # _df.set_value(x, run, y)
                    _df.ix[x, run] = y

            cutoffs_evals[str(cutoff)] = _df


            for run in runs:
                run_metric_scores = _df[run].values

                run_metric_scores_previous = numpy.zeros(len(run_metric_scores))

                if (cutoff > 1):
                    run_metric_scores_previous = cutoffs_evals[str(cutoff - 1)][run].values

                run_metric_scores_diff = [y-x for x,y in zip(run_metric_scores_previous,run_metric_scores)]


                metric_run_score,delta_run_score = numpy.mean(run_metric_scores),numpy.mean(run_metric_scores_diff)
                # print (s,p_value)
                key = ":".join([str(cutoff),run])
                values = dict(chain({"iteration":cutoff,"run":run,metric:metric_run_score,"delta-"+metric:delta_run_score}.iteritems(),delta_metrics[key].iteritems()))
                delta_metrics[key]=values

    for key, value in delta_metrics.iteritems():
        parts = key.split(":")
        metrics_results= [];

        for metric in metrics:
            metrics_results.append(str(value[metric]))
            metrics_results.append(str(value["delta-"+metric]))

        print("{},{}\n".format(",".join(parts),",".join(metrics_results)))


def generate_latex_table(path="evalution.csv",runs=["lm"],baselines=["lm"],symboles=["$^1$","$^2$","$^3$"],metrics=["ct","avg_ct",""],cutoffs=[]):
    """

    Generate latex table
    :param path:
    :param runs:
    :param baseline:
    :param metric:
    :param cutoffs:
    :return:
    """
    evals  = pd.read_csv(path)

    rows = dict()
    for cutoff in cutoffs:
        baseline = baselines[0]
        evals_cutoff = evals[evals['iteration'] == cutoff]
        baseline_evals = evals_cutoff[evals_cutoff['run'] == baseline]
        baseline_evals = baseline_evals[baseline_evals['topic'] != "all"]
        # baseline_scores = baseline_scores[baseline_scores['iteration'] == cutoff]
        # baseline_metric_scores = baseline_scores[metric]


        # data frame for drawing.
        _df = pd.DataFrame();
        sorted_baseline = baseline_evals.sort["topics"]

        # topics = sorted(topics)
        # _df["topic"] = topics
        xlabel="topic"

        for run, group in evals_cutoff.groupby(by="run"):
            sorted_run = group.sort["topic"]

            # for metric in metrics:
            #     baseline_values = sorted_baseline[metric].values
            #     run_values     = sorted_run[metric].values
            #
            #     win, loses, tie = 0, 0, 0
            #     for b,r in zip(baseline_values,run_values):
            #         if b>r:
            #             loses=loses+1
            #         elif b==r:
            #            tie= tie +1
            #         else:
            #             win = win+1
            #
            #     s,p_value=ttest_rel(baseline_values, run_values)
            #     print (s, p_value)

        # for run in runs:
        #     for metric in metrics:
        #         _df[run] = 0
        #
        #         for x in range(len(_df["topic"])):
        #
        #             for run in runs:
        #                 # print (_df)
        #                 run_df = evals_cutoff[evals_cutoff['run'] == run]
        #                 t = _df.iloc[x][xlabel]
        #
        #                 run_f = run_df[run_df[xlabel] == t]
        #                 values = run_f[metric].values
        #                 y = values[0]
        #
        #                 # _df.set_value(x, run, y)
        #                 _df.ix[x, run] = y
        #         for i in range(len(baselines)):
        #             b= baselines[i]
        #             s = symboles[i]
        #             baseline_metric_scores = _df[b].values
        #             for run in runs:
        #
        #                 run_metric_scores = _df[run].values
        #                 baseline_metric_scores = _df[b].values
        #
        #                 win,loses,tie = 0,0,0
        #                 for b,r in zip(baseline_metric_scores,run_metric_scores):
        #                     if b>r:
        #                         loses=loses+1
        #                     elif b==r:
        #                         tie= tie +1
        #                     else:
        #                         win = win+1
        #
        #                 s,p_value=ttest_rel(baseline_metric_scores, run_metric_scores)
        #                 # print (s,p_value)
        #                 # print "{}&{}&{}&{}&{}&{}\n".format(cutoff,run,metric,",".join([str(win),str(loses),str(tie)]),str(p_value<0.05))
        #                 print "{}&{}&{}&{}&{}&{}\\\\\n".format(cutoff, run, metric, ",".join([str(win), str(loses), str(tie)]),str(p_value < 0.05))

def plot_group_by(df,x,y,by,marker_size=200, fontsize='small', label_size=20, title="title",fig_name="analysis",    runs_colors = None):
    markers = itertools.cycle(['>', 'o', '^', 's', '*', 'd', '<', 'v', '>', '8', 'D', '|'])
    colors = itertools.cycle(['black', 'green', 'red', 'blue', 'yellow', 'orange', 'pink',"cyan","magenta"])



    fig, axs = plt.subplots(figsize=(9,4))

    plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.07)
    axs.set_axis_bgcolor('w')

    font = {'family': 'serif',
            'weight': 'normal',
            'size': label_size,
            }
    plt.rc('font',**font)

    plt.rc('xtick', labelsize=label_size)
    plt.rc('ytick', labelsize=label_size)
    plt.rc('axes',labelsize='x-large')

    runs = list()


    if runs_colors ==None:
        runs_colors = dict()
        for name, group in df.groupby(by):
          runs_colors[name]=colors.next()

    for name, group in df.groupby(by):
        runs.append(name)
        group.plot(kind="scatter", x=x, y=y, ax=axs, s=marker_size, c=runs_colors[name], marker=markers.next(),
                   label=name)
        # group.plot.line(x=x, y=y, ax=axs, style=markers.next(),c=runs_colors[name], marker=markers.next(),
        #            label=name)

    # axs.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.62, 1.0),
    #            ncol=3, fancybox=True, shadow=True)
    axs.legend(bbox_to_anchor=(1.03, 1.05))
    plt.title(title,fontsize=fontsize)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    # plt.savefig(fig_name + "-" + by + ".png");


    print ("finish here")

def query_performance(path="evalution.csv",runs=["lm"],baseline="lm",metric="ct",cutoffs=[]):
    evals  = pd.read_csv(path)
    markers = itertools.cycle(['p', '>', '*', '<', 'v', 'o', 'h', 'D'])
    colors = itertools.cycle(['black', 'green', 'red', 'blue', 'yellow', 'orange', 'pink'])

    xlabel="topic"
    ylabel="run"

    for cutoff in cutoffs:
        filtered = evals[evals['iteration'] == cutoff]
        filtered_topics = filtered[filtered['topic'] != -1]
        filtered_metrics= filtered_topics[["topic",metric]]


        run_dfs={}

        # for name, group in filtered.groupby("run"):
        #     run_dfs[name] = group
        #     boxplot_frame_groupby()

        # grouped =filtered_metrics.groupby("topic")
        # boxplot_frame_groupby(grouped,subplots=False)

        print filtered_metrics.columns
        fig=filtered_metrics.boxplot(column=metric,by="topic")
        plt.title(metric + " query performance at iteration " + str(cutoff))
        # get rid of the automatic 'Boxplot grouped by group_by_column_name' title
        plt.suptitle("")
        fig.set_axis_bgcolor("w")


        plt.show()
        plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.07)
        # fig.set_axis_bgcolor('w')
        # fig.title =(metric + " query performance at iteration " + str(cutoff))

        print ("finish iteration")
        # generate_topic_summary(runs=runs,run_dfs=run_dfs,baseline=baseline,xlabel=xlabel,ylabel=ylabel,title="{} {} performance at iteration {}".format(metric,xlabel,str(cutoff)))
        # fig, axs = plt.subplots(figsize=(9, 4))
        # axs.set_axis_bgcolor('w')
        #
        # for name, group in filtered.groupby("run"):
        # # for run in runs:
        # #     run_scores = filtered[filtered['run'] == run]
        #     group.plot( kind="bar",x="topic", y=metric, ax=axs, label=name,color=colors.next(),stacked=True)
        #     #
        #     # axs.legend(fontsize=fontsize)
        #     # plt.title(title, fontsize=fontsize)
        #
        #
        # mng = plt.get_current_fig_manager()
        # mng.window.state('zoomed')
        # plt.show()
        #     # plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.07)
        #     # plt.savefig(fig_name + "-" + by + ".png");


def correlation_matrix(results_csv,metrics):
    df = pd.read_csv(results_csv)
    avg_all_df = df[df["topic"] == "all"]
    out = open("kendall-matrix.csv", mode="w")

    tau_matrix = {}
    all_metrics = []

    for m in metrics:
      for c in [1, 2, 5, 8, 10]:
        all_metrics.append(m + "@" + (str(c)))



    metrics_preferences = {}
    for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        avg_c_df = avg_all_df[avg_all_df["iteration"] == c]


        for m in metrics:
            metrics_preferences[m + "@" + str(c)] = avg_c_df.sort(m, ascending=False)["run"].values


    print ",".join(all_metrics)
    out.write(",".join(all_metrics)+"\n")
    for m in all_metrics:
        m_taus = []
        for m2 in all_metrics:
            tau, p_value = stats.kendalltau(metrics_preferences[m], metrics_preferences[m2])
            m_taus.append("{0:.4f}".format(tau))
        print (m+","+",".join(m_taus))
        out.write(m+","+",".join(m_taus)+ "\n")

    out.close()


def correlation_data(results=["trecdd2016runs.csv"],topics=[],metrics=None):
    out = open("kendall.csv", mode="w")
    print ("topic,base,iteration," + ",".join(metrics))
    out.write("topic,base,iteration," + ",".join(metrics) + "\n")

    metrics_preferences = {}
    for t in topics:
        for c in  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for m in metrics:
                metrics_preferences[t+str(c)+m]=[]

    for result in results:
        df = pd.read_csv(result)
        for t in topics:
            for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                c_df = df[df["iteration"] == c]
                t_c_df = c_df[c_df["topic"]==t]

                result_metrics = set(df.columns).intersection(set(metrics))

                for m in  result_metrics:
                    m_prefs =  t_c_df.sort(m, ascending=False)["run"].values
                    if (len(m_prefs)<1):
                        print ("Somethign wrong m:{}, c:{},m:{}".format(m,str(c),m))

                        t_c_df = c_df[c_df["topic"] == t.replace("DD16-", "")]
                        m_prefs = t_c_df.sort(m, ascending=False)["run"].values

                    metrics_preferences[t+str(c)+m] = m_prefs

    ranking_out = open("runs-rankings.txt", mode="w")
    ranking_out.write("topic,iteration,tau,base,r_ranking,l_rankings\n")
    for t in topics:
        for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for m in metrics:
                ts = []
                for m2 in metrics:
                        m_prefs = metrics_preferences[t + str(c) + m]
                        m2_prefs = metrics_preferences[t + str(c) + m2]

                        #print ("size ({})={}\nsize({})={}".format(m,str(len(m_prefs)),m2,str(len(m2_prefs))))
                        tau, p_value = stats.kendalltau(m_prefs,m2_prefs)
                        ranking_out.write("{},{},{:4f},{}:{},{}:{},\n".format(t,str(c),tau,m, ">".join(m_prefs),m2,">".join(m2_prefs)))
                        # print (tau,p_value)
                        ts.append(str(tau))

                # print ("{},{},{},{}\n".format(t,m, c, ",".join(ts)))
                out.write("{},{},{},{}\n".format(t,m, c, ",".join(ts)))

    out.close()


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
                d[run][topic][str(int(float(iteration)))] = {}

            for m in row.keys() :
                if m not in by:
                    d[run][topic][str(int(float(iteration)))][m]=row[m]

        return d



