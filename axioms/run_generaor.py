import itertools

import math
import multiprocessing
import os
import pickle
import shutil
from multiprocessing import Pool


def trec_dd_run(name, R, M,data_folder,CT):
    d_numbers = {}

    for r in R:
        r_d_numbers = []

        for m in range(1, M + 1):
            r_d_numbers.append(str(m))

        d_numbers[r] = itertools.cycle(r_d_numbers)

    ranks = itertools.cycle([str(m) for m in range(1, M + 1)])
    scores = itertools.cycle([str(m) for m in reversed(range(1, M + 1))])
    lines = []

    for i in range(len(name)):
        r=name[i]
        if r != "_":
            line = "DD16-1\t{}\t{}{}\t{}\t{}".format(str((i/CT)),r, d_numbers[r].next(), scores.next(), R[r])
            lines.append(line.strip())

    ranking = "\n".join(lines)
    if len(lines) > 0:
        file_name = os.path.join(data_folder,name + ".txt")
        f = open(file_name, "w")
        f.write(ranking.strip())
        f.close()
        print (ranking)

def trec_dd_qrels(A, M,data_folder):
    from lxml import etree as ET

    trec_dd=ET.Element("trec_dd")

    domain = ET.SubElement(trec_dd,"domain",name="axioms",total_topic_num = "1",total_subtopic_num=str(len(A)))

    topic_id = "DD16-1"
    num_of_subtopics = len(A)
    topic=ET.SubElement(domain,"topic",name="Axioms query",id=topic_id,num_of_subtopics=str(num_of_subtopics))
    doc_lengths={}

    for i in range(len(A)):
        a=A[i]

        subtopic = ET.SubElement(topic,"subtopic",name=a,id="{}.{}".format(topic_id,str(i+1)), num_of_passages=str(M))

        for m in range(1, M + 1):
            passage = ET.SubElement(subtopic,"passage",id="{}{}1".format(str(i),str(m)))
            doc_lengths["{}{}".format(a,str(m))]=0
            ET.SubElement(passage,"docno").text="{}{}".format(a,str(m))
            ET.SubElement(passage, "rating").text = "1"
            ET.SubElement(passage, "text").text = "Text"
            ET.SubElement(passage, "type").text = "manual"

    pickle.dump(doc_lengths,file=open(os.path.join(data_folder,"qrels","trecdd","{}x-{}-params.pkl".format("".join(sorted(A)),str(M))), 'wb'))

    xml_file = open(os.path.join(data_folder,"qrels","trecdd","{}x-{}-truth.xml".format("".join(sorted(A)),str(M))),"w")
    xml_file.write('<?xml version="1.0"?>\n{}'.format(ET.tostring(trec_dd,pretty_print=True)))



def trec_web_run(name, R, M,data_folder):
    d_numbers = {}

    for r in R:
        r_d_numbers = []

        for m in range(1, M + 1):
            r_d_numbers.append(str(m))

        d_numbers[r] = itertools.cycle(r_d_numbers)

    ranks = itertools.cycle([str(m) for m in range(1, M + 1)])
    scores = itertools.cycle([str(m) for m in reversed(range(1, M + 1))])
    lines = []
    for r in name:
        if r != "_":
            line = "DD16-1\tQ0\t{}{}\t{}\t{}\t{}".format(r, d_numbers[r].next(), ranks.next(), scores.next(), name)
            lines.append(line)

    ranking = "\n".join(lines)
    if len(lines) > 0:
        file_name = os.path.join(data_folder, name + ".txt")
        f = open(file_name, "w")
        f.write(ranking.strip())
        f.close()
        print (ranking)

def trec_adhoc_run(name, R, M,data_folder):
    d_numbers = {}

    for r in R:
        r_d_numbers = []

        for m in range(1, M + 1):
            r_d_numbers.append(str(m))

        d_numbers[r] = itertools.cycle(r_d_numbers)

    ranks = itertools.cycle([str(m) for m in range(1, M + 1)])
    scores = itertools.cycle([str(m) for m in reversed(range(1, M + 1))])
    lines = []
    for r in name:
        if r != "_":
            line = "1\tQ0\t{}{}\t{}\t{}\t{}".format(r, d_numbers[r].next(), ranks.next(), scores.next(), name)
            lines.append(line)

    ranking = "\n".join(lines)
    if len(lines) > 0:
        file_name = os.path.join(data_folder,name + ".txt")
        f = open(file_name, "w")
        f.write(ranking.strip())
        f.close()
        print (ranking)


def generate_trec_runs(run, R, M,data_folder):
        name = "".join(run)
        dd_runs_folder = os.path.join(data_folder, "runs", "trecdd-fake-runs", str(M),"".join(sorted(R.keys())))
        web_runs_folder = os.path.join(data_folder, "runs", "trecweb-fake-runs", str(M),"".join(sorted(R.keys())))
        adhoc_runs_folder = os.path.join(data_folder, "runs", "trecadhoc-fake-runs", str(M),"".join(sorted(R.keys())))

        for CT in range(1,11):
          CT_PATH = os.path.join(dd_runs_folder,str(CT))
          trec_dd_run(name, R, M,os.path.join(dd_runs_folder,str(CT)),CT)


        #trec_web_run(name, R, M,web_runs_folder)
        #if (len(R.keys()) ==2):
         #   trec_adhoc_run(name, R, M,adhoc_runs_folder)


def generate_trec_runs_wrapper(args):
    generate_trec_runs(*args)


def generate_names(a, M):
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

def generate_runs(a, M,data_folder):

    cores = multiprocessing.cpu_count()
    pool = Pool(cores)



    A = a + ["x"]

    R = {}
    dd_runs_folder = os.path.join(data_folder, "runs", "trecdd-fake-runs", str(M), "".join(sorted(A)))
    web_runs_folder = os.path.join(data_folder, "runs", "trecweb-fake-runs", str(M), "".join(sorted(A)))
    adhoc_runs_folder = os.path.join(data_folder, "runs", "trecadhoc-fake-runs", str(M), "".join(sorted(A)))


    shutil.rmtree(dd_runs_folder, ignore_errors=True)
    os.mkdir(dd_runs_folder)

    for CT in range(1,11):
        shutil.rmtree(os.path.join(dd_runs_folder,str(CT)), ignore_errors=True)
        os.mkdir(os.path.join(dd_runs_folder,str(CT)))

    shutil.rmtree(web_runs_folder, ignore_errors=True)
    os.mkdir(web_runs_folder)
    shutil.rmtree(adhoc_runs_folder, ignore_errors=True)
    os.mkdir(adhoc_runs_folder)

    for r in a:
        R[r] = "1"

    R["x"] = "0"
    # runs = generate_names(a,M)
    # runs = set(runs)

    runs = [""]  # empty set

    level_nodes = []
    level_nodes.append([""])


    # pool.map(generate_trec_runs_wrapper,itertools.izip(runs,itertools.repeat(R),itertools.repeat(M),itertools.repeat(data_folder)))

    for m in range(1, M + 1):
        level_nodes.append([])
        for node in level_nodes[m - 1]:
            for x in A:

                child = node + x
                level_nodes[m].append(child)
                print ("generating " +child)
                generate_trec_runs(child, R, M, data_folder)







if __name__ == '__main__':
    a = ["a","b"]

    # Ms = [5,10]
    Ms = [5]


    for m in Ms:
        trec_dd_qrels(a,m,data_folder=os.path.join("..","..","metric-evalutions"))
        generate_runs(a,m,data_folder=os.path.join("..","..","metric-evalutions"))