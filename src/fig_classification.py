#!/usr/bin/env python3

from utils import read_json, read_tfidf, folder_generator_graph
import matplotlib.pyplot as plt
import numpy as np
import fig_utils
import scipy.stats as st
from argparse import ArgumentParser
import os
from utils import computed_path_generator
from pathlib import Path

def confidence(vals):
    return st.t.interval(
        alpha=0.95,
        df=len(vals) - 1,
        loc=np.mean(vals),
        scale=st.sem(vals)
    )


args = ArgumentParser()
args.add_argument("-d", "--data", default="computed/mlp_CoLA_BERT.json")
args = args.parse_args()

data_path = args.data.replace("Representations","computed")

graph_path = args.data.replace("Representations","graphs")
folder_generator_graph(graph_path)

rep_file_list = ["mlp_BERT.json","mlp_GPT2.json","mlp_SBERT.json"]
case_name = data_path.split("/")[-1]
print(case_name)

tf_idf = float(read_tfidf(case_name)) 

for data_file in rep_file_list:
    data = read_json(os.path.join(data_path,data_file))
    print(data)
    PLTARGS = dict(
    capsize=3, capthick=2,
    ms=10, marker=".",
    elinewidth=1
    )
    if data_file=="mlp_GPT2.json":
        rep_list = ["Mean", "Haddamard", "Sum"]
    else:
        rep_list = ["Mean", "Haddamard", "Sum", "CLS"]

    for k in rep_list:
        # take test scores
        ys = [[x[1] for x in data[k.lower()][str(layer)]] for layer in range(12)]
        cs = [confidence(y) for y in ys]
        yerr = [(x[1] - x[0]) / 2 for x in cs]
        plt.errorbar(
            list(range(12)),
            [np.average(y) for y in ys],
            label=k,
            yerr=yerr,
            **PLTARGS
        )

    if data_file!="mlp_GPT2.json":
        # special pooler handling
        ys = [x[1] for x in data["pooler"]["0"]]
        cs = confidence(ys)
        yerr = (cs[1] - cs[0]) / 2
        plt.errorbar(
            [10.5],
            [np.average(ys)],
            label="Pooler",
            yerr=yerr,
            ** PLTARGS
        )

    plt.hlines(
        0.5, 0, 11, label="MCCC",
        linestyle=":", color="tab:gray"
    )

    if tf_idf is not None:
        plt.hlines(
            tf_idf, 0, 11, label="TF-IDF",
            linestyle="-.", color="tab:gray"
        )

    plt.ylabel("Dev accuracy")
    plt.xlabel("Layer")

    plt.tight_layout(pad=0.1)
    plt.legend(ncol=2)
    
    loc = os.path.join(graph_path,data_file.replace(".json", ".pdf"))
    print(loc)
    plt.savefig(loc)
    # plt.show()
    plt.clf()
    plt.close()


#     if model_name=="BERT":
#         # special pooler handling
#         ys = [x[1] for x in data["pooler"]["0"]]
#         cs = confidence(ys)
#         yerr = (cs[1] - cs[0]) / 2
#         plt.errorbar(
#             [10.5],
#             [np.average(ys)],
#             label="Pooler",
#             yerr=yerr,
#             ** PLTARGS
#         )

#     plt.hlines(
#         0.5, 0, 11, label="MCCC",
#         linestyle=":", color="tab:gray"
#     )

#     if tf_idf is not None:
#         plt.hlines(
#             tf_idf, 0, 11, label="TF-IDF",
#             linestyle="-.", color="tab:gray"
#         )

#     plt.ylabel("Dev accuracy")
#     plt.xlabel("Layer")

#     plt.tight_layout(pad=0.1)
#     plt.legend(ncol=2)
    


# tf_idf = float(read_tfidf(args.data,args.target))

# data = read_json(args.data)



# PLTARGS = dict(
#     capsize=3, capthick=2,
#     ms=10, marker=".",
#     elinewidth=1
# )
# if model_name=="SBERT":
#     ys = [x[1] for x in data['embedding']['0']]
#     reps = ["SBERT","tf_idf"]
#     x = np.arange(2)
#     y = [np.average(ys),tf_idf]
#     yerr = np.std(y)
#     fig, ax = plt.subplots()
#     ax.bar(x, y, yerr=yerr, align='center', alpha=0.5, ecolor='black', capsize=10)
#     ax.set_ylabel('Dev Accuracy')
#     ax.set_ylabel('Representations')
#     ax.set_xticks(x)
#     ax.set_xticklabels(reps)
#     ax.yaxis.grid(True)

# else:
#     if model_name=="BERT":
#         rep_list = ["Mean", "Haddamard", "Sum", "CLS"]
#     elif model_name=="GPT":
#         rep_list = ["Mean", "Haddamard", "Sum"]

#     for k in rep_list:
#         # take test scores
#         ys = [[x[1] for x in data[k.lower()][str(layer)]] for layer in range(12)]
#         cs = [confidence(y) for y in ys]
#         yerr = [(x[1] - x[0]) / 2 for x in cs]
#         plt.errorbar(
#             list(range(12)),
#             [np.average(y) for y in ys],
#             label=k,
#             yerr=yerr,
#             **PLTARGS
#         )

#     if model_name=="BERT":
#         # special pooler handling
#         ys = [x[1] for x in data["pooler"]["0"]]
#         cs = confidence(ys)
#         yerr = (cs[1] - cs[0]) / 2
#         plt.errorbar(
#             [10.5],
#             [np.average(ys)],
#             label="Pooler",
#             yerr=yerr,
#             ** PLTARGS
#         )

#     plt.hlines(
#         0.5, 0, 11, label="MCCC",
#         linestyle=":", color="tab:gray"
#     )

#     if tf_idf is not None:
#         plt.hlines(
#             tf_idf, 0, 11, label="TF-IDF",
#             linestyle="-.", color="tab:gray"
#         )

#     plt.ylabel("Dev accuracy")
#     plt.xlabel("Layer")

#     plt.tight_layout(pad=0.1)
#     plt.legend(ncol=2)

# if args.target=="amb":
#     if not  os.path.exists(os.path.join(os.path.join(os.getcwd(),"graphs"),"Ambiguity")):
#         os.mkdir(os.path.join(os.path.join(os.getcwd(),"graphs"),"Ambiguity"))
#     loc = os.path.join(os.path.join(os.getcwd(),os.path.join("graphs","Ambiguity")),(args.data.replace(".json", ".pdf")).replace("computed/",""))
# if args.target=="class":
#     if not  os.path.exists(os.path.join(os.path.join(os.getcwd(),"graphs"),"Grammaticality")):
#         os.mkdir(os.path.join(os.path.join(os.getcwd(),"graphs"),"Grammaticality"))
#     loc = os.path.join(os.path.join(os.getcwd(),os.path.join("graphs","Grammaticality")),(args.data.replace(".json", ".pdf")).replace("computed/",""))

# plt.savefig(loc)
# plt.show()
