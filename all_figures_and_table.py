#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:58:23 2023

@author: ajaykumar
"""

import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

def get_data(dg):


    with open('pos_files/pos_all.json', 'r') as fp:
        pos = json.load(fp)

    node_levels = defaultdict(list)
    for n,d in dg.nodes(data=True):
        node_levels[d["level"]].append(n)
    groups = {
              "carparks": node_levels["B2"]+node_levels["B1"],
              "L1": node_levels["L1"],
              "L2": node_levels["L2"],
              "L3-L4": node_levels["L3"]+["L4_CLL"],
              "L6": node_levels["L6"],
             }

    r7 = ['L7_R1_LL', 'L7_R1_LL_corridor', 'L7_R2_LL_corridor', 'L7_R2_LL']
    c7 = [ n for n in node_levels["L7"] if n not in r7 ]
    groups["L7"] = c7

    r8 = ['L8_R1_LL', 'L8_R1_LL_corridor', 'L8_R2_LL_corridor', 'L8_R2_LL']
    c8 = [ n for n in node_levels["L8"] if n not in r8 ]
    r9 = ['L9_R1_LL', 'L9_R1_LL_corridor', 'L9_R2_LL_corridor', 'L9_R2_LL']
    c9 = [ n for n in node_levels["L9"] if n not in r9 ]
    groups["rooftop_garden"] = c8 + c9

    r4 = [ n for n in node_levels["L4"] if not n=="L4_CLL"]
    r5 = node_levels["L5"]
    r6 = ['L6_Deck_1', 'L6_Deck_2', 'L6_Deck_3', 'L6_Deck_4', 'L6_R1_LL_corridor', 'L6_R1_LL', 'L6_Link_bridge', 'L6_R2_LL_corridor', 'L6_R2_LL']
    r10 = node_levels["L10"]
    r11 = node_levels["L11"]
    groups["residential"] = r4+r5+r6+r7+r8+r9+r10+r11
    node_group = {}
    for k,v in groups.items():
        #print(k,v)
        for n in v:
            node_group[n] = k

    return dg, pos, node_group, groups


def get_node_colors(dg, node_group):
    clrs = sns.color_palette("tab10")
    # sns.palplot(clrs)
    #group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop garden', 'residential']
    group_levels = [ 'residential', 'L1', 'rooftop_garden', 'L2', 'carparks', 'L6' ]
    group_colors = { group_levels[i]: clrs[i] for i in range(len(group_levels))}
    group_colors["L7"] = clrs[6]
    group_colors["L3-L4"] = clrs[6]
    group_colors["crossing"] = clrs[7]
    nodelist = list(dg.nodes())
    loc_listb = group_levels
    loc_listb2 = [ a.replace("_", " ").capitalize() for a in loc_listb ] + ["Other"]
    #loc_clrsb = { loc_listb[i]:clrs[i] for i in range(len(loc_listb)) }
    n_clrsb = [ group_colors[node_group[n]] for n in nodelist ]
    return clrs, n_clrsb, group_colors

def draw_fig2_network(dg, pos, clrs, n_clrsb, group_colors,fig_name):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="w")
    nodelist = list(dg.nodes())
    node_incoming = dg.in_degree(weight="mobility_number")
    nsize = [ (node_incoming[n]+1)*2 for n in nodelist ]
    nx.draw_networkx_nodes(dg, nodelist=nodelist, pos=pos, ax=ax, node_size=nsize, node_color=n_clrsb)
    #nx.draw_networkx_edges(dg, pos=pos, ax=ax, )
    edgelist = dg.edges()
    #print(edgelist)
    edgeweight = [ dg[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*.75+1 if w>0 else 1 for w in edgeweight ]
    nx.draw_networkx_edges(dg, pos=pos, ax=ax, edgelist=edgelist, width=edgewidth,
                           connectionstyle="arc3,rad=0.1", edge_color="xkcd:navy", alpha=.6)

    clabels = [ 'carparks', 'L1', 'L2', 'L6', 'rooftop_garden',  'residential']#, "Other" ]#loc_listb2
    clabels2 = [ 'Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden',  'Residential']
    clrs2 = [ group_colors[k] for k in clabels ]
    legend_elements = []
    for l,c in zip(clabels2, clrs2):
        ll = l.replace("_", " ").capitalize()
        l2d = Line2D([0], [0], color=c, lw=0, marker="o", ls="", label=ll)
        legend_elements.append(l2d)
    l2do = Line2D([0], [0], color=clrs[6], lw=0, marker="o", ls="", label="Other")
    legend_elements.append(l2do)
    ax.legend(handles=legend_elements, ncol=2, loc='lower right')
    ax.set_xlim([-0.8, 0.7])
    ax.set_ylim([-1.1, 0.8])
    #add histogram
    ax = ax.inset_axes([0.8, 0.15, 0.15, 0.15])
    bins=int(max(edgeweight)/2.0)+1
    ax.hist(edgeweight,bins=bins,color="xkcd:navy")
    #add legend
    lines = []
    edges_weight_list = [min(edgewidth),np.median(edgewidth),np.percentile(edgewidth,75),max(edgewidth)]
    texts_edges_weight_list = [min(edgeweight),np.median(edgeweight),np.percentile(edgeweight,75),max(edgeweight)]
    for i, width in enumerate(edges_weight_list):
        lines.append(Line2D([],[], linewidth=width,alpha=.6, color="xkcd:navy"))
    ax.legend(lines, texts_edges_weight_list, bbox_to_anchor=(-4.4, 5.7), frameon=False) 

    plt.tight_layout()
    # ax.set_title('(a) Mobility graph', loc="left")
    fig.savefig(os.path.join("figs_202206", "fig2"+fig_name+".png"), dpi=300, bbox_inches="tight")
    #plt.show()

def get_group_count(dg, groups):
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']

    group_count = {}
    #loc_listb = ['Community_Street', 'Garden_Street', 'Residential_Street', 'Commercial_Street', 'Social_Space', 'Corridor', 'Vertical_Street', 'Entrance_Street']
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']

    for i,k in enumerate(group_levels):
        #temp_pos = pos_by_floor[k]
        ns = groups[k]
        kg = dg.subgraph(ns)
        edgelist = kg.edges()
        edgeweight = [ kg[u][v]["mobility_number"] for u,v in edgelist ]
        destinations = [ v for u,v in edgelist ]
        dest_cat = [ dg.nodes[n]["location_type"] for n in destinations ]
        temp = defaultdict(int)
        for c,v in zip(dest_cat, edgeweight):
            temp[c]+=v
        temp2 = [ temp[c] for c in loc_listb ]
        group_count[k] = temp2
        #break
    oth = [ k for k in groups.keys() if not(k in group_levels) ]
    temp = defaultdict(int)
    for k in oth:
        #print(k)
        ns = groups[k]
        kg = dg.subgraph(ns)
        edgelist = kg.edges()
        edgeweight = [ kg[u][v]["mobility_number"] for u,v in edgelist ]
        destinations = [ v for u,v in edgelist ]
        dest_cat = [ dg.nodes[n]["location_type"] for n in destinations ]
        #temp = defaultdict(int)
        for c,v in zip(dest_cat, edgeweight):
            temp[c]+=v
    temp2 = [ temp[c] for c in loc_listb ]
    group_count["other"] = temp2
    #print(group_count)
    df_group_count = pd.DataFrame.from_dict(group_count, orient='index', columns=loc_listb)
    loc_listb2 = [ a.split("_")[0].lower() if not(a=="Corridor") else "corridor" for a in loc_listb ]

    for i in range(len(loc_listb2)):
        df_group_count[loc_listb2[i]] = df_group_count.iloc[:,:i+1].sum(axis=1)
    df_group_count
    return df_group_count, group_count, loc_listb

def draw_fig1_barh(dg, groups, clrs,fig_name):
    df_group_count, group_count, loc_listb = get_group_count(dg, groups)
    loc_listb2 = [ a.split("_")[0].lower() if not(a=="Corridor") else "corridor" for a in loc_listb ]
    loc_clrsb = { loc_listb2[i]:clrs[i] for i in range(len(loc_listb)) }

    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0].lower() for a in loc_listb ]
    n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ]
    loc_clrsb = {k:v for k,v in zip(loc_listb2, clrs)}

    group_levels_label = ['Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential', 'Other']

    fig, ax = plt.subplots(1, 1, figsize=(6,6), facecolor="w")

    for i,c in enumerate(loc_listb2):
        vs = df_group_count[c].tolist()
        ys = list(range(len(vs)))
        #c2 = loc_listb2[i]
        ax.barh(ys, vs, fc=loc_clrsb[c], label=c.capitalize(), zorder=len(loc_listb2)-i+2, alpha=.9)
    #ax.set_title("({}) {}".format(labs[i], k.capitalize()), loc="left")
    ax.legend(ncol=3, loc='upper right')
    ax.set_yticks(list(range(len(group_count))))
    #ax.set_yticklabels([y.capitalize().replace("_", " ") for y in list(group_count.keys())])
    ax.set_yticklabels(group_levels_label)
    ax.set_xlabel("Incoming flow")
    #ax.set_xscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join("figs_202206", "fig1"+fig_name+".png"), dpi=300, bbox_inches="tight")
    #plt.show()


def get_clr_fig3():
    clrs = sns.color_palette("tab10")
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    loc_clrsb = {k:v for k,v in zip(loc_listb, clrs)}
    #n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ]
    return loc_clrsb, loc_listb

def draw_fig3_networks(dg, groups,fig_name):
    with open('pos_files/pos_all_by_floor_w_ext.json', 'r') as fp:
        pos_by_floor_ext = json.load(fp)
    groups_with_ext = {}
    for k,ns in groups.items():
        #print(len(ns))
        ns2 = []#ns.copy()
        for n in ns:
            temp_ns = list(dg.predecessors(n))
            ns2.extend(temp_ns)
            temp_ns = list(dg.successors(n))
            ns2.extend(temp_ns)
            #break
        ns2 = ns+ns2
        ns2 = list(set(ns2))
        #print(len(ns2))
        groups_with_ext[k] = ns2
        #break
    loc_clrsb, loc_listb = get_clr_fig3()
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']
    group_levels_label = ['Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential']


    fig, axg = plt.subplots(2, 3, figsize=(12,8), facecolor="w")
    axs = axg.flatten()
    labs = "abcdef"
    for i,k in enumerate(group_levels):
        k2 = group_levels_label[i]
        ax = axs[i]
        temp_pos = pos_by_floor_ext[k]
        ns = groups[k]
        ns2 = groups_with_ext[k]
        ext_ns = list(set(ns2) - set(ns))

        kg = dg.subgraph(ns)
        kg2 = dg.subgraph(ns2)
        n_clrs_temp = [ loc_clrsb[dg.nodes[n]["location_type"]] for n in ns ]
        nx.draw_networkx_nodes(kg2, nodelist=ns, pos=temp_pos, ax=ax, node_size=35, node_color=n_clrs_temp)
        #nx.draw_networkx_nodes(kg2, nodelist=ext_ns, pos=temp_pos, ax=ax, node_size=1, node_color='lightgray')
        edgelist = kg.edges()
        edgeweight = [ kg[u][v]["mobility_number"] for u,v in edgelist ]
        edgewidth = [ (np.log2(w)*1+1)/2 if w>0 else 1 for w in edgeweight ]
        nx.draw_networkx_edges(kg2, pos=temp_pos, ax=ax, edgelist=edgelist, width=edgewidth,
                               connectionstyle="arc3,rad=0.1", edge_color="xkcd:navy", alpha=.6)
        edgelist2 = (kg2.edges())
        edgelist_ext = []
        for u,v in edgelist2:
            if (u,v) in edgelist:
                continue
            if (u in ns) or (v in ns):
                edgelist_ext.append((u,v))
        nx.draw_networkx_edges(kg2, pos=temp_pos, ax=ax, edgelist=edgelist_ext, width=.5,
                               connectionstyle="arc3,rad=0.1", arrowsize=1, edge_color="xkcd:navy", alpha=.8)
        #ax.text(0.01,0.99,k, ha="left", va="top", transform=ax.transAxes, )
        ax.set_title("({}) {}".format(labs[i], k2), loc="left")
        xlim_0 = ax.get_xlim()
        ylim_0 = ax.get_ylim()
        zoom = 0.85 if i<4 else 0.88
        xlim_1 = [ x*zoom for x in xlim_0 ]
        ylim_1 = [ y*zoom for y in ylim_0 ]
        ax.set_xlim(xlim_1)
        ax.set_ylim(ylim_1)
        #add histogram
        ax = ax.inset_axes([0.8, 0.08, 0.15, 0.15])
        bins=int(max(edgeweight)/2.0)+1
        ax.hist(edgeweight,bins=bins,color="xkcd:navy")
        #add legend
        lines = []
        edges_weight_list = [min(edgewidth),np.median(edgewidth),max(edgewidth)]
        texts_edges_weight_list = [min(edgeweight),np.median(edgeweight),max(edgeweight)]
        for i, width in enumerate(edges_weight_list):
            lines.append(Line2D([],[], linewidth=width,alpha=.8, color="xkcd:navy"))
        ax.legend(lines, texts_edges_weight_list, bbox_to_anchor=(-3.7, 6.3), frameon=False) 

    clabels = loc_listb
    clrs = [ loc_clrsb[c] for c in clabels ]
    legend_elements = []
    for l,c in zip(clabels, clrs):
        ll = l.split("_")[0]
        l2d = Line2D([0], [0], color=c, lw=0, marker="o", ls="", label=ll)
        legend_elements.append(l2d)
    fig.legend(handles=legend_elements, ncol=8, loc='lower center', bbox_to_anchor=[.5, -.03])
    plt.tight_layout()
    fig.savefig(os.path.join("figs_202206", "fig3"+fig_name+".png"), dpi=300, bbox_inches="tight")
    #plt.show()


def prepare_for_fig4(dg_floor, dg_cat, group_levels, loc_listb2):
    clrs = sns.color_palette("tab10")
    n_clrs_level = [ clrs[i] for i in range(len(group_levels)) ]
    n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ]
    nsize_floor = []
    for n in group_levels:
        edw = dg_floor[n][n]["mobility_number"]
        edw = edw+10
        nsize_floor.append(edw)
    nsize_cat = []
    for n in loc_listb2:
        if dg_cat.has_edge(n,n):
            edw = dg_cat[n][n]["mobility_number"]
        else:
            edw = 1
        edw = edw+10
        nsize_cat.append(edw)
    floor_lab2 = {
        "Carparks":"Car-parks",
        "L1":"Level-1",
        "L2":"Level-2",
        "L6":"Level-6",
        "Rooftop garden":"Rooftop garden",
        "Residential":"Residential",
    }
    with open('pos_files/pos_floor_group.json', 'r') as fp:
        pos_floor = json.load(fp)
    with open('pos_files/pos_cat.json', 'r') as fp:
        pos_cat = json.load(fp)
    pos_floor_lab = {
     'Carparks': (0.25, 1.08),
     'L1': (0.13, 0.7),
     'L2': (-0.25, 0.26),
     'L6': (-0.12, -0.65),
     'Rooftop garden': (-0.3, -1.1),
     'Residential': (0.28, -0.3)}
    pos_cat_lab = {
     'Community': [-0.63, 0.24],
     'Garden': [-0.64, -0.35],
     'Residential': [0.28, -1.0],
     'Commercial': [0.17, 0.68],
     'Social': [0.34, -0.32],
     'Corridor': [-0.4, 0.6],
     'Vertical': [-0.08, -0.28],
     'Entrance': [0.63, 0.48]}


    return nsize_floor, nsize_cat, floor_lab2, pos_floor, pos_cat, pos_floor_lab, pos_cat_lab, n_clrs_level, n_clrs_cat

def make_dg_floor(dg, groups):
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential', 'L3-L4', 'L7']
    groups2 = { k:groups[k] for k in group_levels }
    temp = []
    oth = [ k for k in groups.keys() if not(k in group_levels) ]
    for k in oth:
        ns = groups[k]
        temp.extend(ns)
    # print(temp)
    groups2["other"] = temp
    floor_dic = {}
    for k,ns in groups2.items():
        k2 = k.replace("_", " ").capitalize()
        for n in ns:
            floor_dic[n] = k2
    group_levels = [ l.capitalize() for l in ['carparks', 'L1', 'L2', 'L6', 'rooftop garden', 'residential', 'L3-L4', 'L7'] ]
    dg_floor = nx.DiGraph()
    dg_floor.add_nodes_from(group_levels)
    for u,v,d in dg.edges(data=True):
        u2 = floor_dic[u]
        v2 = floor_dic[v]
        if d["mobility_number"]<=0: continue
        if dg_floor.has_edge(u2,v2):
            dg_floor[u2][v2]["mobility_number"]+=d["mobility_number"]
        else:
            dg_floor.add_edge(u2,v2, mobility_number=d["mobility_number"])
    group_levels = [ l.capitalize() for l in ['carparks', 'L1', 'L2', 'L6', 'rooftop garden', 'residential'] ]

    dg_floor2 = nx.DiGraph()
    dg_floor2.add_nodes_from(group_levels)
    check = ['L3-l4', 'L7']
    check_edge1 = {}
    check_edge2 = {}
    for u,v,d in dg_floor.edges(data=True):
        if not(u in check) and not(v in check):
            #print(u,v,d)
            dg_floor2.add_edge(u, v, mobility_number=d["mobility_number"])
        elif u=="L3-l4" or v=="L3-l4":
            check_edge1[(u,v)] = d["mobility_number"]
            #print(u,v,d)
        elif u=="L7" or v=="L7":
            check_edge2[(u,v)] = d["mobility_number"]
            print(u,v,d)
    # print(dg_floor2.has_edge("L2", "L6"), dg_floor2.has_edge("L6", "L2"))
    six_to_two = min(check_edge1[("L6", "L3-l4")], check_edge1[("L3-l4", "L2")])
    # print(six_to_two)
    # print(dg_floor2["L6"]["L2"])
    dg_floor2["L6"]["L2"]["mobility_number"]+=six_to_two
    # print(dg_floor2["L6"]["L2"])
    two_to_six = min(check_edge1[("L3-l4", "L6")], check_edge1[("L2", "L3-l4")])
    # print(two_to_six)
    # print(dg_floor2["L2"]["L6"])
    dg_floor2["L2"]["L6"]["mobility_number"]+=two_to_six
    # print(dg_floor2["L2"]["L6"])
    # print(dg_floor2.has_edge("Rooftop garden", "L6"), dg_floor2.has_edge("L6", "Rooftop garden"))
    dg_floor2.add_edge("Rooftop garden", "L6", mobility_number=0)
    dg_floor2.add_edge("L6", "Rooftop garden", mobility_number=0)
    six_to_top = min(check_edge2[("L6", "L7")], check_edge2[("L7", "Rooftop garden")])
    # print(six_to_top)
    # print(dg_floor2["L6"]["Rooftop garden"])
    dg_floor2["L6"]["Rooftop garden"]["mobility_number"]+=six_to_top
    # print(dg_floor2["L6"]["Rooftop garden"])
    top_to_six = min(check_edge2[("L7", "L6")], check_edge2[("Rooftop garden", "L7")])
    # print(top_to_six)
    # print(dg_floor2["Rooftop garden"]["L6"])
    dg_floor2["Rooftop garden"]["L6"]["mobility_number"]+=top_to_six
    # print(dg_floor2["Rooftop garden"]["L6"])
    dg_floor = dg_floor2
    return dg_floor

def make_dg_cat(dg):
    cat_dic = {}
    for n,d in dg.nodes(data=True):
        cat_dic[n] = d["location_type"].split("_")[0]
    loc_listb = ['Community_Street', 'Garden_Street', 'Residential_Street', 'Commercial_Street', 'Social_Space', 'Corridor', 'Vertical_Street', 'Entrance_Street']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    dg_cat = nx.DiGraph()
    dg_cat.add_nodes_from(loc_listb2)
    for u,v,d in dg.edges(data=True):
        u2 = cat_dic[u]
        v2 = cat_dic[v]
        if d["mobility_number"]<=0: continue
        if dg_cat.has_edge(u2,v2):
            dg_cat[u2][v2]["mobility_number"]+=d["mobility_number"]
        else:
            dg_cat.add_edge(u2,v2, mobility_number=d["mobility_number"])
    return dg_cat

def draw_fig4_cat_network(dg, groups,fig_name):
    dg_floor = make_dg_floor(dg, groups)
    dg_cat = make_dg_cat(dg)

    group_levels = [ l.capitalize() for l in ['residential', 'L1', 'rooftop garden', 'L2', 'carparks', 'L6' ]]#, 'other'] ]
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    nsize_floor, nsize_cat, floor_lab2, pos_floor, pos_cat, pos_floor_lab, pos_cat_lab, n_clrs_level, n_clrs_cat = prepare_for_fig4(dg_floor, dg_cat, group_levels, loc_listb2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    ax1,ax2 = axs
    nx.draw_networkx_nodes(dg_floor, nodelist=group_levels, pos=pos_floor, ax=ax1, node_size=nsize_floor, node_color=n_clrs_level)
    edgelist = dg_floor.edges()
    edgelist = [(u, v) for u, v in edgelist if not(u==v)]
    edgeweight = [ dg_floor[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
    nx.draw_networkx_edges(dg_floor, pos=pos_floor, ax=ax1, edgelist=edgelist, width=edgewidth,
                           connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
    nx.draw_networkx_labels(dg_floor, pos=pos_floor_lab, labels=floor_lab2, ax=ax1)
    ax1.set_title("(a) Between floor groups", loc="left")
    # print(edgeweight)
    #add histogram
    ax1 = ax1.inset_axes([0.82, 0.15, 0.15, 0.15])
    #bins=int(max(edgeweight)/2.0)+1
    bins=15
    ax1.hist(edgeweight,bins=bins,color="xkcd:navy")
    #add legend
    lines = []
    edges_weight_list = [min(edgewidth),np.median(edgewidth),max(edgewidth)]
    texts_edges_weight_list = [min(edgeweight),np.median(edgeweight),max(edgeweight)]
    for i, width in enumerate(edges_weight_list):
        lines.append(Line2D([],[], linewidth=width,alpha=.6, color="xkcd:navy"))
    ax1.legend(lines, texts_edges_weight_list, bbox_to_anchor=(-4.2, 5.7), frameon=False) 

    #---------------------------------------
    nx.draw_networkx_nodes(dg_cat, nodelist=loc_listb2, pos=pos_cat, ax=ax2, node_size=nsize_cat, node_color=n_clrs_cat)
    edgelist = dg_cat.edges()
    edgelist = [(u, v) for u, v in edgelist if not(u==v)]
    edgeweight = [ dg_cat[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
    nx.draw_networkx_edges(dg_cat, pos=pos_cat, ax=ax2, edgelist=edgelist, width=edgewidth,
                           connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
    nx.draw_networkx_labels(dg_cat, pos=pos_cat_lab, ax=ax2)
    ax2.set_title("(b) Between location categories", loc="left")
    #add histogram
    ax2 = ax2.inset_axes([0.82, 0.15, 0.15, 0.15])
    #bins=int(max(edgeweight)/2.0)+1
    bins=15
    ax2.hist(edgeweight,bins=bins,color="xkcd:navy")
    #add legend
    lines = []
    edges_weight_list = [min(edgewidth),np.median(edgewidth),max(edgewidth)]
    texts_edges_weight_list = [min(edgeweight),np.median(edgeweight),max(edgeweight)]
    for i, width in enumerate(edges_weight_list):
        lines.append(Line2D([],[], linewidth=width,alpha=.6, color="xkcd:navy"))
    ax2.legend(lines, texts_edges_weight_list, bbox_to_anchor=(-4.2, 5.7), frameon=False) 
    # print(edgeweight)
    plt.tight_layout()
    plt.savefig(os.path.join("figs_202206", "fig4"+fig_name+".png"), dpi=300, bbox_inches="tight")
    #plt.show()

def check_sig(r, p):
    r2 = ""
    if p<=0.001:
        r2 = "{:.3f}***".format(r)
    elif p<=0.01:
        r2 = "{:.3f}**".format(r)
    elif p<=0.05:
        r2 = "{:.3f}*".format(r)
    else:
        r2 = "{:.3f}".format(r)
    return r2

def table1_centrality_correlation_table(dg,node_group,groups,fig_name):
    indegs = nx.in_degree_centrality(dg)
    outdegs = nx.out_degree_centrality(dg)
    betw = nx.betweenness_centrality(dg, weight="distance")
    clos = nx.closeness_centrality(dg, distance="distance")
    prs = nx.pagerank_numpy(dg, alpha=.85, weight=None)
    
    node_incoming = dg.in_degree(weight="mobility_number")
    node_outgoing = dg.out_degree(weight="mobility_number")
    # print(sum([ v for k,v in node_incoming ]))
    # print(sum([ v for k,v in node_outgoing ]))


    node_mobility = []

    for n,d in dg.nodes(data=True):
        ng = node_group[n]
        val = node_incoming[n]
        val2 = node_outgoing[n]
        node_mobility.append({ "node":n, "group":ng, "location_type":d["location_type"], "mobility_number":val, "mobility_leaving":val2,
                               "indegree":indegs[n], "outdegree":outdegs[n], "betweenness":betw[n], "closeness":clos[n], "pagerank":prs[n]})
        #print(d)
    node_mob_df = pd.DataFrame.from_dict(node_mobility)
    
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']
    correlate_list = []

    metric_names = ["In-degree", "Closeness", "Betweenness", "PageRank"]#, "Out-degree"
    cols = ["indegree", "closeness", "betweenness", "pagerank"]#, "outdegree"
    res = {"group":"All nodes",}
    temp = node_mob_df[node_mob_df["mobility_number"]>0]
    for c, cname in zip(cols, metric_names):
        x = temp["mobility_number"]
        y = temp[c]
        r,p = stats.pearsonr(x, y) # np.corrcoef(x, y)[0][1]
        r2 = check_sig(r, p)
        r3 = { cname:r2 }
        res.update(r3)
    correlate_list.append(res)

    for g in group_levels:
        gg = g.replace("_"," ").replace("L", "Level ").capitalize()
        ns = groups[g]
        kg = dg.subgraph(ns)
        
        k_indegs = nx.in_degree_centrality(kg)
        k_outdegs = nx.out_degree_centrality(kg)
        k_clos = nx.closeness_centrality(kg, distance="distance")
        k_betw = nx.betweenness_centrality(kg, weight="distance")
        k_prs = nx.pagerank_numpy(kg, alpha=.85, weight=None)
        k_incoming = kg.in_degree(weight="mobility_number")
        k_outgoing = kg.out_degree(weight="mobility_number")
        
        kres = [k_indegs, k_clos, k_betw, k_prs]
        
        
        #ns = list(kg.nodes())
        res = {"group":gg,}
        for cres, cname in zip(kres, metric_names):
            x = [ k_incoming[n] for n in ns ]
            y = [ cres[n] for n in ns ]
            
            x2 = [ xx for xx,yy in zip(x,y) if xx>0 ]
            y2 = [ yy for xx,yy in zip(x,y) if xx>0 ]
            
            r,p = stats.pearsonr(x2, y2)
            r2 = check_sig(r, p)
            r3 = { cname:r2 }
            res.update(r3)
        correlate_list.append(res)
    correlate_table = pd.DataFrame.from_dict(correlate_list)
    correlate_table = correlate_table.set_index("group")
    correlate_table.to_csv('figs_202206/table1'+fig_name+'.csv', index_label="ind")

    print(correlate_table.to_markdown())#tablefmt="grid"))


def table2_internal_external_flow(dg,groups,fig_name):
    floor_order = [ 'Carparks', 'L1', 'L2', 'L6', 'Rooftop garden', 'Residential' ]
    floor_order2 = [ 'Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential' ]
    cat_order = ['Entrance', 'Social', 'Commercial', 'Community', 'Garden', 'Residential',  'Vertical', 'Corridor']
    
    dg_floor = make_dg_floor(dg, groups)
    dg_cat = make_dg_cat(dg)
    
    mat_floor = nx.to_numpy_matrix(dg_floor, weight="mobility_number", nodelist=floor_order)
    # print(mat_floor)
    diag_floor = np.copy(np.diag(mat_floor))
    np.fill_diagonal(mat_floor, 0.)
    # print(mat_floor, diag_floor)

    mat_cat = nx.to_numpy_matrix(dg_cat, weight="mobility_number", nodelist=cat_order)
    # print(mat_cat)
    diag_cat = np.copy(np.diag(mat_cat))
    np.fill_diagonal(mat_cat, 0.)
    # print(mat_cat, diag_cat)
    
    within_floor = diag_floor.tolist()
    between_floor = mat_floor.sum(axis=1).T.tolist()[0]
    dic_prop_floor = {}
    for l,w,b in zip(floor_order2, within_floor, between_floor):
        t = w+b
        dic_prop_floor[l] = { "Total":int(t), "Internal":"{:.2f}".format(w/t), "External":"{:.2f}".format(b/t) }
    df_prop_floor = pd.DataFrame.from_dict(dic_prop_floor, orient="index").reset_index().rename(columns={"index":"Floor/Category"})
    df_prop_floor["Group"] = "Floor"
    #df_prop_floor.append(pd.Series(), ignore_index=True)
    #df_prop_floor.append(pd.Series(), ignore_index=True)
    
    within_cat = diag_cat.tolist()
    between_cat = mat_cat.sum(axis=1).T.tolist()[0]
    dic_prop_cat = {}
    for l,w,b in zip(cat_order, within_cat, between_cat):
        t = w+b
        dic_prop_cat[l] = { "Total":int(t), "Internal":"{:.2f}".format(w/t), "External":"{:.2f}".format(b/t) }
    df_prop_cat = pd.DataFrame.from_dict(dic_prop_cat, orient="index").reset_index().rename(columns={"index":"Floor/Category"})
    df_prop_cat["Group"] = "Category"
    
    #df_prop = df_prop_floor.join(df_prop_cat, how="outer")
    df_prop = df_prop_floor.append(df_prop_cat)
    df_prop = df_prop[["Group", "Floor/Category", "Total", "Internal", "External"]]
    df_prop = df_prop.set_index("Group", "Floor/Category")
    df_prop.to_csv('figs_202206/table2'+fig_name+'.csv', index_label="ind")

    print(df_prop.to_markdown())

def make_dg_floor_OD(G,groups,df_mobility):
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']
    groups2 = { k:groups[k] for k in group_levels }
    temp = []
    oth = [ k for k in groups.keys() if not(k in group_levels) ]
    for k in oth:
        ns = groups[k]
        temp.extend(ns)
    #print(temp)
    groups2["other"] = temp
    fg=[]
    for key,value in groups2.items():
        for v in value:
            fg.append([key,v])

    df_fg = pd.DataFrame.from_records(fg,columns=['fg','zone'])
    
    
    # Add floor group as node attribute to existing Graph
    print('Adding node attributes....')
    i=0
    node_attr_dic={}
    for node in G.nodes():
        node_filtered=df_fg[df_fg['zone']==node]       
        if len(node_filtered)==0:
            floor_group=''
        else:
            floor_group=list(node_filtered['fg'])[0]
        #add it to dic
        if not node in node_attr_dic:
            node_attr_dic[node]={}
        node_attr_dic[node]={'floor_group':floor_group}         
    #use the created dictionary to assign node attributes to graph
    nx.set_node_attributes(G, node_attr_dic) 
    
    # Create graph for floor group
    floor_lab2 = {
        "carparks":"Carparks", 
        "L1":"L1",
        "L2":"L2",
        "L6":"L6",
        "rooftop_garden":"Rooftop garden",
        "residential":"Residential",
    }
    G_od_fg=nx.DiGraph()
    fg_node=nx.get_node_attributes(G,'floor_group')
    for index,row in df_mobility.iterrows():
        if row['yes_no']=='yes':
            i1=row['source']
            i2=row['target']

            fg_i1=fg_node[i1]
            fg_i2=fg_node[i2]
            fg_i1=floor_lab2[fg_i1]
            fg_i2=floor_lab2[fg_i2]

            if not G_od_fg.has_edge(fg_i1, fg_i2):
                G_od_fg.add_edge(fg_i1, fg_i2,mobility_number=1)
            else:
                G_od_fg[fg_i1][fg_i2]["mobility_number"]+=1  
                
    # for u,v,d in G_od_fg.edges(data=True):
    #     print(u,v,d) 
    return G_od_fg

def make_dg_cat_OD(G,df_mobility,fig_name):
    lt_lab2 = {
     'Community_Street':'Community',
     'Garden_Street':'Garden',
     'Residential_Street':'Residential',
     'Commercial_Street':'Commercial',
     'Social_Space':'Social',
     'Corridor':'Corridor',
     'Vertical_Street':'Vertical',
     'Entrance_Street':'Entrance'}
    G_od_lt=nx.DiGraph()
    lt_node=nx.get_node_attributes(G,'location_type')
    # self_loop={}
    for index,row in df_mobility.iterrows():
        if row['yes_no']=='yes':
            i1=row['source']
            i2=row['target']

            lt_i1=lt_node[i1]
            lt_i2=lt_node[i2]
            lt_i1=lt_lab2[lt_i1]
            lt_i2=lt_lab2[lt_i2]
            # if lt_i1==lt_i2:
            #     if not lt_i1 in self_loop:
            #         self_loop[lt_i1]=0
            #     self_loop[lt_i1]+=1
            #     continue
            if not G_od_lt.has_edge(lt_i1, lt_i2):
                G_od_lt.add_edge(lt_i1, lt_i2,mobility_number=1)
            else:
                G_od_lt[lt_i1][lt_i2]["mobility_number"]+=1  
                
    # for u,v,d in G_od_lt.edges(data=True):
    #     print(u,v,d)    
    G_od_lt.add_node('Residential')
    
    if fig_name=='_weekend':
        G_od_lt.add_node('Corridor') 
    return G_od_lt

def draw_fig5_OD_group_network(dg, groups,fig_name,df_mobility):
    G_od_lt=make_dg_cat_OD(dg,df_mobility,fig_name)
    G_od_fg=make_dg_floor_OD(dg,groups,df_mobility)
    
    pos_floor_lab = {
     'Carparks': (0.25, 1.08),
     'L1': (0.13, 0.7),
     'L2': (-0.25, 0.26),
     'L6': (-0.12, -0.65),
     'Rooftop garden': (-0.3, -1.1),
     'Residential': (0.28, -0.3)}
    pos_cat_lab = {
     'Community': [-0.63, 0.24],
     'Garden': [-0.64, -0.35],
     'Residential': [0.28, -1.0],
     'Commercial': [0.17, 0.68],
     'Social': [0.34, -0.32],
     'Corridor': [-0.4, 0.6],
     'Vertical': [-0.08, -0.28],
     'Entrance': [0.63, 0.48]}
    
    fg_node=nx.get_node_attributes(G_od_fg,'internal_mobility_number')
    clrs = sns.color_palette("tab10")
    group_levels=['Residential', 'L1', 'Rooftop garden', 'L2', 'Carparks', 'L6' ]
    nsize_floor = []
    for n in group_levels:
        if G_od_fg.has_edge(n,n):
            edw = G_od_fg[n][n]["mobility_number"]
        else:
            edw = 1
        edw = edw*10
        nsize_floor.append(edw)
        # print(n, edw)
    n_clrs_level = [ clrs[i] for i in range(len(group_levels)) ]


    with open('pos_files/pos_floor_group.json', 'r') as fp:
        pos_floor = json.load(fp)

      
    # pos_floor['Rooftop Garden']=[0.1, -0.5]
    # pos_floor['Carparks']=[0,1]
    floor_lab2 = {
        "Carparks":"Car-parks", 
        "L1":"Level-1",
        "L2":"Level-2",
        "L6":"Level-6",
        "Rooftop garden":"Rooftop garden",
        "Residential":"Residential",
    }


    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    ax1,ax2 = axs
    nx.draw_networkx_nodes(G_od_fg, nodelist=group_levels, pos=pos_floor, ax=ax1, node_size=nsize_floor, node_color=n_clrs_level)
    edgelist = G_od_fg.edges()
    edgelist = [(u, v) for u, v in edgelist if not(u==v)]

    edgeweight = [ G_od_fg[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
    # G_od_fg.remove_edges_from(nx.selfloop_edges(G_od_fg))
    nx.draw_networkx_edges(G_od_fg, pos=pos_floor, ax=ax1, edgelist=edgelist, width=edgewidth, 
                           connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
    nx.draw_networkx_labels(G_od_fg, pos=pos_floor_lab, labels=floor_lab2, ax=ax1)
    ax1.set_title("(a) Between floor groups", loc="left")

    #add histogram
    ax1 = ax1.inset_axes([0.8, 0.15, 0.15, 0.15])
    bins=int(max(edgeweight)/2.0)+1
    ax1.hist(edgeweight,bins=bins,color='black')
    #add legend
    lines = []
    edges_weight_list = [min(edgewidth),np.median(edgewidth),max(edgewidth)]
    texts_edges_weight_list = [min(edgeweight),np.median(edgeweight),max(edgeweight)]
    for i, width in enumerate(edges_weight_list):
        lines.append(Line2D([],[], linewidth=width,alpha=.6, color="xkcd:navy"))
    ax1.legend(lines, texts_edges_weight_list, bbox_to_anchor=(-4.2, 5.7), frameon=False) 
    #-----------------------------------------------------
    #function group
    #lt_node=nx.get_node_attributes(G_od_lt,'internal_mobility_number')
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ] 
    nsize_cat = []
    for n in loc_listb2:
        if G_od_lt.has_edge(n,n):
            edw = G_od_lt[n][n]["mobility_number"]
        else:
            edw = 1
        edw = edw*10
        nsize_cat.append(edw)
        # print(n, edw)


    with open('pos_files/pos_cat.json', 'r') as fp:
        pos_cat = json.load(fp)  
        
    nx.draw_networkx_nodes(G_od_lt, nodelist=loc_listb2, pos=pos_cat, ax=ax2, node_size=nsize_cat, node_color=n_clrs_cat)
    edgelist = G_od_lt.edges()
    edgelist = [(u, v) for u, v in edgelist if not(u==v)]

    edgeweight = [ G_od_lt[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
    # G_od_lt.remove_edges_from(nx.selfloop_edges(G_od_lt))
    nx.draw_networkx_edges(G_od_lt, pos=pos_cat, ax=ax2, edgelist=edgelist, width=edgewidth, 
                            connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
    nx.draw_networkx_labels(G_od_lt, pos=pos_cat_lab, ax=ax2)
    ax2.set_title("(b) Between location categories", loc="left")

    #add histogram
    ax2 = ax2.inset_axes([0.8, 0.15, 0.15, 0.15])
    bins=int(max(edgeweight)/2.0)+1
    ax2.hist(edgeweight,bins=bins,color='black')
    #add legend
    lines = []
    edges_weight_list = [min(edgewidth),np.median(edgewidth),max(edgewidth)]
    texts_edges_weight_list = [min(edgeweight),np.median(edgeweight),max(edgeweight)]
    for i, width in enumerate(edges_weight_list):
        lines.append(Line2D([],[], linewidth=width,alpha=.6, color="xkcd:navy"))
    ax2.legend(lines, texts_edges_weight_list, bbox_to_anchor=(-4.2, 5.7), frameon=False) 


    plt.tight_layout()
    plt.savefig(os.path.join("figs_202206", "fig5"+fig_name+".png"), dpi=300, bbox_inches="tight")

def draw_fig6_OD_matrix(dg,groups,df_mobility,fig_name):
    G_od_lt=make_dg_cat_OD(dg,df_mobility,fig_name)
    G_od_fg=make_dg_floor_OD(dg,groups,df_mobility)
    
    floor_order = [ 'Carparks', 'L1', 'L2', 'L6', 'Rooftop garden', 'Residential' ]
    floor_order2 = [ 'Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential' ]
    cat_order = ['Entrance', 'Social', 'Commercial', 'Community', 'Garden', 'Residential',  'Vertical', 'Corridor']

    mat_floor = nx.to_numpy_matrix(G_od_fg, weight="mobility_number", nodelist=floor_order)
    # print(mat_floor)
    diag_floor = np.copy(np.diag(mat_floor))
    np.fill_diagonal(mat_floor, 0.)
    # print(mat_floor, diag_floor)

    mat_cat = nx.to_numpy_matrix(G_od_lt, weight="mobility_number", nodelist=cat_order)
    # print(mat_cat)
    diag_cat = np.copy(np.diag(mat_cat))
    np.fill_diagonal(mat_cat, 0.)
    # print(mat_cat, diag_cat)

    #mat_floor, mat_floor.sum(axis=1)
    rowsum = mat_floor.sum(axis=1)
    mat_floor2 = np.matrix(np.zeros(mat_floor.shape))
    for i in range(mat_floor.shape[0]):
        for j in range(mat_floor.shape[1]):
            v = mat_floor[i,j]
            b = rowsum[i]
            #print(v,b)
            mat_floor2[i,j] = v/b
    #mat_floor, mat_floor2

    rowsum = mat_cat.sum(axis=1)
    mat_cat2 = np.matrix(np.zeros(mat_cat.shape))
    for i in range(mat_cat.shape[0]):
        for j in range(mat_cat.shape[1]):
            v = mat_cat[i,j]
            b = rowsum[i]
            #print(v,b)
            mat_cat2[i,j] = v/b
            
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white")
    ax1, ax2 = axs
    ax1.imshow(mat_floor2, cmap="Blues", vmin=0, vmax=1)
    ax2.imshow(mat_cat2, cmap="Blues", vmin=0, vmax=1)

    ticklabels = [floor_order2, cat_order]
    diags = [diag_floor, diag_cat]
    mats = [mat_floor, mat_cat]
    mats2 = [mat_floor2, mat_cat2]

    ax1.set_title("(a) Between floor groups", loc="left")
    ax2.set_title("(b) Between location categories", loc="left")

    for i, ax in enumerate(axs):
        tlab = ticklabels[i]
        ax.set_xticks(list(range(len(tlab))))
        ax.set_yticks(list(range(len(tlab))))
        ax.set_xticklabels(tlab, rotation=35, ha="right")
        ax.set_yticklabels(tlab)
        diag = diags[i]
        for j,v in enumerate(diag):
            ax.text(j,j,int(v), ha="center", va='center', zorder=5, c='k', style='normal', fontweight='bold')
        
        mat = mats2[i]
        for a in range(len(diag)):
            for b in range(len(diag)):
                if a==b:continue
                v = mat[a,b]
                if v==0:continue
                fc = 'k' if round(v,2)<0.65 else 'w'
                ax.text(b,a,"{:.2f}".format(v), ha="center", va='center', zorder=5, c=fc, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join("figs_202206", "fig6"+fig_name+".png"), dpi=300, bbox_inches="tight")

def table3_entropy_OD(dg,groups,df_mobility,fig_name):
    G_od_lt=make_dg_cat_OD(dg,df_mobility,fig_name)
    G_od_fg=make_dg_floor_OD(dg,groups,df_mobility)
    
    floor_order = [ 'Carparks', 'L1', 'L2', 'L6', 'Rooftop garden', 'Residential' ]
    cat_order = ['Entrance', 'Social', 'Commercial', 'Community', 'Garden', 'Residential',  'Vertical', 'Corridor']

    mat_floor = nx.to_numpy_matrix(G_od_fg, weight="mobility_number", nodelist=floor_order)
    # print(mat_floor)
    diag_floor = np.copy(np.diag(mat_floor))
    np.fill_diagonal(mat_floor, 0.)
    # print(mat_floor, diag_floor)

    mat_cat = nx.to_numpy_matrix(G_od_lt, weight="mobility_number", nodelist=cat_order)
    # print(mat_cat)
    diag_cat = np.copy(np.diag(mat_cat))
    np.fill_diagonal(mat_cat, 0.)
    # print(mat_cat, diag_cat)
    
    rowsum = [ v[0] for v in mat_floor.sum(axis=1).tolist() ]
    colsum = mat_floor.sum(axis=0).tolist()[0]
    #print(rowsum, colsum)
    mat_floor2 = np.matrix(np.zeros(mat_floor.shape))
    mat_floor3 = np.matrix(np.zeros(mat_floor.shape))
    for i in range(mat_floor.shape[0]):
        for j in range(mat_floor.shape[1]):
            v = mat_floor[i,j]
            b = rowsum[i]
            mat_floor2[i,j] = v/b
            c = colsum[j]
            #print(v,b,c)
            mat_floor3[i,j] = v/c
    """
    made changes -- removed row2 and used row -- takes all connections to calculate entropy
    """
    floor_out_entropy = []
    floor_in_entropy = []
    for i in range(mat_floor2.shape[0]):
        row = mat_floor2[i,:].tolist()[0]
        #row2 = [ r for r in row if r>0 ]
        #print(row2)
        H = -sum([ p*np.log2(p) if p>0 else 0. for p in row ])/np.log2(len(row)) if len(row)>1 else 0
        floor_out_entropy.append(H)
        col = [ v[0] for v in mat_floor3[:,i].tolist() ]
        #col2 = [ r for r in col if r>0 ]
        H2 = -sum([ p*np.log2(p) if p>0 else 0. for p in col ])/np.log2(len(col)) if len(col)>1 else 0
        floor_in_entropy.append(H2)
    # print([ "{:.3f}".format(v) for v in floor_out_entropy ])
    # print([ "{:.3f}".format(v) for v in floor_in_entropy ])

    rowsum = [ v[0] for v in mat_cat.sum(axis=1).tolist() ]
    colsum = mat_cat.sum(axis=0).tolist()[0]
    mat_cat2 = np.matrix(np.zeros(mat_cat.shape))
    mat_cat3 = np.matrix(np.zeros(mat_cat.shape))
    for i in range(mat_cat.shape[0]):
        for j in range(mat_cat.shape[1]):
            v = mat_cat[i,j]
            b = rowsum[i]
            mat_cat2[i,j] = v/b
            c = colsum[j]
            mat_cat3[i,j] = v/c
            
    cat_out_entropy = []
    cat_in_entropy = []
    for i in range(mat_cat2.shape[0]):
        row = mat_cat2[i,:].tolist()[0]
        H = -sum([ p*np.log2(p) if p>0 else 0. for p in row ])/np.log2(len(row))
        H = 0. if H==-0. else H
        cat_out_entropy.append(H)
        col = [ v[0] for v in mat_cat3[:,i].tolist() ]
        H2 = -sum([ p*np.log2(p) if p>0 else 0. for p in col ])/np.log2(len(col))
        H2 = 0. if H2==-0. else H2
        cat_in_entropy.append(H2)
    # print([ "{:.3f}".format(v) for v in cat_out_entropy ])
    # print([ "{:.3f}".format(v) for v in cat_in_entropy ])
    #
    df_entropy_floor = pd.DataFrame.from_dict({"Floor_group":floor_order+["",""],
                                               "floor_out":[ "{:.3f}".format(round(v,3)) for v in floor_out_entropy ]+["",""], 
                                               "floor_in":[ "{:.3f}".format(round(v,3)) for v in floor_in_entropy ]+["",""],
                                              })
    df_entropy_cat = pd.DataFrame.from_dict({"Location_category":cat_order,
                                             "cat_out":[ "{:.3f}".format(round(v,3)) for v in cat_out_entropy ],
                                             "cat_in":[ "{:.3f}".format(round(v,3)) for v in cat_in_entropy ], 
                                            })
    df_entropy_hor = df_entropy_floor.join(df_entropy_cat, )
    print(df_entropy_floor.to_markdown())
    print(df_entropy_cat.to_markdown())
    print(df_entropy_hor.to_markdown())
    df_entropy_hor.to_csv('figs_202206/table3'+fig_name+'.csv', index_label="ind")

    
def table_4_network_basic_info(dg,groups,fig_name):
    data_list = []

    # the whole graph
    n_mob = [ v for k,v in dg.in_degree(weight="mobility_number") ]
    e_mob = [ d["mobility_number"] for u,v,d in dg.edges(data=True) ]
    whole_graph = {"Graph":"All nodes", 
                   "N":dg.number_of_nodes(), 
                   "E":dg.number_of_edges(), 
                   "Total mobility":sum(e_mob), 
                   "Node's mobility range":"{}--{}".format(min(n_mob), max(n_mob)), 
                   "Node's mobility mean (sd)":u"{:.2f} (\u00B1{:.2f})".format(np.mean(n_mob), np.std(n_mob)), 
                   "Edge's mobility range":"{}--{}".format(min(e_mob), max(e_mob)), 
                   "Edge's mobility mean (sd)":u"{:.2f} (\u00B1{:.2f})".format(np.mean(e_mob), np.std(e_mob)), 
                  }
    #print(whole_graph)
    data_list.append(whole_graph)
    
    # the targeted groups
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']
    for i,g in enumerate(group_levels):
        ns = groups[g]
        gg = g if not(g in ["residential", "L6"]) else g+"*"
        if gg[0]=="L": gg = gg.replace("L", "Level ")
        kg = dg.subgraph(ns)
        n_mob = [ v for k,v in kg.in_degree(weight="mobility_number") ]
        e_mob = [ d["mobility_number"] for u,v,d in kg.edges(data=True) ]
        sub_graph = {"Graph":gg.capitalize().replace("_"," "), 
                   "N":kg.number_of_nodes(), 
                   "E":kg.number_of_edges(), 
                   "Total mobility":sum(e_mob), 
                   "Node's mobility range":"{}--{}".format(min(n_mob), max(n_mob)), 
                   "Node's mobility mean (sd)":u"{:.2f} (\u00B1{:.2f})".format(np.mean(n_mob), np.std(n_mob)), 
                   "Edge's mobility range":"{}--{}".format(min(e_mob), max(e_mob)), 
                   "Edge's mobility mean (sd)":u"{:.2f} (\u00B1{:.2f})".format(np.mean(e_mob), np.std(e_mob)), 
                  }
        data_list.append(sub_graph)
        
    # other
    ns1 = groups["L3-L4"]
    ns2 = groups["L7"]
    ns = ns1+ns2
    kg = dg.subgraph(ns)
    n_mob = [ v for k,v in kg.in_degree(weight="mobility_number") ]
    e_mob = [ d["mobility_number"] for u,v,d in kg.edges(data=True) ]
    sub_graph = {"Graph":"Other", 
               "N":kg.number_of_nodes(), 
               "E":kg.number_of_edges(), 
               "Total mobility":sum(e_mob), 
               "Node's mobility range":"{} -- {}".format(min(n_mob), max(n_mob)), 
               "Node's mobility mean (sd)":u"{:.2f} (\u00B1{:.2f})".format(np.mean(n_mob), np.std(n_mob)), 
               "Edge's mobility range":"{} -- {}".format(min(e_mob), max(e_mob)), 
               "Edge's mobility mean (sd)":u"{:.2f} (\u00B1{:.2f})".format(np.mean(e_mob), np.std(e_mob)), 
              }
    data_list.append(sub_graph)
    
    # generate dataframe table
    data_table = pd.DataFrame.from_dict(data_list)
    data_table = data_table.set_index("Graph")
    data_table.to_csv('figs_202206/table4'+fig_name+'.csv', index_label="ind")

    print(data_table.to_markdown())
    
def main():
    fig_name='_1dayeach'
    # dg = nx.read_gexf("../data/ka_network_graph_a_17mar2021_with_mobility.gexf")
    dg = nx.read_gexf("../data_2/ka_network_graph_a_17mar2021_with_mobility"+fig_name+".gexf")

    print('Number of nodes: ',len(dg.nodes()))
    print('Number of edges: ',len(dg.edges()))
    dg, pos, node_group, groups = get_data(dg)
    clrs, n_clrsb, group_colors = get_node_colors(dg, node_group)
    draw_fig1_barh(dg, groups, clrs,fig_name)
    draw_fig2_network(dg, pos, clrs, n_clrsb, group_colors,fig_name)
    draw_fig3_networks(dg, groups,fig_name)
    draw_fig4_cat_network(dg, groups,fig_name)
    
    table1_centrality_correlation_table(dg,node_group,groups,fig_name)
    table2_internal_external_flow(dg,groups,fig_name)
    
    #----------------------------
    #origin-destination analysis
    #----------------------------
    # df_mobility = pd.read_csv('../data/trip_data_ka_dataset_aj_18mar2021.csv')
    df_mobility = pd.read_csv('../data_2/trip_data_ka_dataset_aj_18mar2021'+fig_name+'.csv')

    draw_fig5_OD_group_network(dg, groups,fig_name,df_mobility)
    draw_fig6_OD_matrix(dg,groups,df_mobility,fig_name)
    table3_entropy_OD(dg,groups,df_mobility,fig_name)
    
    table_4_network_basic_info(dg,groups,fig_name)
if __name__ == '__main__':
    main()
