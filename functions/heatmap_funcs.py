import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# def make_annot_df(main_df, desired_cols, map_dict):

#     for col in desired_cols:
#         main_df[col] = main_df[col].map(map_dict)

#     return


def make_heatmap(heatmap_df,title, output_fp, COLOR, DESIRED_INDEX, annot_df=None):

    heatmap_df = heatmap_df.sort_values(DESIRED_INDEX).reset_index(drop=True)

    #set index
    heatmap_df = heatmap_df.set_index(DESIRED_INDEX)

    try:
        if annot_df == None:
            annot_df = heatmap_df.copy()
    except ValueError:
        annot_df = annot_df.sort_values(DESIRED_INDEX).reset_index(drop=True)
        annot_df = annot_df.set_index(DESIRED_INDEX)




    print(annot_df)

    # #remove col in annot df if need
    # annot_cols = list(annot_df)
    # if DESIRED_INDEX in annot_cols:
    #     annot_df.remove(DESIRED_INDEX)

    #make fig
    plt.figure(figsize=(12, 9))
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_df, annot = annot_df, fmt = '', cmap=COLOR,linewidths=0.1, linecolor='black', cbar=True)

    #ax = sns.heatmap(annot_df, annot = heatmap_df, fmt = '', cmap=COLOR,linewidths=0.1, linecolor='black', cbar=True)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)


    for _, spine in ax.spines.items():
        spine.set_visible(True)

    ax.set_title(title)

    fig.savefig(output_fp ,bbox_inches='tight')
    print(output_fp)
    plt.clf()


    return


def get_corr_class_df(main_corr_df, corr_cols,SIG_LEVEL=0.3):

    CLASS_DICT = {
    (-1, -0.7 ): 4,
    (-0.7, -0.5 ): 3,
    (-0.5, -SIG_LEVEL) : 2,
    (-SIG_LEVEL, 0) : 1,
    (0, SIG_LEVEL) : 1,
    (SIG_LEVEL, 0.5) : 2,
    (0.5, 0.7) : 3,
    (0.7, 1) : 4}

    new_corr_df = main_corr_df.copy()

    for corr_col in corr_cols:
        curr_corr_series = new_corr_df[corr_col]

        new_corr_series = []
        for corr in curr_corr_series:
            for corr_tuple,corr_class in CLASS_DICT.items():
                lower = corr_tuple[0]
                upper = corr_tuple[1]

                if (lower <= corr) and (corr < upper):
                    new_corr_series.append(corr_class)
                    break

        new_corr_df[corr_col] = new_corr_series

    print()
    print(new_corr_df)

    return new_corr_df

def get_corr_class_df_v2(main_corr_df, corr_cols):

    CLASS_DICT = {
    (0, 0.25) : 1,
    (0.25, 0.3) : 2,
    (0.3, 0.5) : 3,
    (0.5, 0.7) : 4,
    (0.7, 1) : 5}

    new_corr_df = main_corr_df.copy()

    for corr_col in corr_cols:
        curr_corr_series = new_corr_df[corr_col]

        new_corr_series = []
        for corr in curr_corr_series:

            corr = abs(corr)
            for corr_tuple,corr_class in CLASS_DICT.items():
                lower = corr_tuple[0]
                upper = corr_tuple[1]

                if (lower <= corr) and (corr < upper):
                    new_corr_series.append(corr_class)
                    break

        new_corr_df[corr_col] = new_corr_series

    print()
    print(new_corr_df)

    return new_corr_df



def make_heatmap_v2_with_colors(heatmap_df,title, output_fp, cmap, DESIRED_INDEX, annot_df=None):

    heatmap_df = heatmap_df.sort_values(DESIRED_INDEX).reset_index(drop=True)

    #set index
    heatmap_df = heatmap_df.set_index(DESIRED_INDEX)

    try:
        if annot_df == None:
            annot_df = heatmap_df.copy()
    except ValueError:
        annot_df = annot_df.sort_values(DESIRED_INDEX).reset_index(drop=True)
        annot_df = annot_df.set_index(DESIRED_INDEX)




    print(annot_df)

    # #remove col in annot df if need
    # annot_cols = list(annot_df)
    # if DESIRED_INDEX in annot_cols:
    #     annot_df.remove(DESIRED_INDEX)

    #make fig
    plt.figure(figsize=(12, 9))
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap_df, annot = annot_df, fmt = '', cmap=cmap,linewidths=0.1, linecolor='black', cbar=True)

    #ax = sns.heatmap(annot_df, annot = heatmap_df, fmt = '', cmap=COLOR,linewidths=0.1, linecolor='black', cbar=True)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)


    for _, spine in ax.spines.items():
        spine.set_visible(True)

    ax.set_title(title)

    fig.savefig(output_fp ,bbox_inches='tight')
    print(output_fp)
    plt.clf()


    return
