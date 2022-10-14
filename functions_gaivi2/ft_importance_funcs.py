import pandas as pd 
import os,sys
import numpy as np 

from basic_utils import create_output_dir

def create_ft_importances_without_timestep_info(ft_rank_df,ignore_fts):

    features = list(ft_rank_df["feature"])
    new_fts = []
    for ft in features:
        if ft in ignore_fts:
            new_fts.append(ft)
        else:
            ft_str_list = ft.split("_")
            ft_str_list = ft_str_list[:-1]
            new_ft = "_".join(ft_str_list)
            new_fts.append(new_ft)
            if new_ft == "":
                print("Blank ft: %s"%ft)
                # sys.exit(0)
    ft_rank_df["feature"] = new_fts
    ft_rank_df["importance"] = ft_rank_df.groupby(["feature"])["importance"].transform("sum")
    # ft_rank_df["importance"] = (ft_rank_df["importance"] - ft_rank_df["importance"].min())/(ft_rank_df["importance"].max() - ft_rank_df["importance"].min())
    # ft_rank_df["importance"] = ft_rank_df["importance"]
    ft_rank_df = ft_rank_df.drop_duplicates().reset_index(drop=True)
    ft_rank_df["importance"] =ft_rank_df["importance"]/ft_rank_df["importance"].sum()
    ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return ft_rank_df

def save_ft_info_v2_simpler(output_dir, model,ignore_fts,flattened_fts,tag):

    ft_importances = model.feature_importances_
    num_imp = len(ft_importances)
    num_fts = len(flattened_fts)
    print("\nnum_imp: %d"%num_imp)
    print("\nnum_fts: %d"%num_fts)
    ft_rank_df = pd.DataFrame(data={"feature":flattened_fts, "importance":ft_importances})
    ft_rank_df  = ft_rank_df[["feature", "importance"]]
    ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nft_rank_df")
    print(ft_rank_df)

    ft_dir = output_dir + "Feature-Importances/%s/"%tag
    create_output_dir(ft_dir)
    raw_ft_fp = ft_dir + "Raw-Feature-Ranks-with-Timestep-Info.csv"
    ft_rank_df.to_csv(raw_ft_fp, index=False)
    print(raw_ft_fp)

    print("\nGet ft importances without timestep info...")
    ft_rank_df = create_ft_importances_without_timestep_info(ft_rank_df,ignore_fts)
    print("\nft_rank_df without timestep info")
    print(ft_rank_df)

    ft_fp = ft_dir + "Feature-Ranks-Without-Timestep-Info.csv"
    ft_rank_df.to_csv(ft_fp, index=False)
    print(ft_fp)