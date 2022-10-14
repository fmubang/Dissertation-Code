import pandas as pd
import os,sys
import numpy as np
from basic_utils import create_output_dir,hyphenate_infoID_dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from textwrap  import wrap
import matplotlib.dates as mdates

def get_cur_comp_df_win_info(cur_comp_df, metric, model_tag, baseline_tag):
    model_errors = list(cur_comp_df[model_tag + "_" + metric])
    baseline_errors = list(cur_comp_df[baseline_tag + "_" + metric])
    cur_comp_df["is_winner"]=[1 if model_error < baseline_error else 0 if baseline_error<model_error else "tie" for model_error, baseline_error in zip(model_errors, baseline_errors)]
    print(cur_comp_df)
    return cur_comp_df

def agg_timesteps_to_higher_gran(agg_timestep_amount ,pred_df, pred_col, acutal_col,groupby_cols, timestep_col="timestep" ):

    timestep_group = []
    num_unique_timesteps = pred_df[timestep_col].nunique()
    print("\nnum_unique_timesteps")
    print(num_unique_timesteps)
    cur_group_num = 1
    timestep_to_group_num_dict = {}
    for t in range(1, num_unique_timesteps+1):
        timestep_to_group_num_dict[t] = cur_group_num

        if t%agg_timestep_amount == 0:
            cur_group_num+=1
    pred_df["timestep_group"] = pred_df[timestep_col].map(timestep_to_group_num_dict)

    print(pred_df)

    agg_cols = [pred_col, acutal_col]
    for agg_col in agg_cols:
        pred_df[agg_col] = pred_df.groupby(["timestep_group"]+list(groupby_cols))[agg_col].transform("sum")
        pred_df[agg_col] = np.round(pred_df[agg_col], 0)

    pred_df = pred_df.drop(timestep_col, axis=1)
    pred_df[timestep_col] = pred_df["timestep_group"]
    pred_df = pred_df.drop_duplicates().reset_index(drop=True)



    return pred_df

def get_summary_info_of_baseline_comp_df(comp_df, proper_baseline_dir, y_eval_tag, metric_tag, model_tag, baseline_tag):

    model_error_tag = model_tag + "_" + metric_tag
    baseline_error_tag = baseline_tag + "_" + metric_tag

    mean_model_error_tag =  "mean_" + model_tag + "_" + metric_tag
    mean_baseline_error_tag ="mean_" + baseline_tag + "_" + metric_tag

    nunique_timesteps = comp_df["timestep"].nunique()

    #get infoID view
    infoID_comp_df = comp_df.copy()
    infoID_comp_df[mean_model_error_tag] = infoID_comp_df.groupby(["infoID"])[model_error_tag].transform("mean")
    infoID_comp_df[mean_baseline_error_tag] = infoID_comp_df.groupby(["infoID"])[baseline_error_tag].transform("mean")
    infoID_comp_df["num_trials"] = infoID_comp_df.groupby(["infoID"])["is_winner"].transform("count")
    infoID_comp_df = infoID_comp_df[infoID_comp_df["is_winner"]!="tie"]
    infoID_comp_df["num_trials_no_ties"] = infoID_comp_df.groupby(["infoID"])["is_winner"].transform("count")
    infoID_comp_df["num_infoID_wins"] = infoID_comp_df.groupby(["infoID"])["is_winner"].transform("sum")
    infoID_comp_df["win_freq"] = infoID_comp_df["num_infoID_wins"]/(1.0 * infoID_comp_df["num_trials_no_ties"])
    infoID_keep_cols = ["infoID", mean_model_error_tag,mean_baseline_error_tag,"num_infoID_wins","num_trials" ,"num_trials_no_ties","win_freq"]
    infoID_comp_df= infoID_comp_df[infoID_keep_cols]
    infoID_comp_df = infoID_comp_df.drop_duplicates().reset_index(drop=True)

    print("\ninfoID_comp_df")
    print(infoID_comp_df)

    output_fp = proper_baseline_dir + y_eval_tag + "-" + "InfoID-Summary-Comparisons.csv"
    infoID_comp_df.to_csv(output_fp,index=False)
    print(output_fp)

    #get infoID view
    target_ft_comp_df = comp_df.copy()
    target_ft_comp_df[mean_model_error_tag] = target_ft_comp_df.groupby(["target_ft"])[model_error_tag].transform("mean")
    target_ft_comp_df[mean_baseline_error_tag] = target_ft_comp_df.groupby(["target_ft"])[baseline_error_tag].transform("mean")
    target_ft_comp_df["num_trials"] = target_ft_comp_df.groupby(["target_ft"])["is_winner"].transform("count")
    target_ft_comp_df = target_ft_comp_df[target_ft_comp_df["is_winner"]!="tie"]
    target_ft_comp_df["num_trials_no_ties"] = target_ft_comp_df.groupby(["target_ft"])["is_winner"].transform("count")
    target_ft_comp_df["num_target_ft_wins"] = target_ft_comp_df.groupby(["target_ft"])["is_winner"].transform("sum")
    target_ft_comp_df["win_freq"] = target_ft_comp_df["num_target_ft_wins"]/(1.0 * target_ft_comp_df["num_trials_no_ties"])
    target_ft_keep_cols = ["target_ft", mean_model_error_tag,mean_baseline_error_tag,"num_target_ft_wins","num_trials" ,"num_trials_no_ties","win_freq"]
    target_ft_comp_df= target_ft_comp_df[target_ft_keep_cols]
    target_ft_comp_df = target_ft_comp_df.drop_duplicates().reset_index(drop=True)

    print("\ntarget_ft_comp_df")
    print(target_ft_comp_df)

    output_fp = proper_baseline_dir + y_eval_tag + "-" + "target_ft-Summary-Comparisons.csv"
    target_ft_comp_df.to_csv(output_fp,index=False)
    print(output_fp)

    overall_summary_dict = {}
    overall_summary_dict["target_ft_" + mean_model_error_tag] = [target_ft_comp_df[mean_model_error_tag].mean()]
    overall_summary_dict["infoID_" + mean_model_error_tag] = [infoID_comp_df[mean_model_error_tag].mean()]
    overall_summary_dict["target_ft_" + mean_baseline_error_tag] = [target_ft_comp_df[mean_baseline_error_tag].mean()]
    overall_summary_dict["infoID_" + mean_baseline_error_tag] = [infoID_comp_df[mean_baseline_error_tag].mean()]

    overall_summary_dict["target_ft_mean_model_win_freq"] = [target_ft_comp_df["win_freq"].mean()]
    overall_summary_dict["infoID_mean_model_win_freq"] = [infoID_comp_df["win_freq"].mean()]

    col_order = [
    "target_ft_" + mean_model_error_tag,
    "infoID_" + mean_model_error_tag,
    "target_ft_" + mean_baseline_error_tag,
    "infoID_" + mean_baseline_error_tag,
    "target_ft_mean_model_win_freq",
    "infoID_mean_model_win_freq"
    ]

    overall_summary_df = pd.DataFrame(data=overall_summary_dict)
    overall_summary_df=overall_summary_df[col_order]

    print("\noverall_summary_df")
    print(overall_summary_df)

    output_fp = proper_baseline_dir + y_eval_tag + "-Overall-Summary.csv"
    overall_summary_df.to_csv(output_fp, index=False)
    print(output_fp)


    return infoID_comp_df, target_ft_comp_df,overall_summary_df

def graph_pred_vs_gt_with_baseline_from_comp_df(proper_comp_df, model_tag, baseline_tag, metric_tag,platform,graph_output_dir,LOG=False):
    create_output_dir(graph_output_dir)


    model_pred_col = model_tag + "_pred"
    baseline_pred_col = baseline_tag + "_pred"
    infoIDs = list(proper_comp_df["infoID"].unique())
    num_timesteps = proper_comp_df["timestep"].nunique()
    target_fts = list(proper_comp_df["target_ft"].unique())

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        #============ plot stuff ==============

        coord_list = [0, 1, 2]
        # num_target_fts = len(target_fts)
        # coord_list = coord_list[:num_target_fts]
        fig, axs = plt.subplots(3,figsize=(9,9))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

        for coord,target_ft in enumerate(target_fts):
            print("\nPlotting %s %s %s..."%(platform, infoID, target_ft))

            target_ft_to_axis_coordinates_dict[target_ft] = coord



            temp_df = proper_comp_df[(proper_comp_df["infoID"]==infoID) & (proper_comp_df["platform"]==platform) & (proper_comp_df["target_ft"]==target_ft)]
            temp_df = temp_df.reset_index(drop=True)
            print("\ntemp_df")
            print(temp_df)

            if temp_df.shape[0] != num_timesteps:
                print("\nError! temp_df.shape[0] != num_timesteps")
                print(temp_df.shape[0])
                print(num_timesteps)
                sys.exit(0)
            else:
                print("\nCounts are good...Plotting...")

            #======================= get series =======================
            model_pred_series = temp_df[model_pred_col]
            baseline_pred_series = temp_df[baseline_pred_col]
            gt_series = temp_df["actual"]

            #======================= score stuff =======================
            model_mean_error = temp_df[model_tag + "_" + metric_tag].mean()
            model_wins = temp_df[temp_df["is_winner"]==1].shape[0]
            num_trials_no_ties = temp_df[temp_df["is_winner"]!="tie"].shape[0]
            try:
                model_win_score = model_wins/num_trials_no_ties
            except ZeroDivisionError:
                model_win_score = 0
            model_error_result_tag = "%s %s: %.4f"%(model_tag, metric_tag, model_mean_error)
            # print()
            # print(model_error_result_tag)

            baseline_mean_error = temp_df[baseline_tag + "_" + metric_tag].mean()
            baseline_error_result_tag = "%s %s: %.4f"%(baseline_tag , metric_tag, baseline_mean_error)
            # print(baseline_error_result_tag)

            model_score_tag = "%s win freq: %.4f"%(model_tag,  model_win_score)
            # print(model_score_tag)

            full_report_tag = model_error_result_tag + "\n" + baseline_error_result_tag + "\n" + model_score_tag
            print()
            print(full_report_tag)

            # target_no_underscore = target_ft.replace("_"," ")
            # title_tag = "%s %s"%(infoID, target_no_underscore)
            # title_tag = title_tag + "\n"+ full_report_tag
            # title = ("\n".join(wrap(title_tag, 50)))

            target_no_underscore = target_ft.replace("_"," ")
            infoID_target_tag = "%s %s"%(infoID, target_no_underscore)
            title_tag = infoID_target_tag + "\n"+ full_report_tag
            title = title_tag

            # print("\ncur_y_pred shape")
            # print(cur_y_pred.shape)
            # print("\ncur_y_ground_truth shape")
            # print(cur_y_ground_truth.shape)
            axs[coord].plot(model_pred_series,"-r" ,label=model_tag)
            axs[coord].plot(baseline_pred_series,":g" ,label=baseline_tag)
            axs[coord].plot(gt_series, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')
            ax.legend()

            # if LOG == True:
            #     ax.set_yscale('log')

        # plt.legend()
        plt.tight_layout()
        output_fp = graph_output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)


def create_proper_baseline_comparison_df(baseline_model_metric_df, model_metric_df, rename_cols,comp_merge_cols,proper_baseline_dir,y_eval_tag, metric_tag="rmse"):

    baseline_tag = baseline_model_metric_df["model_tag"].unique()[0]
    model_tag = model_metric_df["model_tag"].unique()[0]

    for col in rename_cols:
        baseline_model_metric_df = baseline_model_metric_df.rename(columns={col: baseline_tag + "_" +col})

    for col in rename_cols:
        model_metric_df = model_metric_df.rename(columns={col: model_tag +"_" +col})

    cur_comp_df = pd.merge(model_metric_df, baseline_model_metric_df, on=comp_merge_cols, how="inner")

    print("\ncur_comp_df")
    print(cur_comp_df)

    cur_comp_df = get_cur_comp_df_win_info(cur_comp_df,metric_tag, model_tag, baseline_tag)

    print("\ncur_comp_df with win info")
    print(cur_comp_df)

    cur_output_dir =proper_baseline_dir + y_eval_tag + "/"
    create_output_dir(cur_output_dir)
    cur_output_fp = cur_output_dir + y_eval_tag + "-" + metric_tag + "-model-baseline-omparisons.csv"
    cur_comp_df.to_csv(cur_output_fp,index=False)
    print(cur_output_fp)

    return cur_comp_df

def insert_metrics_into_pred_df_v2_TEMPORAL(pred_df, infoIDs, platforms, output_dir,target_fts,metrics_to_get,metric_str_to_function_dict):
    #get model tag
    model_tag = pred_df["model_tag"].unique()[0]
    y_tag = pred_df["y_tag"].unique()[0]

    cur_output_dir = output_dir + model_tag + "/"
    create_output_dir(cur_output_dir)

    metric_to_result_list_dict = {}
    for metric in metrics_to_get:
        metric_to_result_list_dict[metric]=[]


    infoID_df_list = []
    target_df_list = []
    metric_df_list = []
    timestep_df_list = []
    pred_df_list = []
    actual_df_list = []

    platform = pred_df["platform"].unique()[0]


    for infoID in infoIDs:
        for target_ft in target_fts:




            temp_df = pred_df[(pred_df["infoID"]==infoID) & (pred_df["target_ft"]==target_ft)]
            print(temp_df)

            cur_timestep_list = [t+1 for t in range(temp_df.shape[0])]

            actuals = list(temp_df["actual"])
            preds = list(temp_df["pred"])

            for metric in metrics_to_get:
                metric_func = metric_str_to_function_dict[metric]

                for actual,pred in zip(actuals,preds):
                    result = metric_func(pd.Series([actual]), pd.Series([pred]))
                    metric_to_result_list_dict[metric].append(result)

            for i in range(temp_df.shape[0]):
                infoID_df_list.append(infoID)
                target_df_list.append(target_ft)

            timestep_df_list = timestep_df_list + list(cur_timestep_list)

            pred_df_list = pred_df_list + list(preds)
            actual_df_list = actual_df_list + list(actuals)

    df = pd.DataFrame(data={"infoID":infoID_df_list, "target_ft":target_df_list})
    for metric,metric_list in metric_to_result_list_dict.items():
        df[metric] = metric_list

    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    df["timestep"]=timestep_df_list
    df["pred"] = pred_df_list
    df["actual"] = actual_df_list
    df["platform"]=platform


    print(df)

    output_fp = cur_output_dir + y_tag + "-" + model_tag + "-metrics.csv"
    df.to_csv(output_fp, index=False)
    print(output_fp)

    # sys.exit(0)

    return df


def graph_pred_vs_gt_options_v2_from_df(y_pred_df, output_dir):
    y_tag = y_pred_df["y_tag"].unique()[0]
    infoIDs = list(y_pred_df["infoID"].unique())

    output_dir = output_dir+y_tag+ "/"
    create_output_dir(output_dir)

    print("\nGraphing %s"%y_tag)
    # y_ground_truth_dict = infoID_train_and_test_array_dict[y_tag]

    hyp_dict = hyphenate_infoID_dict(infoIDs)
    target_fts = list(y_pred_df["target_ft"].unique())



    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]

        cur_infoID_df = y_pred_df[y_pred_df["infoID"]==infoID]
        print("\ncur_infoID_df")
        print(cur_infoID_df)



        num_target_fts = len(target_fts)
        if num_target_fts == 3:
            coord_list = [0, 1, 2]
            fig, axs = plt.subplots(3,figsize=(8,8))
        else:
            coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
            fig, axs = plt.subplots(3, 2,figsize=(8,8))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):

            cur_target_ft_df = cur_infoID_df[cur_infoID_df["target_ft"]==target_ft].reset_index(drop=True)
            print("\ncur_target_ft_df")
            print(cur_target_ft_df)

            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})
            cur_target_ft_df=cur_target_ft_df.sort_values("timestep").reset_index(drop=True)

            cur_y_pred = cur_target_ft_df["pred"].values
            cur_y_ground_truth = cur_target_ft_df["actual"].values

            # cur_y_pred = y_pred[idx]
            # cur_y_ground_truth = y_ground_truth[idx]

            target_no_underscore = target_ft.replace("_"," ")
            title_tag = "%s %s"%(infoID, target_no_underscore)
            title = ("\n".join(wrap(title_tag, 20)))
            # title = '%s \n%s'%(infoID, target_ft)

            if num_target_fts == 6:
                x_coor = coord[0]
                y_coor = coord[1]
                print("\ncur_y_pred shape")
                print(cur_y_pred.shape)
                print("\ncur_y_ground_truth shape")
                print(cur_y_ground_truth.shape)
                axs[x_coor,y_coor].plot(cur_y_pred,":r" ,label="Prediction")
                axs[x_coor,y_coor].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
                axs[x_coor,y_coor].set_title(title)
            else:
                # x_coor = coord[0]
                # y_coor = coord[1]
                print("\ncur_y_pred shape")
                print(cur_y_pred.shape)
                print("\ncur_y_ground_truth shape")
                print(cur_y_ground_truth.shape)
                axs[coord].plot(cur_y_pred,":r" ,label="Prediction")
                axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
                axs[coord].set_title(title)

            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
            idx+=1

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)



def create_prediction_df_with_corr_data_v2_hourly_option(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,y_pred_dict, infoID_train_and_test_array_dict,infoIDs,target_ft_categories,array_tag,model_tag, output_dir,SAVE_PREDS=True):

    if SAVE_PREDS == True:
        cur_output_dir = output_dir + model_tag + "/"
        create_output_dir(cur_output_dir)

    y_tag = "y_" + array_tag
    x_tag = "x_" + array_tag


    all_dfs = []

    try:
        cur_array_type_correlation_materials_dict = array_type_to_correlation_materials_dict[array_tag]
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=cur_array_type_correlation_materials_dict["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"]
        num_overall_timesteps = cur_array_type_correlation_materials_dict["num_overall_timesteps"]
    except:
        print("no corr data, continuing")



    for infoID in infoIDs:


        y_pred =  y_pred_dict[infoID]
        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]



        print("\ny_pred.shape")
        print(y_pred.shape)
        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        num_target_ft_categories = len(target_ft_categories)
        num_timesteps_per_sample = int(y_pred.shape[1]/num_target_ft_categories)
        print("\nnum_timesteps_per_sample")
        print(num_timesteps_per_sample)

        y_pred = y_pred.reshape((y_pred.shape[0] * num_timesteps_per_sample, num_target_ft_categories))
        y_ground_truth = y_ground_truth.reshape((y_ground_truth.shape[0] * num_timesteps_per_sample, num_target_ft_categories))

        print("\ny_pred.shape after timestep hack")
        print(y_pred.shape)
        print("\ny_ground_truth.shape after timestep hack")
        print(y_ground_truth.shape)

        # sys.exit(0)

        for target_ft in target_ft_categories:

            pred_df = pd.DataFrame(data=y_pred, columns=target_ft_categories)
            pred_df=pred_df[[target_ft]]

            ground_truth_df = pd.DataFrame(data=y_ground_truth, columns=target_ft_categories)
            ground_truth_df=ground_truth_df[[target_ft]]

            pred_df = pred_df.rename(columns={target_ft:"pred"})
            ground_truth_df = ground_truth_df.rename(columns={target_ft:"actual"})
            df = pd.concat([pred_df, ground_truth_df], axis=1)
            print(df)

            for platform in platforms:
                if platform in target_ft:
                    df["platform"]=platform
                    break
            df["infoID"] = infoID
            # print(df)
            df["timestep"] = [(i+1) for i in range(df.shape[0])]
            df["target_ft"] = target_ft
            col_order = ["timestep", "pred", "actual", "platform", "infoID","target_ft"]

            for corr_func_str in CORRELATION_FUNCS_TO_USE_LIST:
                col_order.append(corr_func_str)
                corr_vals = []
                for overall_timestep in range(num_overall_timesteps):
                    avg_corr = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][target_ft]
                    corr_vals.append(avg_corr)
                df[corr_func_str] = corr_vals

            print("\ndf")
            print(df)


            df = df[col_order]
            print(df)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    print(df)

    if SAVE_PREDS == True:
        output_fp = cur_output_dir + y_tag + "-" + model_tag + "-predictions.csv"
        df.to_csv(output_fp, index=False)
        print(output_fp)


    return df



def graph_pred_vs_gt_with_baseline_from_comp_df_v3_redo_metrics(proper_comp_df, target_ft_rename_dict,model_type, model_tag,baseline_tag, metric_tag,platform,graph_output_dir,metric_str_to_function_dict,LOG=False):
    create_output_dir(graph_output_dir)


    model_pred_col = model_type + "_pred"
    baseline_pred_col = baseline_tag + "_pred"
    infoIDs = list(proper_comp_df["infoID"].unique())
    num_timesteps = proper_comp_df["timestep"].nunique()
    target_fts = list(proper_comp_df["target_ft"].unique())
    num_target_fts = len(target_fts)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    png_dir = graph_output_dir + "PNGS/"
    create_output_dir(png_dir)

    svg_dir = graph_output_dir + "SVGS/"
    create_output_dir(svg_dir)

    pdf_dir = graph_output_dir + "PDFS/"
    create_output_dir(pdf_dir)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        #============ plot stuff ==============

        coord_list = [0, 1, 2]
        coord_list = coord_list[:num_target_fts]
        fig, axs = plt.subplots(3,figsize=(9,9))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

        for coord,target_ft in enumerate(target_fts):
            print("\nPlotting %s %s %s..."%(platform, infoID, target_ft))

            target_ft_to_axis_coordinates_dict[target_ft] = coord



            temp_df = proper_comp_df[(proper_comp_df["infoID"]==infoID) & (proper_comp_df["platform"]==platform) & (proper_comp_df["target_ft"]==target_ft)]
            temp_df = temp_df.reset_index(drop=True)
            print("\ntemp_df")
            print(temp_df)

            if temp_df.shape[0] != num_timesteps:
                print("\nError! temp_df.shape[0] != num_timesteps")
                print(temp_df.shape[0])
                print(num_timesteps)
                sys.exit(0)
            else:
                print("\nCounts are good...Plotting...")

            #======================= get series =======================
            model_pred_series = temp_df[model_pred_col]
            baseline_pred_series = temp_df[baseline_pred_col]
            gt_series = temp_df["actual"]

            #======================= score stuff =======================
            # model_mean_error = temp_df[model_type + "_" + metric_tag].mean()

            metric_func = metric_str_to_function_dict[metric_tag]
            model_mean_error = metric_func(gt_series, model_pred_series)

            model_error_result_tag = "%s %s: %.4f"%(model_tag, metric_tag, model_mean_error)
            # print()
            # print(model_error_result_tag)

            #baseline_mean_error = temp_df[baseline_tag + "_" + metric_tag].mean()
            baseline_mean_error = metric_func(gt_series, baseline_pred_series)
            baseline_error_result_tag = "%s %s: %.4f"%(baseline_tag , metric_tag, baseline_mean_error)
            # print(baseline_error_result_tag)

            # model_score_tag = "%s win freq: %.4f"%(model_type,  model_win_score)
            # print(model_score_tag)

            full_report_tag = model_error_result_tag + "\n" + baseline_error_result_tag

            print()
            print(full_report_tag)

            # target_no_underscore = target_ft.replace("_"," ")
            # title_tag = "%s %s"%(infoID, target_no_underscore)
            # title_tag = title_tag + "\n"+ full_report_tag
            # title = ("\n".join(wrap(title_tag, 50)))

            new_target_name = target_ft_rename_dict[target_ft]

            target_no_underscore = new_target_name.replace("_"," ")

            infoID_target_tag = "%s - %s"%(infoID, target_no_underscore)
            infoID_target_tag = infoID_target_tag.replace("platform infoID", "platform-topic").title()
            title_tag = infoID_target_tag + "\n"+ full_report_tag
            title = title_tag

            # print("\ncur_y_pred shape")
            # print(cur_y_pred.shape)
            # print("\ncur_y_ground_truth shape")
            # print(cur_y_ground_truth.shape)
            axs[coord].plot(model_pred_series,"-r" ,label=model_tag)
            axs[coord].plot(baseline_pred_series,":g" ,label=baseline_tag)
            axs[coord].plot(gt_series, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='Hours', ylabel='Volume')
            ax.legend()

            if LOG == True:
                ax.set_yscale('log')

        # plt.legend()
        plt.tight_layout()

        output_fp = png_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        print(output_fp)

        output_fp = pdf_dir + "%s.pdf"%( hyp_infoID)
        fig.savefig(output_fp)
        print(output_fp)

        output_fp = svg_dir + "%s.svg"%( hyp_infoID)
        fig.savefig(output_fp)
        print(output_fp)




        plt.close()





def graph_pred_vs_gt_with_baseline_from_comp_df_v2_simple(proper_comp_df, model_tag, baseline_tag, metric_tag,platform,graph_output_dir):
    create_output_dir(graph_output_dir)


    model_pred_col = model_tag + "_pred"
    baseline_pred_col = baseline_tag + "_pred"
    infoIDs = list(proper_comp_df["infoID"].unique())
    num_timesteps = proper_comp_df["timestep"].nunique()
    target_fts = list(proper_comp_df["target_ft"].unique())

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        #============ plot stuff ==============

        coord_list = [0, 1, 2]
        fig, axs = plt.subplots(3,figsize=(9,9))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

        for coord,target_ft in enumerate(target_fts):
            print("\nPlotting %s %s %s..."%(platform, infoID, target_ft))

            target_ft_to_axis_coordinates_dict[target_ft] = coord



            temp_df = proper_comp_df[(proper_comp_df["infoID"]==infoID) & (proper_comp_df["platform"]==platform) & (proper_comp_df["target_ft"]==target_ft)]
            temp_df = temp_df.reset_index(drop=True)
            print("\ntemp_df")
            print(temp_df)

            if temp_df.shape[0] != num_timesteps:
                print("\nError! temp_df.shape[0] != num_timesteps")
                print(temp_df.shape[0])
                print(num_timesteps)
                sys.exit(0)
            else:
                print("\nCounts are good...Plotting...")

            #======================= get series =======================
            model_pred_series = temp_df[model_pred_col]
            baseline_pred_series = temp_df[baseline_pred_col]
            gt_series = temp_df["actual"]

            #======================= score stuff =======================
            model_mean_error = temp_df[model_tag + "_" + metric_tag].mean()
            # model_wins = temp_df[temp_df["is_winner"]==1].shape[0]
            # num_trials_no_ties = temp_df[temp_df["is_winner"]!="tie"].shape[0]
            # try:
            #     model_win_score = model_wins/num_trials_no_ties
            # except ZeroDivisionError:
            #     model_win_score = 0
            model_error_result_tag = "%s %s: %.4f"%(model_tag, metric_tag, model_mean_error)
            # print()
            # print(model_error_result_tag)

            baseline_mean_error = temp_df[baseline_tag + "_" + metric_tag].mean()
            baseline_error_result_tag = "%s %s: %.4f"%(baseline_tag , metric_tag, baseline_mean_error)
            # print(baseline_error_result_tag)

            # model_score_tag = "%s win freq: %.4f"%(model_tag,  model_win_score)
            # print(model_score_tag)

            full_report_tag = model_error_result_tag + "\n" + baseline_error_result_tag
            print()
            print(full_report_tag)

            # target_no_underscore = target_ft.replace("_"," ")
            # title_tag = "%s %s"%(infoID, target_no_underscore)
            # title_tag = title_tag + "\n"+ full_report_tag
            # title = ("\n".join(wrap(title_tag, 50)))

            target_no_underscore = target_ft.replace("_"," ")
            infoID_target_tag = "%s %s"%(infoID, target_no_underscore)
            infoID_target_tag = infoID_target_tag.replace("platform infoID", "platform-topic")
            title_tag = infoID_target_tag + "\n"+ full_report_tag
            title = title_tag

            # print("\ncur_y_pred shape")
            # print(cur_y_pred.shape)
            # print("\ncur_y_ground_truth shape")
            # print(cur_y_ground_truth.shape)
            axs[coord].plot(model_pred_series,"-r" ,label=model_tag)
            axs[coord].plot(baseline_pred_series,":g" ,label=baseline_tag)
            axs[coord].plot(gt_series, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')
            ax.legend()

        # plt.legend()
        plt.tight_layout()
        output_fp = graph_output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)