import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import networkx as nx

def bertopic_merge(news_df, p_value, meandiff):
    df_ipc = pd.read_csv("data/food_crises_cleaned.csv") # Read data into DataFrame
    df_ipc["date"] = pd.to_datetime(df_ipc["year_month"], format="%Y_%m") # Create date column
    df_ipc = df_ipc[["ipc","year_month", "district"]] # Extract useful column
    df_ipc = df_ipc.dropna(subset=['ipc'])
    df_ipc.rename(columns={'year_month':'date', 'district':'location_article'},inplace=True) # Rename columns in order to merge with news df
    df_ipc.sort_values(by=['location_article', 'date'], inplace=True) # Sort the DataFrame by 'location_article' and 'date'
    df_ipc['1_month_lag'] = df_ipc.groupby('location_article')['ipc'].shift(-1) # Group the DataFrame by 'location_article' and apply a one-month lag to 'ipc' within each group
    df_ipc['3_month_lag'] = df_ipc.groupby('location_article')['ipc'].shift(-3) # Group the DataFrame by 'location_article' and apply a one-month lag to 'ipc' within each group

    df_merge = news_df.merge(df_ipc, on=['location_article','date'])

    # Group the DataFrame by 'topics' and aggregate the 'ipc' into lists
    auto_1_lag_group = df_merge.groupby('topics_auto')['1_month_lag'].agg(list).reset_index()
    auto_1_lag_group['1_month_lag'] = auto_1_lag_group['1_month_lag'].apply(lambda x: [item for item in x if not pd.isna(item)])
    auto_3_lag_group = df_merge.groupby('topics_auto')['3_month_lag'].agg(list).reset_index()
    auto_3_lag_group['3_month_lag'] = auto_3_lag_group['3_month_lag'].apply(lambda x: [item for item in x if not pd.isna(item)])

    # Flatten the 1 month lag data and assign group labels
    flattened_data_1_month = [val for sublist in auto_1_lag_group['1_month_lag'] for val in sublist]
    group_labels_1_motnh = [label for label, data_list in zip(auto_1_lag_group['topics_auto'], auto_1_lag_group['1_month_lag']) for _ in data_list]
    # Perform Tukey's HSD post hoc test
    posthoc_1_month = pairwise_tukeyhsd(flattened_data_1_month, group_labels_1_motnh)
    # Convert the TukeyHSDResults to a DataFrame
    df_posthoc_1_month = pd.DataFrame(data=posthoc_1_month._results_table.data[1:], columns=posthoc_1_month._results_table.data[0])
    # Filter rows where 'p-adj' > 0.9 and 'meandiff' < 0.1
    significant_comparisons_1_month = df_posthoc_1_month[(df_posthoc_1_month['p-adj']>p_value) & (abs(df_posthoc_1_month['meandiff']) < meandiff)]

    # Flatten the 3 month lag data and assign group labels
    flattened_data_3_month = [val for sublist in auto_3_lag_group['3_month_lag'] for val in sublist]
    group_labels_3_motnh = [label for label, data_list in zip(auto_3_lag_group['topics_auto'], auto_3_lag_group['3_month_lag']) for _ in data_list]
    # Perform Tukey's HSD post hoc test
    posthoc_3_month = pairwise_tukeyhsd(flattened_data_3_month, group_labels_3_motnh)
    # Convert the TukeyHSDResults to a DataFrame
    df_posthoc_3_month = pd.DataFrame(data=posthoc_3_month._results_table.data[1:], columns=posthoc_3_month._results_table.data[0])
    # Filter rows where 'p-adj' > 0.9 and 'meandiff' < 0.1
    significant_comparisons_3_month = df_posthoc_3_month[(df_posthoc_3_month['p-adj']>p_value) & (abs(df_posthoc_3_month['meandiff']) < meandiff)]

    df_both = significant_comparisons_1_month.merge(significant_comparisons_3_month, on=['group1','group2'])
    df_both['avg_mean'] = (abs(df_both['meandiff_x']) + abs(df_both['meandiff_y']))/2
    df_both = df_both.sort_values(by='avg_mean', ascending=True)
    return df_both[['group1', 'group2','avg_mean', 'p-adj_x', 'p-adj_y']]