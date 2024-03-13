import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import os
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sanbomics.plots import volcano
import gseapy as gp
from gseapy import barplot, dotplot
from gseapy.plot import gseaplot
from sanbomics.tools import id_map
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Get all the file names in the current directory
filenames = glob.glob(r"/courses/BINF6310.202410/students/bhowmick.a/PanC/Included_samples/TWA/*.tsv")

# Create a list to store the DataFrames
dataframes = []

# Iterate over the file names and create a DataFrame for each file
for i in range(len(filenames)):
 # Create a DataFrame from the file
 df = pd.read_csv(filenames[i], sep='\t')

 # Give the DataFrame a name
 dataframe_name = "tw{}".format(i + 1)

 # Add the DataFrame to the list
 dataframes.append((dataframe_name, df))

# Create a dictionary to store the DataFrames with their respective names
twdf = dict(dataframes)

# Print the dictionary
print(twdf)

# Get all the file names in the current directory
tmfilenames = glob.glob(r"/courses/BINF6310.202410/students/bhowmick.a/PanC/Included_samples/TMA/*.tsv")

# Create a list to store the DataFrames
tmdataframes = []

# Iterate over the file names and create a DataFrame for each file
for i in range(len(tmfilenames)):
 # Create a DataFrame from the file
 df = pd.read_csv(tmfilenames[i], sep='\t')

 # Give the DataFrame a name
 tmdataframe_name = "tm{}".format(i + 1)

 # Add the DataFrame to the list
 tmdataframes.append((tmdataframe_name, df))

# Create a dictionary to store the DataFrames with their respective names
tmdf = dict(tmdataframes)

# TMA lncRNA
for i in range(1, 116): tmdf[f'tm{i}'] = tmdf[f'tm{i}'][tmdf[f'tm{i}']["gene_type"] == "lncRNA"]

# TWA lncRNA
for i in range(1, 37): twdf[f'tw{i}'] = twdf[f'tw{i}'][twdf[f'tw{i}']["gene_type"] == "lncRNA"]

# Get unstranded data from TWA and TWM
twun = pd.DataFrame()
for i in range(1, 37):
    new_column = pd.DataFrame(zip(twdf[f'tw{i}']['unstranded']), columns=[f'tw{i}'])
    twun = pd.concat([twun, new_column], axis=1)

tmun = pd.DataFrame()
for i in range(1, 116):
    new_column = pd.DataFrame(zip(tmdf[f'tm{i}']['unstranded']), columns=[f'tm{i}'])
    tmun = pd.concat([tmun, new_column], axis=1)

twcounts = pd.DataFrame(zip(tmdf['tm1']['gene_id']), columns=['geneid'])

counts = pd.concat([twcounts, tmun, twun], axis=1)

counts = counts.set_index('geneid')

counts = counts[counts.sum(axis=1) > 0]

# Split the column values into a list of strings
index = counts.index.str.split('.', n=1).str[0]

# Keep the first element of the list
counts.index = index

counts = counts.T

m_list = ['m'] * 115
w_list = ['w'] * 36
condition_list = m_list + w_list

metadata = pd.DataFrame(zip(counts.index, condition_list), columns=['sample', 'condition'])
metadata = metadata.set_index('sample')

# Revised metadata creation
# Check that 'condition' has two unique values ('m' and 'w')
print(metadata['condition'].unique())

dds = DeseqDataSet(counts=counts, metadata=metadata, design_factors=['condition'])

dds.deseq2()

stat_res = DeseqStats (dds, contrast=('condition','m','w'))

stat_res.summary()

res = stat_res.results_df

mapper = id_map(species = 'human')

res['symbol'] = res.index.map(mapper.mapper)

degs = res[(abs(res.log2FoldChange) > 0.5)&(res.padj<0.05)]

# Save the volcano plot
volcano(res, symbol='symbol')
plt.savefig("/courses/BINF6310.202410/students/bhowmick.a/PanC/volcano_plot.png")
plt.close()  # Close the figure to free up resources

pdegs =pd.DataFrame(degs)

pdegs.to_csv('DEGs.csv',sep=',')

dds.layers['log1p']= np.log1p(dds.layers['normed_counts'])

ddsigs= dds[:,degs.index]

cluster = pd.DataFrame(ddsigs.layers['log1p'].T, index=ddsigs.var_names, columns=ddsigs.obs_names)

sns.clustermap(cluster, z_score=0, color= 'red')

heat= cluster.T

correlation = heat.corr()
# Create a new dataframe with the gene names as the index and columns
heatmap_df = pd.DataFrame(correlation, index=cluster.index, columns=cluster.index)

correlation.to_csv('correlations.csv', sep=',')

# Save the clustermap
sns.clustermap(correlation)
plt.savefig("/courses/BINF6310.202410/students/bhowmick.a/PanC/clustermap_plot.png")
plt.close()  # Close the figure to free up resources

# GSEA set up
ranking = res[['symbol', 'stat']].dropna().sort_values('stat', ascending=False)

# GSEA
pre_res4 = gp.prerank(rnk=ranking, gene_sets="/courses/BINF6310.202410/students/bhowmick.a/PanC/databases/MSigDB.gmt", min_size=1, max_size=1000)
out4 = []

for term in list(pre_res4.results):
    out4.append([
        term,
        pre_res4.results[term]['fdr'],
        pre_res4.results[term]['es'],
        pre_res4.results[term]['nes'],
        pre_res4.results[term]['pval'],
        pre_res4.results[term]['matched_genes']
    ])

out_df4 = pd.DataFrame(out4, columns=['Term', 'fdr', 'es', 'nes', 'pval', 'matched_genes'])
out_df4.to_csv('MSigDb.csv', sep=',')

axs = pre_res4.plot(
    terms=out_df4.iloc[0:10].Term,
    show_ranking=True,
    figsize=(4, 5),
    legend_kws={'loc': (1.2, 0)}
)
plt.savefig("/courses/BINF6310.202410/students/bhowmick.a/PanC/GSEA_plot.png")
plt.close()  # Close the figure to free up resources

# Enrichment score plot
# Load data from CSV file
df = pd.read_csv("/courses/BINF6310.202410/students/bhowmick.a/PanC/databases/MSigDBplot.csv") 

# Select the top 10 terms
top_terms = df.head(10)

# Extract information for plotting
terms = top_terms['Term'].tolist()
es_values = top_terms['es'].tolist()
ranks = top_terms.index + 1  # Add 1 to make the rank start from 1

# Print extracted values for debugging
print("Terms:", terms)
print("ES Values:", es_values)
print("Ranks:", ranks)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(ranks, es_values, color='blue', label='Top 10 Terms')

# Add labels for each point
for term, rank, es in zip(terms, ranks, es_values):
    plt.text(rank, es, term, ha='left', va='bottom', color='black', fontsize=8)

plt.xlabel('Rank')
plt.ylabel('Enrichment Score (ES)')
plt.title('Top 10 Enriched Terms')
plt.legend()

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/GSEA_gene_rank_enrichment_score_plot.png"
plt.savefig(save_path)

