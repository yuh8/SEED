import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

base = pd.read_csv('generated_molecules_base.csv')
chembl = pd.read_csv('generated_molecules_chembl.csv')
rl = pd.read_csv('generated_molecules_rl.csv')

base.loc[:, 'Source'] = 'SEEM'

chembl.loc[:, 'Source'] = 'ChEMBL'

rl.loc[:, 'Source'] = 'SEEM_RL'

breakpoint()
df_all = pd.concat([base, chembl.sample(n=10000, replace=False), rl])

df_all.reset_index(drop=True, inplace=True)

sns.displot(df_all, x="sa", hue="Source", kind="kde", fill=True)
plt.show()
