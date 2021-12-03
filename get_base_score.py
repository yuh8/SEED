import numpy as np
import pandas as pd
from src.reward_utils import (get_logp_reward, get_sa_reward,
                              get_qed_reward, get_cycle_reward)


df_base = pd.read_csv("D:/seed_data/generator/train_data/df_train.csv", low_memory=False)[:1000000]
gen_samples_df = []
count = 0
for _, row in df_base.iterrows():
    gen_sample = {}
    try:
        gen_sample["Smiles"] = row.Smiles
        gen_sample['logp'] = np.round(get_logp_reward(row.Smiles), 4)
        gen_sample['sa'] = np.round(get_sa_reward(row.Smiles), 4)
        gen_sample['cycle'] = get_cycle_reward(row.Smiles)
        gen_sample['qed'] = np.round(get_qed_reward(row.Smiles), 4)
    except:
        continue
    gen_samples_df.append(gen_sample)
    count += 1
    print("{} / {} done".format(count, df_base.shape[0]))


gen_samples_df = pd.DataFrame(gen_samples_df)
gen_samples_df.to_csv('generated_molecules_chembl.csv', index=False)
