from collections import defaultdict
from datasets import get_dataset_config_names
from datasets import load_dataset, DatasetDict

# #XTREME benchmark where XTREME stands for cross-lingual transfer evaluation for multilingual encoders.
# xtreme_subsets = get_dataset_config_names('xtreme')
# print(f'XTREME has {len(xtreme_subsets)} configurations')
#
# # using PAN subset from the XTREME benchmark dataset
# PAN_subsets = [subset for subset in xtreme_subsets if subset.startswith('PAN')]

langs = ['de', 'fr', 'it', 'en']
fracs = [0.629, 0.229, 0.084, 0.059]

# return a dataset dict if the key doesn't exist
panx_ch = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    # load monolingual corpus
    ds = load_dataset('xtreme', name=f'PAN-X.{lang}')
    for split in ds:
        panx_ch[lang][split] = ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows)))





