import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'diaper', 'beer', 'egg'],
    ['milk', 'diaper', 'beer', 'cola'],
    ['bread', 'milk', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'cola']
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

freq_items = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)

print('Frequent Itemsets:
', freq_items)
print('
Association Rules:
', rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
