import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['run','t','V'])
for i in range(5):
    df = pd.concat([df,pd.DataFrame({'run':i,'t':[j for j in range(10)],'V':[np.random.random() for _ in range(10)]})],ignore_index=True)




print(df)
    
mean = df.groupby('t').mean()
mean_cut = df[['t','V']].groupby('t').mean(True)
print(mean)
print(mean_cut)
print(mean-mean_cut)