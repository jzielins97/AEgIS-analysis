import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os

bins = np.arange(0,1e-4,10e-9)
df:list[pl.DataFrame] = []
for i in range(2):
    df.append(pl.read_parquet(os.path.join(os.path.dirname(__file__),f"calibration_runs_part{i+1}.parquet")))
    print(df[i]) 
    fig,ax = plt.subplots()
    for run in df[i]["run"]:
        df_tmp = (df[i].filter(pl.col("run")==run)
                .explode("t_s","signal_V")
                .with_columns(bins=pl.col("t_s").cut(bins,include_breaks=True))
                .unnest("bins")
                .rename({"breakpoint":"t_bin_s"})
                .group_by("t_bin_s").agg(pl.col("signal_V").mean(),
                                         err=pl.col("signal_V").std())
                .sort("t_bin_s"))
        if i == 1:
            df_tmp = df_tmp.with_columns(pl.col("t_bin_s") - 1e-5)    
        df_tmp = df_tmp.filter(pl.col("t_bin_s").is_between(0,10e-6))   
        ax.errorbar(df_tmp['t_bin_s'],df_tmp['signal_V'],yerr=df_tmp['err'],fmt='o',label=run)
    # ax.legend()

# for i in range(len(df)):
#     df[i] = df[i].explode("t_s","signal_V")
#     if i == 1:
#         df[i] = df[i].with_columns(pl.col("t_s") - 1e-5)
# df:pl.DataFrame = pl.concat(df)
# df = df.group_by(pl.all().exclude("t_s","signal_V")).agg(pl.col("t_s"),pl.col("signal_V")).write_parquet(os.path.join(os.path.dirname(__file__),"calibration_runs.parquet"))
plt.show()