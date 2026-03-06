import polars as pl
import os
import seaborn as sns
import matplotlib.pyplot as plt    

corrector_mapping = {
    "H1_":"DHZE05",
    "V1_":"DVTE05",
    "H2_":"DHZE08",
    "V2_":"DVTE08",
    "H3_":"DHZE14",
    "V3_":"DVTE14",
    "QD1_":"QDNE08",
    "QF1_":"QFNE09",
    "QD2_":"QDNE14",
    "QF2_":"QFNE15"
}

data = pl.read_parquet(os.path.join(os.path.dirname(__file__),"data","ELENA_correctors.parquet"))
print(data)

fig1,ax1 = plt.subplots(5,1,sharex=True,figsize=(16,10))

# offsets
offsets = data.filter(pl.col("run").is_between(492042,492482))
for param in ["H_offset_mm","V_offset_mm"]:
    sns.scatterplot(offsets,x='run',y=param,label=param,ax=ax1[0])

fig2,ax2 = plt.subplots(5,1,sharex=True,figsize=(16,10))
angles = data.filter(pl.col("run").is_between(492483,492651))
for param in ["H_angle_mrad","V_angle_mrad"]:
    sns.scatterplot(angles,x='run',y=param,label=param,ax=ax2[0])

for ax,data_set in zip([ax1,ax2],[offsets,angles]):
    for param in [f"H{idx}_L" for idx in range(1,4)]:
        sns.scatterplot(data_set,x='run',y=param,label=f"{param.replace(param[:-1],corrector_mapping[param[:-1]])}:MEAS.V.VALUE",ax=ax[1])
    # for param in [f"H{idx}_R" for idx in range(1,4)]:
    #     sns.scatterplot(offsets,x='run',y=param,label=param,ax=ax[1])
    for param in [f"V{idx}_T" for idx in range(1,4)]:
        sns.scatterplot(data_set,x='run',y=param,label=f"{param.replace(param[:-1],corrector_mapping[param[:-1]])}:MEAS.V.VALUE",ax=ax[2])
    # for param in [f"V{idx}_B" for idx in range(1,4)]:
    #     sns.scatterplot(offsets,x='run',y=param,label=param,ax=ax[2])
    # for param in [f"QD{idx}_N" for idx in range(1,3)]:
    #     sns.scatterplot(offsets,x='run',y=param,label=param,ax=ax[3])
    for param in [f"QD{idx}_P" for idx in range(1,3)]:
        sns.scatterplot(data_set,x='run',y=param,label=f"{param.replace(param[:-1],corrector_mapping[param[:-1]])}:MEAS.V.VALUE",ax=ax[3])
    # for param in [f"QF{idx}_N" for idx in range(1,3)]:
    #     sns.scatterplot(offsets,x='run',y=param,label=param,ax=ax[4])
    for param in [f"QF{idx}_P" for idx in range(1,3)]:
        sns.scatterplot(data_set,x='run',y=param,label=f"{param.replace(param[:-1],corrector_mapping[param[:-1]])}:MEAS.V.VALUE",ax=ax[4])

for ax in ax1:
    ax.set_ylabel("value")
for ax in ax2:
    ax.set_ylabel("value")

fig1.tight_layout()
fig1.savefig(os.path.join(os.path.dirname(__file__),"plots","correctors_offsets_scan.png"))
fig2.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__),"plots","correctors_angles_scan.png"))
plt.show()
