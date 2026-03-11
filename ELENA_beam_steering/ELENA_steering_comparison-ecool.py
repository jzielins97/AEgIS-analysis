from ALPACA_wrapper import ALPACA_load_data
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
import matplotlib.colors as mcolors
import seaborn as sns
import os

pl.Config.set_tbl_cols(20)
FIGSIZE = (6,5)

def plot_single_run_image(img_data:list|np.typing.NDArray,img_width:int,img_height:int,cmap:list[float],norm:mcolors.Normalize=mcolors.LogNorm(),ax:plt.Axes=None)->tuple[QuadContourSet,plt.Axes]:
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)
    x_px, y_px = np.meshgrid([x+0.5 for x in range(img_width)],
                             [y+0.5 for y in range(img_height)])
    intensity = np.reshape(img_data,(img_height,img_width)) # run_data.select("MCP_img").item()
    im = ax.contourf(x_px, y_px, intensity, 100, cmap=cmap, norm=norm)    
    return im,ax


def plot_single_image(data:pl.DataFrame,run:int,showfig:bool=False):
    sns.set_context("paper",font_scale=0.8)
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=data.select(pl.col("MCP_img").list.min().min()).item(),vmax=data.select(pl.col("MCP_img").list.max().max()).item())
    fig,ax = plt.subplots(figsize=FIGSIZE)
    run_data = data.filter(pl.col("Run Number")==run)
    # print(run_data)
        
    im,_ = plot_single_run_image(img_data=run_data.select("MCP_img").item(),
                                img_width=run_data.select("MCP_img_width").item(),
                                img_height=run_data.select("MCP_img_height").item(),
                                cmap=cmap,
                                norm=norm,
                                ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
        
    # cleanup rest of the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{run}')
    # ax.text(20,20,f'V_OFFSET={run_data.select("V_offset_mm").item()} (requested {run_data.select("V_offset_requested").item()})\nV_ANGLE={run_data.select("V_angle_mrad").item()}  (requested {run_data.select("V_angle_requested").item()})\nH_OFFSET={run_data.select("H_offset_mm").item()} (requested {run_data.select("H_offset_requested").item()})\nH_ANGLE={run_data.select("H_angle_mrad").item()}  (requested {run_data.select("H_angle_requested").item()})',color='white')
    ax.text(20,20,f"{run_data.select('settings').item()}")
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__),"plots",f"{run}_{run_data.select('settings').item()}.png"))
    if showfig:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    variables_of_interest = [
        'SC56_coinc*events_in_interval',
        'SC56_coinc*event_clock',
        'Hikrobot*acq_0*height',
        'Hikrobot*acq_0*width',
        'Hikrobot*acq_0*C_flatten_data'
    ]+[
        f'ELENA_Parameters*H{id}_Corrector_{pos}' for id in [1,2,3] for pos in ['L','R']
    ]+[
        f'ELENA_Parameters*V{id}_Corrector_{pos}' for id in [1,2,3] for pos in ['T','B']    
    ]+[
        f'ELENA_Parameters*{dir}{id}_{pos}' for dir in ['QF','QD'] for id in [1,2] for pos in ['P','N']
    ]
                            
    column_names = [
        'SC56_coinc*events_in_interval',
        'SC56_coinc*event_clock',
        'MCP_img_height',
        'MCP_img_width',
        'MCP_img',
        'DHZE05L',
        'DHZE05R',
        'DHZE08L',
        'DHZE08R',
        'DHZE14L',
        'DHZE14R',
        'DVTE05T',
        'DVTE05B',
        'DVTE08T',
        'DVTE08B',
        'DVTE14T',
        'DVTE14B',
        'QFNE09P',
        'QFNE09N',
        'QFNE15P',
        'QFNE15N',
        'QDNE08P',
        'QDNE08N',
        'QDNE14P',
        'QDNE14N'
    ]
    
    default_steering = [493824]
    optimised_steering = [493823,493825,493826]
    image_runs = [493827]

    data = ALPACA_load_data(
        runs = default_steering + optimised_steering + image_runs,
        # first_run=493022,
        # last_run=493033,
        variables_of_interest=variables_of_interest,
        # directories_to_flush=[],
        # speed_mode=False,
        column_names=column_names,
        output_file_name="ELENA_comparison-ecool.parquet",
        dtype={'SC12_coinc*event_clock':pl.List,
            'MCP_img_height':pl.Int32,
            'MCP_img_width':pl.Int32,
            'MCP_img':pl.List},
        download_from_ALPACA='missing',
        cleanup_at_exit=True)

    print(data)

    data = data.with_columns(pl.when(pl.col("Run Number").is_in(default_steering)).then(pl.lit("default"))
                            .when(pl.col("Run Number").is_in(optimised_steering)).then(pl.lit("optimiser"))
                            .when(pl.col("Run Number")==493034).then(pl.lit("10D optimised"))
                            .otherwise(pl.lit("images")).alias("settings"))

    data_mean=data.filter(~pl.col("Run Number").is_in(image_runs)).group_by("settings").agg(pl.col("SC12_coinc*events_after_hot_dump").mean().alias("mean counts"),
                                            pl.col("SC12_coinc*events_after_hot_dump").std().alias("std counts"),
                                            )
    print(data_mean)
    best_settings = data.filter(pl.col("SC56_coinc*events_in_interval")==pl.col("SC56_coinc*events_in_interval").max())
    print(best_settings)
    
    for run in image_runs:
        try:
            print(run)
            plot_single_image(data,run,showfig=False)
        except Exception as e:
            print(type(e))
            print(e)
    plt.show()