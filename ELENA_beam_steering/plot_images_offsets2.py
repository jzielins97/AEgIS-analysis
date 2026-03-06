from ALPACA_wrapper import ALPACA_load_data
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
import matplotlib.colors as mcolors
import seaborn as sns
import os

# parameters
FIGSIZE = (6,5) # (18,15)
# small scan offsets 492673:492688
# small scan angles 492693:492709
FIRST_RUN=492693
LAST_RUN=492955
SCAN_TYPE = "angle" # angle or offset
UNIT = "mm" if SCAN_TYPE.lower() == "offset" else "mrad"
V_VALUES_TO_PLOT = np.flip(np.arange(-15,1,5)) if SCAN_TYPE.lower() == "offset" else np.flip(np.arange(0,7,2))
H_VALUES_TO_PLOT = np.arange(-15,1,5) if SCAN_TYPE.lower() == "offset" else np.flip(np.arange(-3,4,2))

def load_data(first_run:int|None=None,last_run:int|None=None,bad_runs:list[int]|None=None,filename:str="PbarsOnHHMCP_scan_2.parquet")->pl.DataFrame:
    data = pl.scan_parquet(os.path.join(os.path.dirname(__file__),"data",filename))
    if first_run is not None:
        data = data.filter(pl.col("run")>=first_run)
    if last_run is not None:
        data = data.filter(pl.col("run")<=last_run)
    if bad_runs is not None:
        data = data.filter(~pl.col("run").is_in(bad_runs))
    return data.collect()


def plot_single_run_image(img_data:list|np.typing.NDArray,img_width:int,img_height:int,cmap:list[float],norm:mcolors.Normalize=mcolors.LogNorm(),ax:plt.Axes=None)->tuple[QuadContourSet,plt.Axes]:
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)
    # get x,y arrays
    # run_data = data.filter(pl.col("run")==run)
    # img_width = run_data.select("MCP_img_width").item()
    # img_height = run_data.select("MCP_img_height").item()
    x_px, y_px = np.meshgrid([x+0.5 for x in range(img_width)],
                             [y+0.5 for y in range(img_height)])
    intensity = np.reshape(img_data,(img_height,img_width)) # run_data.select("MCP_img").item()
    im = ax.contourf(x_px, y_px, intensity, 100, cmap=cmap, norm=norm)    
    return im,ax

def plot_all_images(data:pl.DataFrame):
    # load data 
    sns.set_context("paper",font_scale=1.3)    
    
    norm = mcolors.Normalize(vmin=data.select(pl.col("MCP_img").arr.min().min()).item(),vmax=data.select(pl.col("MCP_img").arr.max().max()).item())
    fig,axes = plt.subplots(len(H_VALUES_TO_PLOT),len(V_VALUES_TO_PLOT),squeeze=False,sharex=True,sharey=True,figsize=FIGSIZE)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    data_filter = data.filter(pl.col(f"V_{SCAN_TYPE.lower()}_requested").is_in(V_VALUES_TO_PLOT) & pl.col(f"H_{SCAN_TYPE.lower()}_requested").is_in(H_VALUES_TO_PLOT))
    for run in pl.Series(data_filter.select("run")).to_list():
        run_data = data_filter.filter(pl.col("run")==run)
        print(run)
        H_value = run_data.select(f'H_{SCAN_TYPE.lower()}_{UNIT}').item()
        H_value_requested = run_data.select(f'H_{SCAN_TYPE.lower()}_requested').item()
        V_value = run_data.select(f'V_{SCAN_TYPE.lower()}_{UNIT}').item()
        V_value_requested = run_data.select(f'V_{SCAN_TYPE.lower()}_requested').item()
        values_set = H_value==H_value_requested and V_value==V_value_requested
        print(f"\tH {SCAN_TYPE.lower()} = {H_value} should be {H_value_requested} ({H_value==H_value_requested})")
        print(f"\tV {SCAN_TYPE.lower()} = {V_value} should be {V_value_requested} ({V_value==V_value_requested})")
        
        col = np.where(H_VALUES_TO_PLOT == H_value_requested)[0][0]
        row = np.where(V_VALUES_TO_PLOT == V_value_requested)[0][0]
        
        ax:plt.Axes = axes[row][col]
        im,_ = plot_single_run_image(img_data=run_data.select("MCP_img").item(),
                                     img_width=run_data.select("MCP_img_width").item(),
                                     img_height=run_data.select("MCP_img_height").item(),
                                     cmap=plt.get_cmap('magma'), # if values_set else plt.get_cmap("gray"),
                                     norm=norm,
                                     ax=ax)

        ax.set_xlabel(H_value,color='black' if H_value==H_value_requested else 'red')
        ax.set_ylabel(V_value,color='black' if V_value==V_value_requested else 'red')
        
        if col == len(H_VALUES_TO_PLOT)-1:
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(V_value_requested)
        
        if row == 0:
            ax.set_title(H_value_requested)
    
    ax = fig.add_subplot(111, frameon=False)
    ax.set_xlabel(f"H {SCAN_TYPE.upper()} set",labelpad=15)
    ax.set_ylabel(f"V {SCAN_TYPE.upper()} set",labelpad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2 = fig.add_subplot(111, frameon=False)
    ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel(f"H {SCAN_TYPE.upper()} requested",labelpad=15)
    ax2.set_ylabel(f"V {SCAN_TYPE.upper()} requested",labelpad=15)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()
    # fig.subplots_adjust(left=0.1, top=0.9,bottom=0.1,right=0.9)
    fig.savefig(os.path.join(os.path.dirname(__file__),"plots",f'scan_{SCAN_TYPE.lower()}_results.png'))
    plt.show()
    
def plot_single_image(data:pl.DataFrame,run:int,showfig:bool=False):
    sns.set_context("paper",font_scale=0.8)
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=data.select(pl.col("MCP_img").arr.min().min()).item(),vmax=data.select(pl.col("MCP_img").arr.max().max()).item())
    fig,ax = plt.subplots(figsize=FIGSIZE)
    run_data = data.filter(pl.col("run")==run)
    print(run_data)
        
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
    ax.text(20,20,f'V_OFFSET={run_data.select("V_offset_mm").item()} (requested {run_data.select("V_offset_requested").item()})\nV_ANGLE={run_data.select("V_angle_mrad").item()}  (requested {run_data.select("V_angle_requested").item()})\nH_OFFSET={run_data.select("H_offset_mm").item()} (requested {run_data.select("H_offset_requested").item()})\nH_ANGLE={run_data.select("H_angle_mrad").item()}  (requested {run_data.select("H_angle_requested").item()})',color='white')
    
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__),"plots",f'{run}_Voffset{run_data.select("V_offset_requested").item()}_Hoffset{run_data.select("H_offset_requested").item()}-Vangle{run_data.select("V_angle_requested").item()}-Hangle{run_data.select("H_angle_requested").item()}.png'))
    if showfig:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    print("Load data")
    data = load_data(first_run=FIRST_RUN,last_run=LAST_RUN)
    print(data)
    # plot_single_image(data,492955,showfig=True)
    # plot_all_images(data)
    for run in range(492948,492956):
        try:
            print(run)
            plot_single_image(data,run,showfig=False)
        except Exception as e:
            print(type(e))
            print(e)