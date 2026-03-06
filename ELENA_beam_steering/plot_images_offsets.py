import ALPACA.data.finalize as finalize
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
import matplotlib.colors as mcolors
import seaborn as sns
import os

FIGSIZE = (18,18)

def load_data(first_run:int=492042,last_run:int=492482,bad_runs:list[int]|None=None,download_with_ALPACA:bool=False,filename:str="PbarsOnHHMCP_offsets.parquet",verbose:str=False)->pl.DataFrame:
    if download_with_ALPACA:
        data = finalize.generate(first_run=first_run,
                        last_run=last_run,
                        elog_results_filename='HHMCP',
                        known_bad_runs=[] if bad_runs is None else bad_runs,
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'Batman*acq_0*HedgeHog_MCP_Degrees',
                            'Hikrobot*acq_0*height',
                            'Hikrobot*acq_0*width',
                            'Hikrobot*acq_0*C_flatten_data',
                            'ELENA_Parameters*H_offset_mm',
                            'ELENA_Parameters*V_offset_mm',
                            'ELENA_Parameters*H_angle_mrad',
                            'ELENA_Parameters*V_angle_mrad'
                            ],
                            directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                            speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'
        if verbose:
            print(data['Run_Number*Run_Number*__value'.replace('*','_')])
            print(data['ELENA_Parameters*H_offset_mm'.replace('*','_')])
            print(data['ELENA_Parameters*V_offset_mm'.replace('*','_')])
            print(data['ELENA_Parameters*H_angle_mrad'.replace('*','_')])
            print(data['ELENA_Parameters*V_angle_mrad'.replace('*','_')])
            print(data['Batman*acq_0*HedgeHog_MCP_Degrees'.replace('*','_')])
            print(data['Hikrobot*acq_0*width'.replace('*','_')])
            print(data['Hikrobot*acq_0*height'.replace('*','_')])
            print(data['Hikrobot*acq_0*C_flatten_data'.replace('*','_')])
        
        data = pl.DataFrame(
            {
                'run':data['Run_Number*Run_Number*__value'.replace('*','_')],
                'ELENA_H_offset_mm':data['ELENA_Parameters*H_offset_mm'.replace('*','_')],
                'ELENA_V_offset_mm':data['ELENA_Parameters*V_offset_mm'.replace('*','_')],
                'ELENA_H_angle_mrad':data['ELENA_Parameters*H_angle_mrad'.replace('*','_')],
                'ELENA_V_angle_mrad':data['ELENA_Parameters*V_angle_mrad'.replace('*','_')],
                'HH_MCP_position':data['Batman*acq_0*HedgeHog_MCP_Degrees'.replace('*','_')],
                'MCP_img_width':data['Hikrobot*acq_0*width'.replace('*','_')],
                'MCP_img_height':data['Hikrobot*acq_0*height'.replace('*','_')],
                'MCP_img':data['Hikrobot*acq_0*C_flatten_data'.replace('*','_')]
            }
        )
        data.write_parquet(os.path.join(os.path.dirname(__file__),"data",filename))
    else:
        data = pl.scan_parquet(os.path.join(os.path.dirname(__file__),"data",filename))
    offsets = pl.DataFrame({
        'run':[run for run in range(492042,492483)],
        'V_offset_set':[v for v in np.linspace(-10,10,21) for _ in np.linspace(-10,10,21)],
        'H_offset_set':[h for _ in np.linspace(-10,10,21) for h in np.linspace(-10,10,21)]
    })
    data = data.join(offsets.lazy(),on='run').with_columns(((pl.col("ELENA_H_offset_mm")==pl.col("H_offset_set")) & (pl.col("ELENA_V_offset_mm")==pl.col("V_offset_set"))).alias("values set")).filter(pl.col("run").is_between(first_run,last_run)).collect()
    
    if verbose:
        print(data)
        
    return data   


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
    V_offset_to_plot = np.flip(np.arange(-10,9,2))
    H_offset_to_plot = np.arange(-10,9,2)
    
    norm = mcolors.Normalize(vmin=data.select(pl.col("MCP_img").arr.min().min()).item(),vmax=data.select(pl.col("MCP_img").arr.max().max()).item())
    fig,axes = plt.subplots(len(H_offset_to_plot),len(V_offset_to_plot),squeeze=False,sharex=True,sharey=True,figsize=FIGSIZE)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    data_filter = data.filter(pl.col("V_offset_set").is_in(V_offset_to_plot) & pl.col("H_offset_set").is_in(H_offset_to_plot))
    for run in pl.Series(data_filter.select("run")).to_list():
        run_data = data_filter.filter(pl.col("run")==run)
        print(run)
        H_offset = run_data.select('ELENA_H_offset_mm').item()
        H_offset_set = run_data.select('H_offset_set').item()
        V_offset = run_data.select('ELENA_V_offset_mm').item()
        V_offset_set = run_data.select('V_offset_set').item()
        print(f"\tH offset = {H_offset} should be {H_offset_set} ({run_data.select('values set').item()})")
        print(f"\tV offset = {V_offset} should be {V_offset_set} ({run_data.select('values set').item()})")
        
        col = np.where(H_offset_to_plot == H_offset_set)[0][0] # int((run_data.select("V_offset_set").item() + 10) % 21)
        row = np.where(V_offset_to_plot == V_offset_set)[0][0] # int((run_data.select("H_offset_set").item() + 10) % 21)
        
        ax:plt.Axes = axes[row][col]
        im,_ = plot_single_run_image(img_data=run_data.select("MCP_img").item(),
                                     img_width=run_data.select("MCP_img_width").item(),
                                     img_height=run_data.select("MCP_img_height").item(),
                                     cmap=plt.get_cmap('plasma') if run_data.select('values set').item() else plt.get_cmap("gray"),
                                     norm=norm,
                                     ax=ax)

        ax.set_xlabel(H_offset,color='black' if H_offset==H_offset_set else 'red')
        ax.set_ylabel(V_offset,color='black' if V_offset==V_offset_set else 'red')
        
        if col == len(H_offset_to_plot)-1:
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(V_offset_set)
        
        if row == 0:
            ax.set_title(H_offset_set)
    
    ax = fig.add_subplot(111, frameon=False)
    ax.set_xlabel("H offset set",labelpad=15)
    ax.set_ylabel("V offset set",labelpad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2 = fig.add_subplot(111, frameon=False)
    ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("H offset requested",labelpad=15)
    ax2.set_ylabel("V offset requested",labelpad=15)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()
    # fig.subplots_adjust(left=0.1, top=0.9,bottom=0.1,right=0.9)
    fig.savefig(os.path.join(os.path.dirname(__file__),"plots",'offset_scan_results.png'))
    plt.show()
    
def plot_single_image(data:pl.DataFrame,run:int,showfig:bool=False):
    sns.set_context("paper",font_scale=1.3)
    cmap = plt.get_cmap('magma')
    norm = mcolors.Normalize(vmin=data.select(pl.col("MCP_img").arr.min().min()).item(),vmax=data.select(pl.col("MCP_img").arr.max().max()).item())
    fig,ax = plt.subplots(figsize=FIGSIZE)
    run_data = data.filter(pl.col("run")==run)
        
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
    # ax.set_title(f'{run},V_OFFSET={run_data.select("V_offset_set").item()},H_OFFSET={run_data.select("H_offset_set").item()} ({run_data.select("values set").item()})')
    ax.set_title(f'{run}, V_OFFSET={run_data.select("ELENA_V_offset_mm").item()} (requested {run_data.select("V_offset_set").item()}), H_OFFSET={run_data.select("ELENA_H_offset_mm").item()}  (requested {run_data.select("H_offset_set").item()})')
    
    
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__),"plots",f'{run}-V_OFFSET={run_data.select("V_offset_set").item()}-H_OFFSET={run_data.select("H_offset_set").item()}.png'))
    if showfig:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    print("Load data")
    data = load_data()
    print(data)
    # plot_all_images(data)
    for run in range(492042,492483):
        try:
            print(run)
            plot_single_image(data,run,showfig=False)
        except Exception as e:
            print(type(e))
            print(e)