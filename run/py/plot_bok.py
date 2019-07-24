
from sklearn import decomposition
from sklearn.manifold import TSNE
import numpy as np
import random
from sklearn import metrics


import uuid
import pandas as pd
import sys
import bokeh 
from bokeh.plotting import figure
from bokeh.io import output_notebook,output_file, show, reset_output
from bokeh.models import ColumnDataSource,  LabelSet
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from bokeh.models import LinearColorMapper, FixedTicker, FuncTickFormatter, ColorBar

from bokeh.models import ColumnDataSource, HoverTool,BoxZoomTool, ResetTool, WheelZoomTool, LinearColorMapper,LogColorMapper, CategoricalColorMapper, Select, CustomJS,  LogTicker, ColorBar, BasicTicker



def bokey_plot(dictionary_input, folder_bokey_default,mode="single", output=False, id_=None,info="",color_map_mode="continuous"):
    """
    as input dictionnary of the form
    - label : {x:, y: , label:[labels]}
    mode allows you to specify if you want everything on one single plot or if you want distinct plots
    """
    reset_output()
    source = {}
    i = 0
    color = ["blue","#ee6666"]
    assert color_map_mode in ["continuous","divergent"]
    if color_map_mode=="continuous": color_map= cm.OrRd
    elif color_map_mode=="divergent": color_map=cm.rainbow
    if id_ is None:
        id_ = str(uuid.uuid4())[0:8]

    if mode == "single":
        p = figure(plot_width=800, plot_height=1000, 
        		   tools=[BoxZoomTool(), ResetTool(), WheelZoomTool()],
                   toolbar_sticky=False, toolbar_location="right",
                   title='T-SNE '+info)  # x_range=Range1d(-6,6),y_range=Range1d(int(min(y))-1,int(max(y))+1))

    for key in dictionary_input.keys():
        if mode == "distinct":
            p = figure(plot_width=800, plot_height=1000,
                       title='T-SNE '+info)  # x_range=Range1d(-6,6),y_range=Range1d(int(min(y))-1,int(max(y))+1))

        source[key] = ColumnDataSource(data=dict(height=dictionary_input[key]["x"],
                                                 weight=dictionary_input[key]["y"],
                                                 names=dictionary_input[key]["label"]))

        colors = ["#%02x%02x%02x" %(int(r), int(g), int(b)) for r, g, b, _ in (255)*color_map(Normalize(vmin=0,vmax=5)(dictionary_input[key]["color"]))]
        colors_legend = ["#%02x%02x%02x" %(int(r), int(g), int(b)) for r, g, b, _ in (255)*color_map(Normalize(vmin=0,vmax=5)(np.sort(list(set(dictionary_input[key]["color"])))))]

        color_mapper = LinearColorMapper(palette=colors_legend)
        ticker = FixedTicker(ticks=[0, 1, 2, 3, 4,5])
        formatter = FuncTickFormatter(code="""
                                        function(tick) {
                                            data = {0: '0-10', 1: '10-20', 2: '20-30', 3: '30-40', 4: '40-50',50: '50plus'}
                                            return data[tick], " ,
                                        }
                                        """)

        cbar = ColorBar(color_mapper=color_mapper, ticker=ticker, formatter=formatter,
                        major_tick_out=0, major_tick_in=0, major_label_text_align='left',
                        major_label_text_font_size='100pt', label_standoff=5)

        p.scatter(x='weight', y='height', size=8, source=source[key], legend=key)# color=colors)
        p.add_layout(cbar)

        labels = LabelSet(x='weight', y='height', text='names', level='glyph',
                          x_offset=5, y_offset=5, source=source[key], render_mode='canvas')
        p.add_layout(labels)
        i += 1
        if output:
            output_file(folder_bokey_default+id_+"_tsne_"+key+"_"+info+"_bok.html")
            print(folder_bokey_default+id_+"_tsne_"+key+"_"+info+"_bok.html")
            show(p)
        #if mode == "distinct":
        #    output_notebook()
        #    show(p)
    if mode == "single":
        output_notebook()
        show(p)