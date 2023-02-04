"""
Function that displays a grid of crowd movies and plotted bokeh figures of position heatmaps.
"""

from bokeh.io import show
from bokeh.models import Div
import ipywidgets as widgets
from bokeh.layouts import gridplot
from IPython.display import display

def display_crowd_movies(widget_box, curr_name, desc, divs, bk_figs):
    """
    display the crowd movies in jupyter notebook.

    Args:
    divs (list of bokeh.models.Div): list of HTML Div objects containing videos to display

    Returns:
    """

    # Set HTML formats
    movie_table = """
                    <html>
                    <head>
                    <style>
                        .output {
                            display: contents;
                            height: auto;
                        }
                        .row {
                            display: flex;
                            flex-wrap: wrap;
                            vertical-align: center;
                            width: 900px;
                            text-align: center;
                        }

                        .column {
                            width: 50%;
                            text-align: center;
                        }

                        .column {
                          vertical-align: center;
                        }

                        table {
                            display: inline-block;
                        }

                        h3 {
                            text-align: center;
                        }
                    </style>
                    </head>""" + \
                  f"""
                    <body>
                    <h3>Name: {curr_name}</h3>
                    <h3>Description: {desc}</h3>
                    <br>
                    <div class="row"; style="background-color:#ffffff; height:auto;">
                  """

    # Create div grid
    for i, div in enumerate(divs):
        if (i % 2 == 0) and i > 0:
            # make a new row
            movie_table += '</div>'
            col = f"""
                      <div class="row"; style="background-color:#ffffff; height:auto;">
                          <div class="column">
                              {div}
                          </div>
                    """
        else:
            # put movie in column
            col = f"""
                      <div class="column">
                          {div}
                      </div>
                    """
        movie_table += col

    # Close last div
    movie_table += '</div>\
                    </body>\
                    </html>'

    div2 = Div(text=movie_table)

    # Display
    display(widget_box)
    show(div2)

    if len(bk_figs) > 0:
        gp = gridplot(bk_figs, ncols=2, plot_width=250, plot_height=250)

        # Create Output widget object to center grid plot in view
        output = widgets.Output(layout=widgets.Layout(align_items='center'))
        with output:
            show(gp)

        display(output)