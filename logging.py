import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

class Logger:

    def __init__(self, logging_columns=[]):
        self.data_frame = pd.DataFrame(columns=logging_columns).rename_axis(index="epochs", columns="metrics")
        self.logging_columns = logging_columns
        sb.set_palette("deep")

    def log(self, **kwargs):
        data = {}
        for k in self.logging_columns:
            data[k] = kwargs[k]
        if self.data_frame.index.size == 0:
            self.data_frame.loc[0] = data
        else:
            self.data_frame.loc[self.data_frame.index[-1] + 1] = data
    
    def create_lineplot(self, columns_to_plot=None):
        if columns_to_plot is None:
            columns_to_plot = self.logging_columns

        fig = sb.lineplot(
            data=self.data_frame[columns_to_plot], dashes=False
        )

        return fig

        