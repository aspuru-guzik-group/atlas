import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def process_cycles(df):
    df = df[-0.75 < df["V"]]
    df = df[df["V"] < 0.75]

    df["c"] = df["c"].astype(int)
    # Get Red Ox
    (_, reduction), (_, oxidation) = df.groupby(
        np.diff(df["V"].array, append=0) > 0
    )

    fp_cfg = partial(find_peaks, width=10, distance=10, rel_height=0.9)

    oxi_peaks = map(
        lambda xs: xs[1][["V", "I"]].iloc[fp_cfg(xs[1]["I"])[0]],
        oxidation.groupby("c"),
    )
    red_peaks = map(
        lambda xs: xs[1][["V", "I"]].iloc[fp_cfg(-xs[1]["I"])[0]],
        reduction.groupby("c"),
    )
    return tuple(oxi_peaks), tuple(red_peaks)


if __name__ == "__main__":
    HEADER = ("I", "V", "R", "T", "h", "t", "c")
    results = pd.read_csv(sys.argv[1], names=HEADER)
    results = results[results["c"] >= 4]
    results["I"] *= 1e6
    results["smoothed_I"] = gaussian_filter1d(results["I"], sigma=2)

    oxi_peaks, _ = process_cycles(results)

    def checknum(x):
        y = x["I"].argmax()
        if pd.isna(y):
            return -1
        return x.iloc[y]

    oxivalues = pd.concat(
        map(checknum, oxi_peaks), axis=1, ignore_index=True
    ).T
    max_val = oxivalues.iloc[np.argmax(oxivalues["V"])]

    line = sns.lineplot(
        data=results, x="V", y="I", sort=False, color="#477A7B"
    )
    sns.scatterplot(
        x=max_val["V"], y=[max_val["I"]], color="#8F5652", s=15, marker="X"
    )
    line.set(xlabel="Volage / V")
    line.set(ylabel="Current / μA")
    fig = line.get_figure()
    ax = fig.axes[0]
    # ax.set(ylim=[1.094+0.000650, 1.094+0.000750])
    fig.set_size_inches(2.4, 1.8)
    fig.tight_layout()
    fig.savefig(Path(sys.argv[1]).stem + ".svg")
    plt.clf()
