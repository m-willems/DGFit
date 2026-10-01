import argparse

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib

from astropy.io import fits

from dgfit.obsdata import ObsData


def get_krange(x, logaxis=False, in_range=[0]):
    prange = np.array([0.0, 0.0])
    if logaxis:
        gindxs = x > 0
        min_x = np.amin(x[gindxs])
        max_x = np.amax(x[gindxs])
    else:
        min_x = np.amin(x)
        max_x = np.amax(x)

    if logaxis:
        max_x = np.log10(max_x)
        if min_x <= 0.0:
            min_x = max_x - 10.0
        else:
            min_x = np.log10(min_x)

    delta = max_x - min_x
    prange[0] = min_x - 0.1 * delta
    prange[1] = max_x + 0.1 * delta

    if logaxis:
        prange = np.power(10.0, prange)

    if len(in_range) > 1:
        prange[0] = np.minimum(prange[0], in_range[0])
        prange[1] = np.maximum(prange[1], in_range[1])

    return prange


def main():
    # commandline parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "obsfile", help="Data file giving the observational data that was fit"
    )
    parser.add_argument(
        "dustproperty",
        help="What dust property needs to be shown for both model and data",
        choices=["emission", "extinction", "albedo", "g"],
    )
    parser.add_argument(
        "--filename",
        nargs="+",
        help=(
            "file with the dust model details "
            + "(size distribution, extinction, etc.)"
        ),
    )
    parser.add_argument(
        "--start", help="include the starting model", action="store_true"
    )
    parser.add_argument(
        "--inverse_lambda",
        action="store_true",
        help="put the extinction plot in units 1/lambda",
    )
    parser.add_argument(
        "--no_ylogscale", action="store_true", help="don't put the yaxis in logscale"
    )
    parser.add_argument(
        "--add_fitted_line",
        action="store_true",
        help="plot the fitted line to the data, only available for albedo and g",
    )
    parser.add_argument(
        "-p", "--png", help="save figure as a png file", action="store_true"
    )
    parser.add_argument(
        "-e", "--eps", help="save figure as an eps file", action="store_true"
    )
    parser.add_argument("-pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # setup the plots
    fontsize = 16
    font = {"size": fontsize}

    matplotlib.rc("font", **font)
    matplotlib.rc("lines", linewidth=2)
    matplotlib.rc("axes", linewidth=2)
    matplotlib.rc("xtick.major", width=2)
    matplotlib.rc("ytick.major", width=2)

    fig, ax = pyplot.subplots(
        ncols=1,
        nrows=5,
        figsize=(10, 15),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 1], "hspace": 0},
    )
    ax1 = ax[0]
    colors = ["r", "b", "g", "c"]
    markers = ["^", "o", "x", "v"]
    ltype = "-"

    # open the DGFit results
    files = []
    for name in args.filename:
        files.append(name)
    j = 0

    for file in files:
        hdulist = fits.open(file)

        comps = []
        for i in range(hdulist[0].header["NCOMPS"]):
            hdu = hdulist[i + 1]
            comps.append(hdu.header["EXTNAME"])

        hdu = hdulist[args.dustproperty.upper()]

        xlogscale = True
        ylogscale = True

        # get the observed data
        OD = ObsData(args.obsfile)
        rechte = []
        models = ["WD01", "ZDA04", "HD23", "Y24"]

        if args.dustproperty == "emission":
            waves = OD.ir_emission_waves
            data_waves = hdu.data["WAVE"]
            data = OD.ir_emission_av
            data_unc = OD.ir_emission_av_unc
            data_name = "EMIS"
            ylabel = r"$S$ $[MJy$ $sr^{-1}$ $A(V)^{-1}]$"
            if args.no_ylogscale:
                ylogscale = False
            ylim = False
            mark = 5

        elif args.dustproperty == "extinction":
            waves = OD.ext_waves
            data_waves = hdu.data["WAVE"]
            if args.inverse_lambda:
                waves = 1 / waves
                data_waves = 1 / data_waves
                xlogscale = False
            data = OD.ext_alav
            data_unc = OD.ext_alav_unc
            data_name = "EXT"
            ylabel = r"$A(\lambda)/A(V)$"
            if args.no_ylogscale:
                ylogscale = False
            ylim = False
            mark = 10

        elif args.dustproperty == "albedo":
            waves = OD.scat_a_waves
            data_waves = hdu.data["WAVE"]
            data = OD.scat_albedo
            data_unc = OD.scat_albedo_unc
            data_name = args.dustproperty.upper()
            ylabel = "Albedo"
            ylogscale = False
            xlogscale = False
            ylim = True
            for wave in data_waves:
                b = (1.0027468894049667 * wave) + 0.2950590005599959
                rechte.append(b)
            c = [
                0.484,
                0.550,
                0.595,
                0.614,
                0.615,
                0.606,
                0.597,
                0.497,
            ]  # include the Drude profile
            rechte[11] = 1 - c[1]
            rechte[12] = 1 - c[2]
            rechte[13] = 1 - c[3]
            rechte[14] = 1 - c[4]
            rechte[16] = 1 - c[5]
            rechte[17] = 1 - c[6]
            rechte[19] = 1 - c[7]
            rechte[15] = (rechte[14] + rechte[16]) / 2
            rechte[18] = (rechte[17] + rechte[19]) / 2
            mark = 1

        elif args.dustproperty == "g":
            waves = OD.scat_g_waves
            data_waves = hdu.data["WAVE"]
            data = OD.scat_g
            data_unc = OD.scat_g_unc
            data_name = args.dustproperty.upper()
            ylabel = "g"
            xlogscale = False
            ylogscale = False
            ylim = True
            for wave in data_waves:
                b = (0.08768592371741601 * wave) + 0.6176183611711199
                rechte.append(b)
            mark = 1

        ax1.plot(
            data_waves,
            hdu.data[data_name],
            colors[j] + ltype,
            marker=markers[j],
            label=models[j],
            markevery=mark,
        )

        residuals = (data - hdu.data[data_name]) / data
        unc = data_unc / data
        if args.dustproperty in ["albedo", "g"]:
            ax[j + 1].errorbar(
                data_waves,
                residuals,
                yerr=unc,
                fmt="o",
                color="black",
                capsize=3,
                label=models[j],
            )
        else:
            ax[j + 1].plot(
                data_waves,
                residuals,
                "-",
                color="black",
                label=models[j],
            )
            ax[j + 1].fill_between(
                waves,
                residuals - unc,
                residuals + unc,
                color="k",
                alpha=0.3,
                label="Uncertainty",
            )
        ax[j + 1].axhline(0, color=colors[j], linestyle="--", linewidth=1)
        if args.inverse_lambda:
            ax[j + 1].set_xlabel(r"$1/\lambda \ [1/\mu m]$", fontsize=fontsize)
        else:
            ax[j + 1].set_xlabel(r"$\lambda \ [\mu m]$", fontsize=fontsize)
        ax[j + 1].set_ylim(-0.75, 0.75)
        # ax[j+1].legend(loc="lower right")
        # ax[j+1].annotate(f"{models[j]}", (3000, -0.5), fontsize=20, color="black")

        j += 1
    ax[1].annotate(f"{models[0]}", (0.2, 0.4), fontsize=20, color="black")
    ax[2].annotate(f"{models[1]}", (0.2, -0.4), fontsize=20, color="black")
    ax[3].annotate(f"{models[2]}", (0.2, 0.4), fontsize=20, color="black")
    ax[4].annotate(f"{models[3]}", (0.2, 0.4), fontsize=20, color="black")

    ax[3].set_ylabel("Residuals\n (data - model)/data", fontsize=fontsize)

    if args.dustproperty in ["albedo", "g"]:
        ax1.errorbar(
            waves, data, data_unc, fmt="ko", label="Observed", capsize=3, markevery=mark
        )
    else:
        ax1.plot(waves, data, "-", label="Observed", color="black", markevery=mark)
        ax1.fill_between(
            waves,
            data - data_unc,
            data + data_unc,
            color="k",
            alpha=0.3,
            label="Uncertainty",
        )

    if args.add_fitted_line:
        ax1.plot(data_waves, rechte, "darkgrey", label="Fit")

    if ylogscale:
        ax1.set_yscale("log")
    if xlogscale:
        ax1.set_xscale("log")

    if args.inverse_lambda:
        ax1.set_xlabel(r"$1/\lambda \ [1/\mu m]$", fontsize=fontsize)
    else:
        ax1.set_xlabel(r"$\lambda \ [\mu m]$", fontsize=fontsize)

    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.legend(loc="lower right")
    ax1.set_xlim(get_krange(data_waves, logaxis=xlogscale))
    ax1.set_ylim(get_krange(data, logaxis=ylogscale))
    if ylim:
        ax1.set_ylim([0.0, 1.0])

    for axis in ax:
        axis.tick_params(
            which="both",  # apply to both major and minor ticks
            direction="in",  # put ticks inside the axes
            # top=True,
            right=True,  # ticks on all sides
            # labelbottom=True   # show x labels on all axes
        )

    ax[4].tick_params(
        which="both",  # apply to both major and minor ticks
        direction="in",  # put ticks inside the axes
        # top=True,
        right=True,  # ticks on all sides
        labelbottom=True,  # show x labels on all axes
    )

    pyplot.tight_layout()

    pyplot.show()

    # show or save
    basename = f"{args.dustproperty}_models"
    if args.png:
        fig.savefig(basename + ".png")
    elif args.eps:
        fig.savefig(basename + ".eps")
    elif args.pdf:
        fig.savefig(basename + ".pdf")
