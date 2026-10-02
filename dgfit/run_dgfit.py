import importlib.resources as importlib_resources
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import h5py

from scipy.optimize import minimize
import emcee
from multiprocessing import Pool
import corner
from nautilus import Prior
from nautilus import Sampler
from scipy.stats import loguniform

from dgfit.dustmodel import (
    DustModel,
    MRN77DustModel,
    WD01DustModel,
    ZDA04DustModel,
    HD23DustModel,
    Y24DustModel,
    Lognormal,
)
from dgfit.obsdata import ObsData
from dgfit.dustclasses import DustCompositions
from dgfit.prior_transform import LogUniformLin


def DGFit_cmdparser():
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "obsfile", help="Data file giving the observational data to be fit"
    )
    parser.add_argument(
        "--sizedisttype",
        default="WD01",
        choices=["bins", "MRN77", "WD01", "ZDA04", "HD23", "Y24", "lognormals"],
        help="Size distribution type",
    )
    parser.add_argument(
        "--fitobs",
        nargs="+",
        default="all",
        choices=["extinction", "iremission", "abundance", "albedo", "g", "all"],
        help="Which observations to fit",
    )

    parser.add_argument(
        "--composition",
        nargs="+",
        default=["Carbonaceous-LD01", "Silicates-DL84"],
        choices=DustCompositions.all_compositions,
        help="Which grains to use",
    )

    parser.add_argument(
        "--no_variable_ISRF",
        action="store_false",
        help="disable the variable radiation field",
    )

    parser.add_argument(
        "--mcmc", help="Do MCMC sampling using emcee package", action="store_true"
    )

    parser.add_argument(
        "-f",
        "--fast",
        help="MCMC: Use minimal walkers, steps, burns to debug code",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--slow",
        help="MCMC: Use lots of walkers, n_steps, n_burn",
        action="store_true",
    )
    parser.add_argument(
        "--burnfrac",
        type=float,
        default=0.1,
        help="MCMC: Fractional portion of nsteps for burn in",
    )
    parser.add_argument(
        "--nsteps", type=int, default=1000, help="MCMC: Number of samples for full run"
    )
    parser.add_argument(
        "--everynth", type=int, default=2, help="Use every nth grain size"
    )
    parser.add_argument(
        "--chain", action="store_true", help="MCMC: Store the chain in an ascii file"
    )
    parser.add_argument(
        "--limit_abund",
        action="store_true",
        help="Hard limit on size distribution based on abundances",
    )
    parser.add_argument(
        "--usemin",
        action="store_true",
        help="MCMC: Find min before EMCEE",
    )
    parser.add_argument(
        "-r", "--read", default=None, help="Read size distribution from disk"
    )
    parser.add_argument(
        "-t", "--tag", default="GrainBow_test", help="basename to use for output files"
    )
    parser.add_argument(
        "--nolarge",
        action="store_true",
        help="Deweight sizes bigger than the cutoff size",
    )
    parser.add_argument(
        "--cutoff", nargs="+", default=[5.0], help="The cutoff size in micron"
    )
    parser.add_argument(
        "--weight_by_average_unc",
        action="store_true",
        default=False,
        help="weight the observations by the average uncertainty, divide by number of points",
    )
    parser.add_argument(
        "--start_ISRF", type=int, default=1, help="Strength of ISRF to start"
    )
    parser.add_argument(
        "--regularization",
        action="store_true",
        default=False,
        help="add a smoothness criterium to the size distribution",
    )
    parser.add_argument(
        "--fitting_package",
        default="nautilus",
        choices=["nautilus", "emcee"],
        help="Fitting package to use",
    )
    parser.add_argument(
        "--cornerplot",
        action="store_true",
        default=False,
        help="Generate a corner plot of the posterior distributions",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Nautilus: Allows you to run in parallel",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=4,
        help="Nautilus: Number of cores to use if you run parallel",
    )
    parser.add_argument(
        "--nlivepoints",
        type=int,
        default=2000,
        help="Nautilus: Number of live points to use",
    )
    parser.add_argument(
        "--result_from_file",
        default="none",
        help="Nautilus: give the name of the file with the parameters",
    )
    parser.add_argument(
        "--abundance_factor",
        action="store_true",
        help="Add an extra parameter to calculate the abundance fractions",
    )
    parser.add_argument(
        "--start_size", nargs="+", default=["none"], help="The start size in micron"
    )

    return parser


def set_obs_for_fitting(obsdata, fitobs):
    """
    parse the requested list of observations for fitting and set the
    appropriate variables
    """

    fitobs_list = []
    if not (obsdata.fit_extinction and (("extinction" in fitobs) or ("all" in fitobs))):
        obsdata.fit_extinction = False
    else:
        fitobs_list.append("extinction")
    if not (obsdata.fit_abundance and (("abundance" in fitobs) or ("all" in fitobs))):
        obsdata.fit_abundance = False
    else:
        fitobs_list.append("abundance")
    if not (
        obsdata.fit_ir_emission and (("iremission" in fitobs) or ("all" in fitobs))
    ):
        obsdata.fit_ir_emission = False
    else:
        fitobs_list.append("ir_emission")
    if not (obsdata.fit_scat_a and (("albedo" in fitobs) or ("all" in fitobs))):
        obsdata.fit_scat_a = False
    else:
        fitobs_list.append("scat albedo")
    if not (obsdata.fit_scat_g and (("g" in fitobs) or ("all" in fitobs))):
        obsdata.fit_scat_g = False
    else:
        fitobs_list.append("scat g")

    return fitobs_list


def calc_sizedist_fact(dustmodel, obsdata):

    results = dustmodel.eff_grain_props(obsdata)
    natoms = results["natoms"]
    factor_C = 1
    factor_sil = 1
    for atomname in natoms.keys():
        if natoms[atomname] > (
            obsdata.abundance_av[atomname][0] + obsdata.abundance_av[atomname][1]
        ):
            if atomname == "C":
                newfactor_C = natoms[atomname] / obsdata.abundance_av[atomname][0]
                if newfactor_C > factor_C:
                    factor_C = newfactor_C
            else:
                newfactor_sil = natoms[atomname] / obsdata.abundance_av[atomname][0]
                if newfactor_sil > factor_sil:
                    factor_sil = newfactor_sil

    return factor_C, factor_sil


# ============================================================================
# PARAMETER SPECIFICATION DATA FOR ALL SIZE DISTRIBUTION TYPES
# ============================================================================

PARAM_SPECS = {
    "MRN77": {
        "all_carbonaceous": {
            "use_factor": "factor_C",
            "params": ["C", "alpha", "a_min", "a_max"],
            "indexed": True,
        },
        "all_silicates": {
            "use_factor": "factor_sil",
            "params": ["C", "alpha", "a_min", "a_max"],
            "indexed": True,
        },
    },
    "WD01": {
        "Silicates-DL84": {
            "use_factor": "factor_sil",
            "params": ["C_s", "a_ts", "alpha_s", "beta_s"],
            "indexed": False,
        },
        "Carbonaceous-LD01": {
            "use_factor": "factor_C",
            "params": ["C_g", "a_tg", "alpha_g", "beta_g", "a_cg", "b_C"],
            "indexed": False,
        },
    },
    "ZDA04": {
        "Carbonaceous-LD01": {
            "use_factor": "factor_C",
            "params": ["A", "c_0", "b_0", "b_1", "m_1", "a_3", "m_3"],
            "indexed": True,
        },
        "Graphite-LD93": {
            "use_factor": "factor_C",
            "params": [
                "A",
                "c_0",
                "b_0",
                "b_1",
                "a_1",
                "m_1",
                "b_3",
                "a_3",
                "m_3",
                "b_4",
                "a_4",
                "m_4",
            ],
            "indexed": True,
        },
        "Silicates-DL84": {
            "use_factor": "factor_sil",
            "params": [
                "A",
                "c_0",
                "b_0",
                "b_1",
                "a_1",
                "m_1",
                "b_3",
                "a_3",
                "m_3",
                "b_4",
                "a_4",
                "m_4",
            ],
            "indexed": True,
        },
        "amC-ACH2-Z96": {
            "use_factor": "factor_C",
            "params": ["A", "c_0", "b_0", "b_1", "a_1", "m_1"],
            "indexed": True,
        },
    },
    "HD23": {
        "Carbonaceous-DL07": {
            "use_factor": "factor_C",
            "params": ["B_1", "B_2"],
            "indexed": False,
        },
        "AstroDust-DH21": {
            "use_factor": "factor_sil",
            "params": [
                "B_ad",
                "a_0",
                "sigma_ad",
                "A_0",
                "A_1",
                "A_2",
                "A_3",
                "A_4",
                "A_5",
            ],
            "indexed": False,
        },
    },
    "Y24": {
        "a-C-J16": {
            "use_factor": "factor_C",
            "params": ["A", "alpha", "a_C", "a_t", "gamma"],
            "indexed": "partial",  # Only A is indexed, others are not
        },
        "a-C:H-J16": {
            "use_factor": "factor_C",
            "params": ["A", "a_0", "sigma"],
            "indexed": True,
        },
        "X35-X50B-D22": {
            "use_factor": "factor_sil",
            "params": ["A", "a_0", "sigma"],
            "indexed": True,
        },
    },
    "lognormals": {
        "all_carbonaceous": {
            "use_factor": "factor_C",
            "params": ["A_s", "sigma_s", "a_s", "A_b", "sigma_b", "a_b"],
            "indexed": True,
        },
        "all_silicates": {
            "use_factor": "factor_sil",
            "params": ["A_s", "sigma_s", "a_s", "A_b", "sigma_b", "a_b"],
            "indexed": True,
        },
    },
}


# ============================================================================
# GENERIC PARAMETER SETUP FUNCTION (replaces all 5 setparams_* functions)
# ============================================================================


def setparams_generic(
    dustmodel, obsdata, factor_C, factor_sil, ISRF, f_abund, sizedisttype
):
    """
    Generic parameter setup function for all size distribution types.

    Replaces setparams_MRN77, setparams_WD01, setparams_ZDA04,
    setparams_HD23, and setparams_Y24.
    """
    pnames = []
    p0 = []
    prior_ranges = []
    logs = []

    specs = PARAM_SPECS.get(sizedisttype, {})

    for i, component in enumerate(dustmodel.components):
        comp_name = component.name

        # Get the spec for this component
        if comp_name in specs:
            spec = specs[comp_name]
        elif "all_carbonaceous" in specs and comp_name in dustmodel.carbonaceous_names:
            spec = specs["all_carbonaceous"]
        elif "all_silicates" in specs and comp_name in dustmodel.silicate_names:
            spec = specs["all_silicates"]
        else:
            raise ValueError(
                f"No parameter spec found for {comp_name} in {sizedisttype}"
            )

        # Determine which factor to use for this component
        use_factor_name = spec["use_factor"]
        use_factor = factor_C if use_factor_name == "factor_C" else factor_sil

        # Extract component parameters and prior ranges
        cparams = dustmodel.parameters[comp_name]
        cprior_ranges = dustmodel.prior_ranges[comp_name]
        clogs = dustmodel.logs[comp_name]

        # Check if parameters are indexed by component loop index
        is_indexed = spec.get("indexed", False)

        # Build parameter list from spec
        param_keys = spec["params"]

        for param_key in param_keys:
            # Construct actual parameter key (with indexing if needed)
            if is_indexed == "partial":
                # For partial indexing (e.g., a-C-J16), only "A" parameters are indexed
                if param_key == "A":
                    actual_key = f"{param_key}_{i}"
                else:
                    actual_key = param_key
            elif is_indexed:
                actual_key = f"{param_key}_{i}"
            else:
                actual_key = param_key

            # Get value
            param_value = cparams[actual_key]

            # Apply abundance factor correction if this is an amplitude parameter
            param_base = actual_key.split("_")[0]
            if param_base in ["A", "C", "B"]:
                param_value = param_value / use_factor

            p0.append(param_value)

            # Get prior range with abundance factor correction
            prior_range = cprior_ranges[actual_key]
            if isinstance(prior_range, (list, tuple)):
                if param_base in ["A", "C", "B"]:
                    prior_range = [r / use_factor for r in prior_range]
                prior_ranges.append(np.array(prior_range))
            else:
                prior_ranges.append(prior_range)

            # Get log flag
            logs.append(clogs[actual_key])
            pnames.append(actual_key)

        # Add abundance fraction parameter if enabled
        if f_abund:
            abund_key = f"F_a_{i}"
            if abund_key in cparams:
                p0.append(cparams[abund_key])
            else:
                p0.append(1.0)
            prior_ranges.append(np.array([0.2, 5.0]))
            logs.append(False)
            pnames.append(abund_key)

    # Add ISRF parameter if enabled
    if ISRF:
        cparams = dustmodel.parameters["Radiation field"]
        p0.append(cparams["RF"])
        prior_ranges.append(np.array([0.0001, 30]))
        logs.append(False)
        pnames.append("RF")

    return p0, prior_ranges, logs, pnames


def add_priors_nautilus(pnames, logs, prior_ranges, prior):
    for i, name in enumerate(pnames):
        if name == "RF":
            prior.add_parameter("RF", dist=(0.0001, 30))
            logs.append(False)

        elif "F_a_" in name:
            prior.add_parameter(f"{name}", dist=(0.2, 5))
            logs.append(False)

        else:
            if logs[i]:
                prior.add_parameter(
                    f"{name}", dist=loguniform(prior_ranges[i][0], prior_ranges[i][1])
                )
            else:
                prior.add_parameter(
                    f"{name}", dist=(prior_ranges[i][0], prior_ranges[i][1])
                )


def sample_prior(prior_ranges, logs, n_samples=1000):
    """
    Generate samples from the same priors used by Nautilus.
    """

    n_params = len(prior_ranges)

    samples = np.zeros((n_samples, n_params))

    for i in range(n_params):

        lower, upper = prior_ranges[i]

        if logs[i]:
            # log-uniform prior
            samples[:, i] = 10 ** np.random.uniform(
                np.log10(lower),
                np.log10(upper),
                n_samples
            )

        else:
            # uniform prior
            samples[:, i] = np.random.uniform(
                lower,
                upper,
                n_samples
            )

    return samples


def main():
    parser = DGFit_cmdparser()

    args = parser.parse_args()

    # set the basename of the output
    basename = f"{args.tag}_{args.sizedisttype}"

    # save the start time
    start_time = time.process_time()

    if args.fitting_package == "emcee":
        # emcee parameters
        if args.fast:
            print("using the fast params")
            nsteps = 100
            burnfrac = 0.1
        elif args.slow:
            print("using the slow params")
            nsteps = 10000
            burnfrac = 0.2
        else:
            burnfrac = float(args.burnfrac)
            nsteps = int(args.nsteps)

    # get the location of the provided data
    ref = importlib_resources.files("dgfit") / "data"

    # get the observed data
    obsdata = ObsData(args.obsfile)

    # determine what to fit based on what exists and the commandline args
    fitobs_list = set_obs_for_fitting(obsdata, args.fitobs)

    # get the dust model on the full wavelength grid
    compnames = list(args.composition)
    with importlib_resources.as_file(ref) as data_path:
        dustmodel_full = DustModel(
            componentnames=compnames,
            path=str(data_path) + "/indiv_grain/",
            every_nth=args.everynth,
            limit_abundances=args.limit_abund,
            variable_ISRF=args.no_variable_ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            regularization=args.regularization,
        )
    for i, comp in enumerate(compnames):
        print(
            f"# of grain sizes for {comp} = {len(dustmodel_full.components[i].sizes)}"
        )

    prior = Prior()
    sizedisttype = args.sizedisttype
    ISRF = args.no_variable_ISRF
    f_abund = args.abundance_factor
    pnames = []

    # Create the appropriate DustModel based on sizedisttype
    if sizedisttype == "MRN77":
        dustmodel = MRN77DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            abundance_factor=f_abund,
        )
    elif sizedisttype == "WD01":
        dustmodel = WD01DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            abundance_factor=f_abund,
        )
    elif sizedisttype == "ZDA04":
        dustmodel = ZDA04DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            abundance_factor=f_abund,
        )
    elif sizedisttype == "HD23":
        dustmodel = HD23DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            abundance_factor=f_abund,
        )
    elif sizedisttype == "Y24":
        dustmodel = Y24DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            abundance_factor=f_abund,
        )
    elif sizedisttype == "lognormals":
        dustmodel = Lognormal(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            abundance_factor=f_abund,
        )
    elif sizedisttype == "bins":
        # bins dustmodel will be created in the parameter setup section
        pass
    elif args.read is not None:
        dustmodel = dustmodel_full
    else:
        print("Size distribution choice not known")
        exit()

    # Setup parameters using generic function for non-bins cases
    if sizedisttype in ["MRN77", "WD01", "ZDA04", "HD23", "Y24", "lognormals"]:
        p0, prior_ranges, logs, _pnames = setparams_generic(
            dustmodel, obsdata, 1, 1, ISRF, f_abund, sizedisttype
        )
        pnames += _pnames
        dustmodel.set_size_dist(p0)

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)
            p0, prior_ranges, logs, _pnames = setparams_generic(
                dustmodel,
                obsdata,
                factor_C,
                factor_sil,
                ISRF,
                f_abund,
                sizedisttype,
            )
            dustmodel.set_size_dist(p0)

        if args.fitting_package == "nautilus":
            add_priors_nautilus(pnames, logs, prior_ranges, prior)

    # replace the default size distribution with one from a file
    elif args.read is not None:
        dustmodel.read_sizedist_from_file(args.read)

    elif sizedisttype == "bins":
        if args.fitting_package != "nautilus":
            print("Only the nautilus sampling package is supported for bins")
            exit()
        dustmodel = DustModel(
            dustmodel=dustmodel_full,
            obsdata=obsdata,
            limit_abundances=args.limit_abund,
            variable_ISRF=ISRF,
            divide_npoints=args.weight_by_average_unc,
            start_ISRF=args.start_ISRF,
            regularization=args.regularization,
            abundance_factor=f_abund,
        )

        if args.limit_abund:
            factor_C, factor_sil = calc_sizedist_fact(dustmodel, obsdata)

        eps = 1e-6
        p0 = []
        logs = []
        prior_ranges = []
        lowers = []
        uppers = []
        used_sizes = []
        n_deweighted_comp = []
        for k in range(0, dustmodel.n_components):
            n_deweighted = 0

            # Calculate the upper limit of the priors
            n_grains = dustmodel.calculate_priors(dustmodel.components[k], obsdata)
            for kk in range(len(dustmodel.components[k].size_dist)):
                if args.nolarge:
                    if dustmodel.components[k].sizes[kk] > (
                        float(args.cutoff[k]) * 1e-4
                    ):
                        print(
                            "Deweighting size",
                            np.round(
                                dustmodel.components[k].sizes[kk] * 10000, decimals=4
                            ),
                            "microns for component",
                            k + 1,
                        )
                        prior.add_parameter(f"c{k + 1}_s{kk + 1}", dist=0)
                        p0.append(0)
                        lowers.append(0)
                        uppers.append(0)
                        n_deweighted += 1
                        continue

                if args.start_size[0] != "none":
                    if (dustmodel.components[k].sizes[kk] * 10000) < float(
                        args.start_size[k]
                    ):
                        print(
                            "Deweighting size",
                            np.round(
                                dustmodel.components[k].sizes[kk] * 10000, decimals=4
                            ),
                            "microns for component",
                            k + 1,
                        )
                        prior.add_parameter(f"c{k + 1}_s{kk + 1}", dist=0)
                        p0.append(0)
                        lowers.append(0)
                        uppers.append(0)
                        n_deweighted += 1
                        continue

                pnames += [
                    f"{dustmodel.components[k].name}_{np.round(dustmodel.components[k].sizes[kk] * 10000, decimals=5)}"
                ]
                upper = n_grains[kk]
                if args.abundance_factor:
                    upper *= 1
                lower = (dustmodel.components[k].size_dist[kk] / 1e7) / 1e5

                upper *= 1 + (
                    eps * (k + kk + 1)
                )  # making sure the prior limits are not exactly the same, to avoid correlations that are not there
                lower *= 1 - (eps * (k + kk + 1))

                start_value = dustmodel.components[k].size_dist[kk] / 1e5

                if args.limit_abund:
                    new_factor_C = factor_C * dustmodel.components[k].sizes[kk]
                    new_factor_sil = factor_sil * dustmodel.components[k].sizes[kk]
                    if dustmodel.components[k].name in dustmodel.carbonaceous_names:
                        upper /= new_factor_C
                        lower /= new_factor_C
                        start_value /= new_factor_C
                    else:
                        upper /= new_factor_sil
                        lower /= new_factor_sil
                        start_value /= new_factor_sil
                    upper *= 10 / len(dustmodel.components[k].sizes)

                upper *= 10    

                # prior.add_parameter(
                #     f"c{k + 1}_s{kk + 1}", dist=loguniform(lower, upper)
                # )
                prior.add_parameter(
                                    f"c{k + 1}_s{kk + 1}", dist=LogUniformLin(lower, upper)
                                )
                p0.append(start_value)
                logs.append(True)
                prior_ranges.append([lower, upper])
                lowers.append(lower)
                uppers.append(upper)
                used_sizes.append(float(dustmodel.components[k].sizes[kk] * 10000))

            n_deweighted_comp.append(n_deweighted)
            if args.abundance_factor:
                pnames += [f"F_a{k}"]
                prior.add_parameter(f"F_a{k}", dist=(0.2, 5))
                p0.append(1)
                lowers.append(0.2)
                uppers.append(5)
                logs.append(False)
                prior_ranges.append([0.2, 5])

        if ISRF:
            pnames += ["RF"]
            prior.add_parameter("RF", dist=(0.0001, 30))
            p0.append(1)
            logs.append(False)
            prior_ranges.append([0.0001, 30])
            lowers.append(0.0001)
            uppers.append(30)

        dustmodel.set_size_dist(p0)
        np.savetxt(
            f"priors_{args.tag}.txt", np.column_stack((uppers, lowers)), fmt="%.10e"
        )  # Save priors to a file

    else:
        print("Size distribution choice not known")
        exit()

    # save the starting model
    dustmodel.save_results(basename + "_sizedist_start.fits", obsdata)

    # setup time
    setup_time = time.process_time()
    print(
        "Setup time taken: ",
        np.round((setup_time - start_time) / 60.0, decimals=3),
        " min",
    )

    if args.fitting_package == "nautilus":

        def loglike(a):
            x = np.array(list(a.values()))
            return dustmodel.lnprob(x, obsdata, dustmodel)

        if args.result_from_file == "none":

            if args.parallel:
                sampler = Sampler(
                    prior,
                    loglike,
                    n_live=args.nlivepoints,
                    filepath=f"checkpoint_{basename}_sizedist.hdf5",
                    pool=args.ncores,
                )
            else:
                sampler = Sampler(
                    prior,
                    loglike,
                    n_live=args.nlivepoints,
                    filepath=f"checkpoint_{basename}_sizedist.hdf5",
                )
            sampler.run(verbose=True)
            opt_time = time.process_time()
            print(
                "Optimizer time taken: ",
                np.round((opt_time - setup_time) / 60.0, decimals=2),
                " min",
            )
            print(f"Evidence: {sampler.log_z}")

            points, log_w, log_l = sampler.posterior()
            map_index = np.argmax(log_l)
            opt_params = points[map_index]
            weights = np.exp(log_w)
            # prior_points = sample_prior(
            #     prior_ranges,
            #     logs,
            #     n_samples=100000
            #     )  # Sample from the prior for comparison   

            with h5py.File(f"posterior_samples_{args.tag}.h5", "w") as f:
                f["points"] = points
                f["log_w"] = log_w
                f["log_l"] = log_l
                f["labels"] = pnames

            if args.cornerplot:
                n_params = points.shape[1]
                chunk_size = 4

                for start in range(0, n_params, chunk_size):
                    end = start + chunk_size
                    #priors_subset = prior_points[:, start:end]
                    pts_subset = points[:, start:end]
                    labels_subset = pnames[start:end]
                    opt_subset = opt_params[start:end]
                    #logs_subset = logs[start:end]
                    #priors_subset[:, logs_subset] = np.log(priors_subset[:, logs_subset])
                    #pts_subset[:, logs_subset] = np.log(pts_subset[:, logs_subset])

                    # fig = corner.corner(
                    #     priors_subset,
                    #     bins=100,
                    #     labels=labels_subset,
                    #     color="gray",
                    #     plot_datapoints=False,
                    #     plot_density=False,
                    #     plot_contours=True,
                    #     fill_contours=False,
                    #     levels=[0.68,0.95],
                    # )


                    # corner.corner(
                    #     pts_subset,
                    #     weights=weights,
                    #     bins=100,
                    #     labels=labels_subset,
                    #     color="purple",
                    #     show_titles=True,
                    #     title_fmt=".3g",
                    #     quantiles=[0.16,0.5,0.84],
                    #     levels=[0.68,0.95],
                    #     plot_datapoints=False,
                    #     #range=np.repeat(0.999,len(labels_subset)),
                    #     fig=fig,
                    # )

                    fig = corner.corner(
                        pts_subset,
                        weights=weights,
                        bins=100,
                        labels=labels_subset,
                        color="purple",
                        show_titles=True,
                        title_fmt=".3g",
                        quantiles=[0.16, 0.5, 0.84],
                        levels=[0.68, 0.95],
                        plot_datapoints=False,
                        range=np.repeat(0.999, len(labels_subset)),
                    )
                    corner.overplot_lines(fig, opt_subset, color="red")
                    plt.show()
                    plt.close(fig)

        else:
            x_max = np.loadtxt(f"{args.result_from_file}")
            opt = np.array(x_max)
            lowers = np.array(lowers)
            uppers = np.array(uppers)
            opt_params = np.zeros(len(uppers))
            p = 0
            for i, value in enumerate(uppers):
                if value != 0:
                    k = i - p
                    opt_params[i] = lowers[i] * (uppers[i] / lowers[i]) ** opt[k]
                else:
                    opt_params[i] = 0
                    p += 1
            if f_abund:
                opt_params[-2] = lowers[-2] + (opt[-2] * (uppers[-2] - lowers[-2]))
            if ISRF:
                opt_params[-1] = lowers[-1] + (opt[-1] * (uppers[-1] - lowers[-1]))

        new_opt = np.zeros(len(p0))
        index = 0
        for i, value in enumerate(p0):
            if value != 0:
                new_opt[i] = opt_params[index]
                index += 1
        opt_params = new_opt

        dustmodel.set_size_dist(opt_params)
        print(f"ln(p): {dustmodel.lnprob(opt_params, obsdata, dustmodel)}")
        if args.regularization:
            print(f"ln(p) without regularization: {dustmodel.fracs[5]}")
        print(
            f"ln(p) extinction: {dustmodel.fracs[0]} = {np.round((dustmodel.fracs[0] / dustmodel.fracs[5]) * 100, 2)}%"
        )
        print(
            f"ln(p) emission: {dustmodel.fracs[2]} = {np.round((dustmodel.fracs[2] / dustmodel.fracs[5]) * 100, 2)}%"
        )
        print(
            f"ln(p) abundances: {dustmodel.fracs[1]} = {np.round((dustmodel.fracs[1] / dustmodel.fracs[5]) * 100, 2)}%"
        )
        print(
            f"ln(p) albedo: {dustmodel.fracs[3]} = {np.round((dustmodel.fracs[3] / dustmodel.fracs[5]) * 100, 2)}%"
        )
        print(
            f"ln(p) g: {dustmodel.fracs[4]} = {np.round((dustmodel.fracs[4] / dustmodel.fracs[5]) * 100, 2)}%"
        )
        oname = f"{basename}_sizedist_best_optimizer.fits"
        dustmodel.save_results(oname, obsdata)

    if args.fitting_package == "emcee":
        # do simple optimization to find the best fit
        call_count = {"n": 0}

        def nll(*args):
            call_count["n"] += 1
            if call_count["n"] % 1000 == 0:
                print(
                    f"Call {call_count['n']}: ln(p) = {-dustmodel.lnprob(*args)}"
                )  # added this line to check when the minimizer converges
                print(f"Number of points: {dustmodel.fracs[5]}")
                print(
                    f"Extinction: {round(np.abs(dustmodel.fracs[0]) * 100, 2)}% ({obsdata.ext_npts})"
                )
                print(
                    f"Emission: {round(np.abs(dustmodel.fracs[2]) * 100, 2)}% ({obsdata.ir_emission_npts})"
                )
                print(
                    f"Abundance: {round(np.abs(dustmodel.fracs[1]) * 100, 2)}% ({obsdata.abundance_npts})"
                )
                print(
                    f"Albedo: {round(np.abs(dustmodel.fracs[3]) * 100, 2)}% ({obsdata.scat_a_npts})"
                )
                print(
                    f"g: {round(np.abs(dustmodel.fracs[4]) * 100, 2)}% ({obsdata.scat_g_npts})"
                )
                comps = dustmodel.components
                grain = comps[0]
                print(f"ISRF: {grain.RF_strength}")
            return -dustmodel.lnprob(*args)

        soln = minimize(
            nll,
            p0,
            args=(obsdata, dustmodel),
            method="Nelder-Mead",
            options={"maxiter": 500000, "maxfev": 500000, "disp": True},
        )
        opt_params = soln.x
        dustmodel.set_size_dist_parameters(opt_params)
        opt_time = time.process_time()
        print("optimizer time taken: ", (opt_time - setup_time) / 60.0, " min")

        if args.mcmc:
            p0 = opt_params
            # more emcee setup
            ndim = len(p0)
            nwalkers = 2 * ndim

            print(f"fitting {fitobs_list}")
            print(f"# params = {ndim}")
            print(f"# walkers = {nwalkers}")
            print(f"# burnfrac = {burnfrac}")
            print(f"# steps = {nsteps}")

            # setting up the walkers to start "near" the inital guess
            p = dustmodel.initial_walkers(p0, nwalkers)

            # Set up the backend to save the samples for the emcee runs
            emcee_samples_file = f"{basename}_chain.h5"
            backend = emcee.backends.HDFBackend(emcee_samples_file)
            backend.reset(nwalkers, ndim)

            # setup the sampler
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    dustmodel.lnprob,
                    args=(obsdata, dustmodel),
                    pool=pool,
                    backend=backend,
                )

                # do the sampling
                sampler.run_mcmc(p, nsteps, progress=True)

            emcee_time = time.process_time()
            print("emcee time taken: ", (emcee_time - opt_time) / 60.0, " min")

            # best fit dust params
            oname = "%s_sizedist_best_fin.fits" % (basename)
            dustmodel.save_best_results(oname, sampler, obsdata)

            # 50p dust params
            oname = "%s_sizedist_fin.fits" % (basename)
            dustmodel.save_50percentile_results(
                oname, sampler, obsdata, nburn=int(burnfrac * nsteps)
            )

            if args.cornerplot and ndim < 30:
                # plot the walker chains for all parameters
                nwalkers, nsteps, ndim = sampler.chain.shape
                fig, ax = plt.subplots(ndim, sharex=True, figsize=(13, 13))
                walk_val = np.arange(nsteps)
                for i in range(ndim):
                    for k in range(nwalkers):
                        ax[i].plot(walk_val, sampler.chain[k, :, i], "-")
                        ax[i].set_ylabel(pnames[i])
                fig.savefig(f"{basename}_walker_param_values.png")
                plt.close(fig)

                # plot the 1D and 2D likelihood functions in a traditional triangle plot
                nwalkers, nsteps = sampler.lnprobability.shape
                # discard the 1st burn_frac (burn in)
                flat_samples = sampler.get_chain(
                    discard=int(burnfrac * nsteps), flat=True
                )
                nflatsteps, ndim = flat_samples.shape
                fig = corner.corner(
                    flat_samples,
                    labels=pnames,
                    show_titles=True,
                    title_fmt=".3f",
                    use_math_text=True,
                )
                plt.show()
                fig.savefig(f"{basename}_param_triangle.png")


if __name__ == "__main__":
    main()
