#!/Users/johnfield/anaconda3/bin/python

"""
This workbook supports post-processing and choropleth mapping of regional-scale
DayCent simulation batches, as well as strata-level comparative results
visualization.
"""

import constants
import itertools
import matplotlib
matplotlib.use('MacOSX')  # https://github.com/JuliaPy/PyPlot.jl/issues/454#issuecomment-527269350
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import scipy
import statsmodels.api as sm
import subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# define conversion constants
c_concentration = constants.c_concentration
n_to_n2o = constants.n_to_n2o
c_to_co2 = constants.c_to_co2
n2o_gwp100_ar5 = constants.n2o_gwp100_ar5
ch4_gwp100_ar5 = constants.ch4_gwp100_ar5
g_m2_to_kg_ha = constants.g_m2_to_kg_ha
kg_ha_to_Mg_ha = constants.kg_ha_to_Mg_ha
g_m2_to_Mg_ha = constants.g_m2_to_Mg_ha

results_path = '/Users/johnfield/Desktop/CBI_mapping/'
# scope = ['MT', 'WY', 'CO', 'NM', 'TX', 'OK', 'KS', 'NE', 'SD', 'ND', 'MN',
#          'IA', 'MO', 'AR', 'LA', 'MS', 'IL', 'WI', 'MI', 'IN', 'OH', 'KY',
#          'TN', 'AL']
scope = ''


def data_loss(target_row_count, df):
    """Calculate change in dataframe row number relative to previous/target
    value.
    Return: new row count, percent loss relative to target value
    """

    new_row_count = df.shape[0]
    data_loss_percentage = ((target_row_count - new_row_count) /
                            float(target_row_count)) * 100.0
    return new_row_count, data_loss_percentage


def grep_cksum(grep_string, files):
    """ Perform a grep command and compute the checksum of the resulting string
    in order to check for differences between subsections of multiple large
    .csv files.
    """
    print("Performing grep operations for string '{}'...".format(grep_string))
    ref_cksum = ""
    match = True
    for e, file in enumerate(files):
        print("\tCalculating checksum value for {}...".format(file))
        cmd = "egrep '{}' {} | cksum".format(grep_string, file)
        current_cksum = subprocess.getoutput(cmd)
        print("\t\t", current_cksum)
        if not e:
            ref_cksum = current_cksum
        if current_cksum != ref_cksum:
            match = False
    if match:
        print("IDENTICAL grep results across the files tested")
    else:
        print("DIFFERENT grep results across the files tested")


def surface_fit(xs, ys, zs, order=1, ax_labels=('X', 'Y', 'Z', '')):
    """Create and display a 3D scatter plot of the data, including a linear or
    quadratic fitted surface. Adapated from
    https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    """

    # transpose data and convert to numpy array object
    data = [xs, ys, zs]
    data = list(zip(*data))
    data = np.asarray(data)

    # define a regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])    # coefficients
        Z = C[0] * X + C[1] * Y + C[2]

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:,:2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C).reshape(X.shape)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    ax.set_zlabel(ax_labels[2])
    plt.title(ax_labels[3])
    plt.show()


def fips_mapping(county_df, title, column_mapped, legend_title, filename,
                 linspacing, divergent=False, reverse=False):

    print("\t\tMapping {} results at county scale, saved at {}".
          format(column_mapped, filename))

    # use 'linspacing' parameters to a bin list, and specify rounding if values are small-ish
    bin_list = np.linspace(linspacing[0], linspacing[1], linspacing[2]).tolist()
    rounding = True
    if linspacing[1] < 10:
        rounding = False

    kwargs = {}
    if scope:
        kwargs['scope'] = scope

    if divergent:
        # convert matplotlib (r, g, b, x) tuple color format to 'rgb(r, g, b)' Plotly string format
        cmap = get_cmap('RdBu')  # or RdYlBu for better differentiation vs. missing data squares in tiling map
        custom_rgb_cmap = [cmap(x) for x in np.linspace(0, 1, (linspacing[2] + 1))]
        custom_plotly_cmap = []
        for code in custom_rgb_cmap:
            plotly_code = 'rgb({},{},{})'.format(code[0] * 255.0, code[1] * 255.0, code[2] * 255.0)
            custom_plotly_cmap.append(plotly_code)
        if reverse:
            custom_plotly_cmap.reverse()

        kwargs['state_outline'] = {'color': 'rgb(100,100,100)', 'width': 1.0}
        kwargs['colorscale'] = custom_plotly_cmap

    fig = ff.create_choropleth(fips=county_df['fips'],
                               values=county_df[column_mapped].tolist(),
                               binning_endpoints=bin_list,
                               round_legend_values=rounding,
                               county_outline={'color': 'rgb(255,255,255)', 'width': 0.25},
                               legend_title=legend_title,
                               title=title,
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               **kwargs)
    py.offline.plot(fig, filename=(results_path + filename))
    # to make interactive, include_plotlyjs='https://cdn.plot.ly/plotly-1.42.3.min.js'
    # as per https://github.com/plotly/plotly.py/issues/1429


def run_results(run_fpath, results_path, lis_filename, ys_filename):
    """ Load, characterize & merge DayCent batch runtable file with .lis &
    year_summary.out results.
    Return: pandas dataframes with merged annual results, and runtable
    """

    # read & characterize DayCent runtable, .lis & year_summary.out results
    print("\t\tReading DayCent runtable {}...".format(run_fpath))
    run_df = pd.read_csv(run_fpath, skiprows=[1])   # skip SQL datatype row
    strata_count = run_df.shape[0]
    run_soils = run_df['mukey_int'].nunique()
    run_climates = run_df.groupby(['gridx', 'gridy']).ngroups
    run_counties = run_df['fips'].nunique()
    run_area = run_df[run_df['sch_file'] == 'idle_switchgrass75.sch']['tot_ha'].sum()
    unique_sch = run_df['sch_file'].unique()
    print("\t\t\tRuntable contains {:,} strata (rows) covering {:,} soils,".
          format(strata_count, run_soils))
    print("\t\t\t{:,} climate grid cells, {:,} counties, and {:,.0f} ha".
          format(run_climates, run_counties, run_area))
    print("\t\t\tsimulated with the following DayCent schedule files:")
    print("\t\t\t", unique_sch)

    print("\t\tReading DayCent .lis results at {}...".
          format(results_path + lis_filename))
    lis_df = pd.read_csv(results_path + lis_filename, skiprows=[1])
    lis_count = lis_df.shape[0]
    lis_strata = lis_df['strata_no'].nunique()
    print("\t\t\tDayCent .lis results contain {:,} result-years (rows) covering {:,} strata".
          format(lis_count, lis_strata))

    print("\t\tReading DayCent year_summary results at {}...".
          format(results_path + ys_filename))
    ys_df = pd.read_csv(results_path + ys_filename, skiprows=[1])
    ys_count = ys_df.shape[0]
    # fewer rows than .lis, which reports initial state the year after simulation
    ys_strata = ys_df['strata_no'].nunique()
    print("\t\t\tDayCent year_summary.out results contain {:,} result-years (rows) covering {:,} strata".
          format(ys_count, ys_strata))

    # merge .lis & year_summary.out results
    print("\t\tMerging DayCent .lis and year_summary.out results...")
    full_df = pd.merge(lis_df, ys_df, on=['strata_no', 'crop', 'land_type', 'time'])
    row_count, loss_percentage = data_loss(lis_count, full_df)
    print("\t\t\tDayCent .lis -- year_summary.out merger yields {:,} rows ({:.3f} % loss)".
          format(row_count, loss_percentage))

    # perform unit conversions for per-area yields & absolute GHG emissions
    print("\t\tPerforming unit conversions on DayCent yield & GHG results...")
    full_df['yield_Mg_ha'] = ((full_df['crmvst'] * g_m2_to_Mg_ha) / c_concentration)
    full_df['dSOC_MgC_ha'] = (full_df['d_somsc'] * g_m2_to_Mg_ha)
    full_df['dN2ON_kgN_ha'] = (full_df['N2Oflux'] * g_m2_to_kg_ha)
    full_df['iN2ON_kgN_ha'] = ((0.0075 * full_df['strmac(2)'] +
                                0.01 * full_df['volpac'] +
                                0.01 * full_df['NOflux']) * g_m2_to_kg_ha)
    full_df['kgCH4_ox_ha'] = (full_df['CH4'] * g_m2_to_kg_ha)
    full_df['ghg_MgCO2e_ha'] = (full_df['dSOC_MgC_ha'] * c_to_co2 * -1.0) + \
                               ((full_df['dN2ON_kgN_ha'] + full_df['iN2ON_kgN_ha']) *
                                kg_ha_to_Mg_ha * n_to_n2o * n2o_gwp100_ar5) + \
                               (full_df['kgCH4_ox_ha'] * kg_ha_to_Mg_ha * ch4_gwp100_ar5 * -1.0)

    return full_df, run_df, strata_count, run_area


def strata_yields(annual_df, run_df, test_strata):
    """Extract annual harvested biomass results for a single strata, and
    regress against the associated .wth file climate data.
    """

    # determine & print analysis metadata for reference
    fips = run_df.loc[run_df['strata_no'] == test_strata, 'fips'].values[0]
    print("Plotting annual yield detail for strata {} in FIPS {}".
          format(test_strata, fips))

    # retrieve .wth file
    wth_path = "/Volumes/wcnr-network/Research/paustian/AFRI/NARR_gridxy_wth/"
    narrx = run_df.loc[run_df['strata_no'] == test_strata, 'gridx'].values[0]
    narry = run_df.loc[run_df['strata_no'] == test_strata, 'gridy'].values[0]
    wth_filename = "NARR_{}_{}.wth".format(narrx, narry)
    wth_fpath = wth_path + wth_filename
    print("\tReading NARR climate data at {}...".format(wth_fpath))
    wth_names = ['DOM', 'MO', 'year', 'DOY', 'Tmax', 'Tmin', 'precip']
    wth_df = pd.read_csv(wth_fpath, sep='\t', names=wth_names)

    # calculate growing season totals/averages
    seasonal_wth_df = wth_df[wth_df['MO'].isin([4, 5, 6, 7, 8])]
    seasonal_wth_df['Tavg'] = (seasonal_wth_df['Tmin'] + seasonal_wth_df['Tmax']) / 2.0
    annunal_wth_df = seasonal_wth_df.groupby('year').agg({'Tmax': 'mean','Tavg': 'mean', 'precip': 'sum'})
    annunal_wth_df = annunal_wth_df.reset_index()   # necessary to access 'year' column

    # determine simulation start year corresponding to 1979 in NARR .wth file
    # NOTE: this is unique to GECP abandoned land simulations
    peak_year = run_df.loc[run_df['strata_no'] == test_strata, 'peak_year'].values[0]
    start_year = (int(round((peak_year+2000)/2.0))) + 1

    # re-index .wth data to sync with future DayCent simulation results, and
    # visualize in simple illustrative line plot
    # NOTE: this is unique to GECP abandoned land simulations
    phasing = (2030 - start_year) % 31
    NARR_2030_year = 1979 + phasing
    offset = 2030 - NARR_2030_year
    annunal_wth_df['index_year'] = np.where(annunal_wth_df['year']+offset<2030,
                                            annunal_wth_df['year']+offset+31,
                                            annunal_wth_df['year']+offset)
    # print(annunal_wth_df)
    ax = annunal_wth_df.plot(x='year', y='precip', kind='line', label='precip')
    ax.set_ylabel('Apr-Aug total precip (cm)')
    annunal_wth_df.plot(x='year', y='Tavg', kind='line', label='Tavg',
                        color='r', ax=ax, secondary_y=True)
    plt.ylabel('Apr–Aug avg. daily temp (C)')

    # retrieve strata yield results for future simulations
    yield_df = annual_df[(annual_df['strata_no'] == test_strata) &
                         (annual_df['time'] > 2019)]

    # merge the analyzed weather data with yield results & create surface plot
    regression_df = pd.merge(yield_df, annunal_wth_df, left_on='time', right_on='index_year')
    # print(regression_df)
    surface_fit(regression_df['Tavg'],
                regression_df['precip'],
                regression_df['yield_Mg_ha'],
                order=2,
                ax_labels=('Apr–Aug avg. daily temp (C)',
                           'Apr-Aug total precip (cm)',
                           'Biomass yield (Mg ha-1)',
                           'Yield vs. weather'))


def strata_comparison(base_annual_df, trt_annual_df, test_strata, variable,
                      trt_label="", ylabel=''):
    """Create comparison plot of strata results for both 'base' and 'treatment'
    cases.
    """

    base_subset = base_annual_df[(base_annual_df['strata_no'] == test_strata) &
                                 (base_annual_df['time'] > 2019)]
    trt_subset = trt_annual_df[(trt_annual_df['strata_no'] == test_strata) &
                               (trt_annual_df['time'] > 2019)]
    ax = base_subset.plot(x='time', y=variable, kind='line',
                          label='baseline')
    trt_subset.plot(x='time', y=variable, kind='line', color='r', ax=ax,
                    label=trt_label)
    ax.set_ylabel(ylabel)


def strata_aggregate(annual_df, run_df, expected_strata, run_area, description):
    """Aggregate to strata|scenario-level, dropping all past 'idle' period
    results (note that pass-through columns need to be individually specified).
    Then merge with strata-level soil & climate metadata.
    """

    # filter the annual results, and aggregate by strata
    print("\t\tFiltering out idle period results, and grouping by strata...")
    strata_df = annual_df[annual_df['time'].between(2020, 2068, inclusive=True)][
        ['strata_no', 'land_type', 'yield_Mg_ha', 'dSOC_MgC_ha', 'dN2ON_kgN_ha', 'iN2ON_kgN_ha', 'kgCH4_ox_ha', 'ghg_MgCO2e_ha']].groupby(
            ['strata_no', 'land_type']).mean()

    std_df = annual_df[annual_df['time'].between(2020, 2068, inclusive=True)][
        ['strata_no', 'land_type', 'yield_Mg_ha']].groupby(['strata_no', 'land_type']).std()
    std_df.columns = ['yield_std_Mg_ha']
    strata_df = pd.merge(strata_df, std_df, left_index=True, right_index=True)

    strata_df = strata_df.reset_index()   # otherwise 'land_type' gets dropped in the next merge
    strata_count = strata_df.shape[0]
    strata_loss = ((expected_strata - strata_count) / float(expected_strata)) * 100.0
    print("\t\t\tFiltered, aggregated results represent {:,} strata ({:.3f}% loss relative to runtable)".format(
        strata_count, strata_loss))

    # load & characterize soil, climate & county metadata
    ssurgo_metadata_fpath = '/Users/johnfield/Desktop/GCEP_local/ssurgo2012_texture_depth.csv'
    narr_metadata_fpath = '/Users/johnfield/Desktop/GCEP_local/narr_31yr_avg.csv'
    county_metadata_fpath = '/Users/johnfield/Desktop/GCEP_local/eastern_US_fips.csv'

    print("\t\tReading soil metadata from {}...".format(ssurgo_metadata_fpath))
    soil_df = pd.read_csv(ssurgo_metadata_fpath)
    soil_count = soil_df.shape[0]
    print("\t\t\tMetadata read for {:,} soils".format(soil_count))

    print("\t\tReading climate metadata from {}...".format(narr_metadata_fpath))
    wth_df = pd.read_csv(narr_metadata_fpath)
    wth_count = wth_df.shape[0]
    print("\t\t\tMetadata read for {:,} climate grid cells".format(wth_count))

    print("\t\tReading FIPS codes from {}...".format(county_metadata_fpath))
    county_metadata_df = pd.read_csv(county_metadata_fpath)
    fips_count = county_metadata_df.shape[0]
    print("\t\t\tFIPS list contains codes for {:,} counties".format(fips_count))

    # merge strata results w/ runtable & soil/climate metadata
    print("\t\tMerging strata results with runtable, soil & weather metadata...")
    strata_df = pd.merge(strata_df, run_df, on='strata_no')
    strata_df = pd.merge(strata_df, soil_df, left_on='mukey_int', right_on='mukey')
    strata_df = pd.merge(strata_df, wth_df, on=['gridx', 'gridy'])
    strata_count, strata_loss = data_loss(strata_count, strata_df)
    print("\t\t\tStrata -- runtable -- soil/weather metadata merger retains {:,} strata ({:.3f}% loss)".
          format(strata_count, strata_loss))
    strata_area = strata_df[strata_df['land_type'] == 'switchgrass75']['tot_ha'].sum()
    area_loss = ((run_area - strata_area) / float(run_area)) * 100.0
    print("\t\t\tStrata -- runtable -- soil/weather metadata merger covers {:,.0f} ha ({:.3f}% loss relative runtable)".
          format(strata_area, area_loss))

    # calculate area totals
    strata_df['yield_Mg'] = strata_df['yield_Mg_ha'] * strata_df['tot_ha']
    strata_df['dSOC_MgC'] = strata_df['dSOC_MgC_ha'] * strata_df['tot_ha']
    strata_df['dN2ON_kgN'] = strata_df['dN2ON_kgN_ha'] * strata_df['tot_ha']
    strata_df['iN2ON_kgN'] = strata_df['iN2ON_kgN_ha'] * strata_df['tot_ha']
    strata_df['kgCH4_ox'] = strata_df['kgCH4_ox_ha'] * strata_df['tot_ha']
    strata_df['ghg_MgCO2e'] = strata_df['ghg_MgCO2e_ha'] * strata_df['tot_ha']
    strata_df['sand_weight'] = strata_df['sand'] * strata_df['tot_ha']
    strata_df['silt_weight'] = strata_df['silt'] * strata_df['tot_ha']
    strata_df['clay_weight'] = strata_df['clay'] * strata_df['tot_ha']
    strata_df['ave_temp_weight'] = strata_df['ave_temp'] * strata_df['tot_ha']
    strata_df['precip_ave_weight'] = strata_df['precip_ave'] * strata_df['tot_ha']
    strata_df['yield_std'] = strata_df['yield_std_Mg_ha'] * strata_df['tot_ha']

    strata_df.to_csv(results_path + '{}-strata_results.csv'.format(description))

    return strata_df, county_metadata_df


def county_aggregate(strata_df, county_metadata_df, description):

    print("\t\tAggregating results to county scale, and computing area-weighted performance...")
    county_df = strata_df[['fips',
                           'tot_ha',
                           'yield_Mg',
                           'dSOC_MgC',
                           'dN2ON_kgN',
                           'iN2ON_kgN',
                           'kgCH4_ox',
                           'ghg_MgCO2e',
                           'sand_weight',
                           'silt_weight',
                           'clay_weight',
                           'ave_temp_weight',
                           'precip_ave_weight',
                           'yield_std']].groupby('fips').sum()

    county_df['yield_Mg_ha'] = county_df['yield_Mg'] / county_df['tot_ha']
    county_df['dSOC_MgC_ha'] = county_df['dSOC_MgC'] / county_df['tot_ha']
    county_df['dN2ON_kgN_ha'] = county_df['dN2ON_kgN'] / county_df['tot_ha']
    county_df['iN2ON_kgN_ha'] = county_df['iN2ON_kgN'] / county_df['tot_ha']
    county_df['kgCH4_ox_ha'] = county_df['kgCH4_ox'] / county_df['tot_ha']
    county_df['ghg_MgCO2e_ha'] = county_df['ghg_MgCO2e'] / county_df['tot_ha']
    county_df['sand'] = county_df['sand_weight'] / county_df['tot_ha']
    county_df['silt'] = county_df['silt_weight'] / county_df['tot_ha']
    county_df['clay'] = county_df['clay_weight'] / county_df['tot_ha']
    county_df['ave_temp'] = county_df['ave_temp_weight'] / county_df['tot_ha']
    county_df['precip_ave'] = county_df['precip_ave_weight'] / county_df['tot_ha']
    county_df['yield_std_avg'] = county_df['yield_std'] / county_df['tot_ha']

    county_df = pd.merge(county_df, county_metadata_df, on='fips')
    county_df.to_csv(results_path + '{}-county_scale_results.csv'.format(description))

    return(county_df)


def national_totals(county_df, scenario):
    annual_biomass_Mg = county_df['yield_Mg'].sum()
    annual_biomass_Mt = annual_biomass_Mg / 1000000.0
    fuel_yield = 318  # L Mg-1
    annual_fuel_GL = (annual_biomass_Mg * fuel_yield) / 1000000000.0
    annual_dSOC_Mg = county_df['dSOC_MgC'].sum()
    annual_dSOC_Mt = annual_dSOC_Mg / 1000000.0
    print("\t\t{} scenario produces {:.1f} Mt biomass annually ({:.1f} GL fuel equivalent) with {:.3f} Mt soil C change.".
           format(scenario, annual_biomass_Mt, annual_fuel_GL, annual_dSOC_Mt))

    return annual_biomass_Mt, annual_dSOC_Mt


def scenario_process(description, run_fpath, results_path, lis_filename,
                     ys_filename):

    print("\tProcessing results from scenario '{}'...".format(description))

    # load baseline results, aggregate, and compute national totals
    annual_df, run_df, strata_count, run_area = run_results(run_fpath,
                                                            results_path,
                                                            lis_filename,
                                                            ys_filename)
    strata_df, county_metadata_df = strata_aggregate(annual_df, run_df,
                                                     strata_count, run_area,
                                                     description)
    county_df = county_aggregate(strata_df, county_metadata_df, description)
    print()
    national_totals(county_df, description)
    print()
    print()

    return annual_df, strata_df, county_df, run_df


def uniform_comparison(title_text, description_string, base_county_df,
                       trt_county_df):
    """Compares between two different sets of DayCent simulation results
    assuming that they are each implemented uniformly across the comparison
    area. Creates choropleth maps showing difference in yield (absolute &
    relative), SOC change rate and net biogenic GHG emissions (absolute only).
    """

    # merge county-aggregated baseline and treatment results
    sensitivity_df = pd.merge(base_county_df, trt_county_df, on='fips',
                              suffixes=('_base', '_trt'))

    # compute and map % change results for yield, SOC, and total GHGs
    sensitivity_df['yield_Mg_ha_change'] = sensitivity_df['yield_Mg_ha_trt'] - sensitivity_df['yield_Mg_ha_base']
    sensitivity_df['yield_Mg_percent'] = ((sensitivity_df['yield_Mg_trt'] -
                                          sensitivity_df['yield_Mg_base']) /
                                           sensitivity_df['yield_Mg_base']) * 100.0
    sensitivity_df['dSOC_kgC_ha_change'] = (sensitivity_df['dSOC_MgC_ha_trt'] - sensitivity_df['dSOC_MgC_ha_base']) * 1000.0
    sensitivity_df['ghg_MgCO2e_ha_change'] = sensitivity_df['ghg_MgCO2e_ha_trt'] - sensitivity_df['ghg_MgCO2e_ha_base']

    fips_mapping(sensitivity_df,
                 'Yield response to {}'.format(title_text), 'yield_Mg_ha_change', '(Mg ha-2)',
                 'Yield-{}.html'.format(description_string),
                 (-2, 2, 41), divergent=True)
    fips_mapping(sensitivity_df,
                 'Yield response to {}'.format(title_text), 'yield_Mg_percent', '% change',
                 'Yield_percent-{}.html'.format(description_string),
                 (-10, 10, 21), divergent=True)
    fips_mapping(sensitivity_df,
                 'SOC change response to {}'.format(title_text), 'dSOC_kgC_ha_change', '(kg C ha-1 y-1)',
                 'dSOC-{}.html'.format(description_string),
                 (-75, 75, 31), divergent=True)
    fips_mapping(sensitivity_df,
                 'Total GHG emissions response to {}'.format(title_text), 'ghg_MgCO2e_ha_change', '(Mg CO2e ha-1 y-1)',
                 'GHG-{}.html'.format(description_string),
                 (-.30, .30, 31), divergent=True, reverse=True)

    print()
    print("\tBase scenario results (uniform deployment):")
    national_totals(base_county_df, "base")
    print()
    print("\tTreatment scenario results (uniform deployment):")
    national_totals(trt_county_df, "treatment")
    print()

    return sensitivity_df


def tiled_comparison(trt_title, base_title, trt_strata_df, base_strata_df):
    """Compares between two different sets of DayCent simulation results
    assuming that the highest-yielding scenario (base or treatment) is
    implemented for each strata. Creates choropleth maps showing fraction of
    country area on which treatment is implemented, and associated county
    changes in total yield, SOC change, and net biogenic GHG emissions.
    """

    # merge strata-aggregated baseline and treatment results
    transfer = ['strata_no', 'fips', 'yield_Mg_ha', 'dSOC_MgC_ha', 'ghg_MgCO2e_ha',
                'tot_ha']
    df = pd.merge(base_strata_df[transfer], trt_strata_df[transfer],
                  on='strata_no', suffixes=('_base', '_trt'))

    # conditional columns for area trt > base, & total production & GHGs
    df['trt_selected_ha'] = np.where(df['yield_Mg_ha_trt'] > df['yield_Mg_ha_base'],
                                     df['tot_ha_base'], 0.0)
    df['selected_yield_Mg'] = np.where(df['yield_Mg_ha_trt'] > df['yield_Mg_ha_base'],
                                     df['yield_Mg_ha_trt'] * df['tot_ha_base'], df['yield_Mg_ha_base'] * df['tot_ha_base'])
    df['selected_dSOC_MgC'] = np.where(df['yield_Mg_ha_trt'] > df['yield_Mg_ha_base'],
                                     df['dSOC_MgC_ha_trt'] * df['tot_ha_base'], df['dSOC_MgC_ha_base'] * df['tot_ha_base'])
    df['selected_ghg_MgCO2e'] = np.where(df['yield_Mg_ha_trt'] > df['yield_Mg_ha_base'],
                                     df['ghg_MgCO2e_ha_trt'] * df['tot_ha_base'], df['ghg_MgCO2e_ha_base'] * df['tot_ha_base'])

    # group by county, compute selected area fraction & map results
    county_df = df[['fips_base', 'tot_ha_base',
                    'selected_yield_Mg', 'selected_dSOC_MgC',
                    'selected_ghg_MgCO2e', 'trt_selected_ha']].groupby('fips_base').sum()
    county_df['fips'] = county_df.index
    county_df['selected_area_percent'] = (county_df['trt_selected_ha'] / county_df['tot_ha_base']) * 100.0

    # changing columns for consistency w/ national_totals(), uniform_comparison()
    county_df['tot_ha'] = county_df['tot_ha_base']
    county_df['yield_Mg'] = county_df['selected_yield_Mg']
    county_df['yield_Mg_ha'] = county_df['yield_Mg'] / county_df['tot_ha']
    county_df['dSOC_MgC'] = county_df['selected_dSOC_MgC']
    county_df['dSOC_MgC_ha'] = county_df['dSOC_MgC'] / county_df['tot_ha']
    county_df['ghg_MgCO2e'] = county_df['selected_ghg_MgCO2e']
    county_df['ghg_MgCO2e_ha'] = county_df['ghg_MgCO2e'] / county_df['tot_ha']

    fips_mapping(county_df,
                 'Fraction of county land selected for {} variety over {} variety'.format(trt_title, base_title), 'selected_area_percent', '(%)',
                 'Area_{}_over_{}.html'.format(trt_title, base_title),
                 (-20, 120, 29), divergent=True, reverse=True)
    print()
    national_totals(county_df, "Tiled")
    print()

    return county_df


###############################################################################

### Compare raw data file contents ###
# grep_cksum(",118,208,|,119,208,",
#            ['/Users/johnfield/Desktop/greptest.csv',
#             '/Users/johnfield/Desktop/greptest\ 2.csv'])
#
# # selecting representative NARR cells from states that failed sensitivity test
# # Marion Co, IN (fips=18097) 115,228
# # Oswego Co, NY (fips=36075) 136,249
# # Stevens Co, KS (fips=20189) 98,190
# # Pecos Co, TX (fips=48371) 75,186
# # Polk Co, FL (fips=12105) 80,253
#
# grep_cksum(",115,228,|,136,249,|,98,190,|,75,186,|,80,253,",
#            ['/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,13.26__eastern_US_runtable_incl81__79__CBI_baseline/X.lis',
#             '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,12.43__eastern_US_runtable_incl81__90__drought_sensitivity/X.lis',
#             '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,13.15__eastern_US_runtable_incl81__89__allocation_sensitivity/X.lis',
#             '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,13.21__eastern_US_runtable_incl81__91__productivity_sensitivity/X.lis'])


### Initial raw results processing & annual results visualization ###
base_annual_df, base_strata_df, base_county_df, run_df = scenario_process(
    'CBI_baseline',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,13.26__eastern_US_runtable_incl81__79__CBI_baseline/',
    'X.lis', 'year_summary.out')


strata_df, county_metadata_df = strata_aggregate(base_annual_df, run_df, 1, 1, 'CBI_baseline')

print(base_annual_df[base_annual_df['strata_no'] == 123456].to_string())



moisture_annual_df, moisture_strata_df, moisture_county_df, run_df = scenario_process(
    'Decreased_moisture_dependence',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-11-01,00.37__eastern_US_runtable_incl81__90__drought_sensitivity/',
    'X.lis', 'year_summary.out')

allocation_annual_df, allocation_strata_df, allocation_county_df, run_df = scenario_process(
    'Increased_belowground_allocation',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-11-08,20.22__eastern_US_runtable_incl81__89__allocation_sensitivity/',
    'X.lis', 'year_summary.out')

prdx_annual_df, prdx_strata_df, prdx_county_df, run_df = scenario_process(
    'Increased_productivity',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-11-08,20.18__eastern_US_runtable_incl81__91__productivity_sensitivity/',
    'X.lis', 'year_summary.out')

prdx4_annual_df, prdx4_strata_df, prdx4_county_df, run_df = scenario_process(
    '4percent_productivity_increase',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-11-14,22.13__eastern_US_runtable_incl81__95__4percent_productivity/',
    'X.lis', 'year_summary.out')

cn_annual_df, cn_strata_df, cn_county_df, run_df = scenario_process(
    'Decreased_N_requirement',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-11-09,14.46__eastern_US_runtable_incl81__92__CN_sensitivity/',
    'X.lis', 'year_summary.out')

drought_annual_df, drought_strata_df, drought_county_df, run_df = scenario_process(
    'Increased_drought_tolerance',
    '/Users/johnfield/Desktop/GCEP_local/eastern_US_runtable_incl81.csv',
    '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-11-10,07.49__eastern_US_runtable_incl81__94__FSETH1_sensitivity/',
    'X.lis', 'year_summary.out')

# # visualize annual results, comparing baseline & treatment
# # stratas = base_annual_df['strata_no'].unique()
# test_strata = 138232  # 134895
# strata_yields(base_annual_df, run_df, test_strata)
# strata_comparison(base_annual_df, drought_annual_df,
#                   test_strata, 'yield_Mg_ha',
#                   trt_label='Improved drought tolerance',
#                   ylabel='Biomass yield (Mg ha-1)')


### Conduct county-scale sensitivity comparison & map ###
base_county_df = pd.read_csv(results_path + 'CBI_baseline-county_scale_results.csv')
moisture_county_df = pd.read_csv(results_path + 'Decreased_moisture_dependence-county_scale_results.csv')
allocation_county_df = pd.read_csv(results_path + 'Increased_belowground_allocation-county_scale_results.csv')
prdx_county_df = pd.read_csv(results_path + 'Increased_productivity-county_scale_results.csv')
prdx4_county_df = pd.read_csv(results_path + '4percent_productivity_increase-county_scale_results.csv')
cn_county_df = pd.read_csv(results_path + 'Decreased_N_requirement-county_scale_results.csv')
drought_county_df = pd.read_csv(results_path + 'Increased_drought_tolerance-county_scale_results.csv')

county_df = base_county_df
description = 'CBI_baseline'
map = False
if map:
    fips_mapping(county_df,
                 'Abandoned land availability', 'tot_ha', '(ha)',
                 '{}-Area.html'.format(description),
                 (0, 100000, 21))
    fips_mapping(county_df,
                 'Soil sand fraction (area weighted)', 'sand', '(%)',
                 '{}-Sand.html'.format(description),
                 (0, 95, 20))
    fips_mapping(county_df,
                 'Soil clay fraction (area weighted)', 'clay', '(%)',
                 '{}-Clay.html'.format(description),
                 (0, 60, 13))
    fips_mapping(county_df,
                 'Average annual precipitation (area weighted)', 'precip_ave', '(cm)',
                 '{}-Precip.html'.format(description),
                 (30, 150, 13))
    fips_mapping(county_df,
                 'Average temperature (area weighted)', 'ave_temp', '(C)',
                 '{}-Temp.html'.format(description),
                 (5, 25, 21))

    # map per-area yield & absolute GHG results
    fips_mapping(county_df,
                 'Switchgrass area yield', 'yield_Mg_ha', '(Mg ha-1 y-1)',
                 '{}-Yield.html'.format(description),
                 (4, 24, 21))
    fips_mapping(county_df,
                 'Absolute soil carbon increase under switchgrass', 'dSOC_MgC_ha', '(Mg C ha-1 y-1)',
                 '{}-absolute_SOC_divergent.html'.format(description),
                 (-0.25, 0.25, 21),
                 divergent=True)
    fips_mapping(county_df,
                 'Absolute direct N2O emissions under switchgrass', 'dN2ON_kgN_ha', '(kg N2O-N ha-1 y-1)',
                 '{}-absolute_dN2ON.html'.format(description),
                 (0.8, 3, 22))
    fips_mapping(county_df,
                 'Absolute indirect N2O emissions under switchgrass', 'iN2ON_kgN_ha', '(kg N2O-N ha-1 y-1)',
                 '{}-absolute_iN2ON.html'.format(description),
                 (0, 0.5, 21))
    fips_mapping(county_df,
                 'Absolute CH4 oxidation under switchgrass', 'kgCH4_ox_ha', '(kg CH4 ha-1 y-1)',
                 '{}-absolute_CH4_ox.html'.format(description),
                 (0.8, 2.8, 21))
    fips_mapping(county_df,
                'Absolute net biogenic GHG balance under switchgrass', 'ghg_MgCO2e_ha', '(Mg CO2e ha-1 y-1)',
                 '{}-absolute_GHG_balance.html'.format(description),
                 (-1.6, 1.6, 17),
                 divergent=True, reverse=True)

    # map total county yield & absolute GHG results
    fips_mapping(county_df,
                 'Total biomass production', 'yield_Mg', '(Mg y-1)',
                 '{}-summed_Biomass.html'.format(description),
                 (0, 1000000, 21))
    fips_mapping(county_df,
                 'Total absolute soil carbon increase', 'dSOC_MgC', '(Mg C y-1)',
                 '{}-summed_absolute_SOC.html'.format(description),
                 (-10000, 10000, 21),
                 divergent=True)
    fips_mapping(county_df,
                 'Total absolute biogenic GHG flux', 'ghg_MgCO2e', '(Mg CO2e y-1)',
                 '{}-summed_absolute_GHG_flux.html'.format(description),
                 (-50000, 50000, 21),
                 divergent=True, reverse=True)

# analyze sensitivities & map results
moisture_sens_df = uniform_comparison('10% decrease tolerance increase',
                                      'Decreased_moisture_dependence',
                                      base_county_df, moisture_county_df)

allo_sens_df = uniform_comparison('10% increase in belowground allocation',
                                  'Increased_belowground_allocation',
                                  base_county_df, allocation_county_df)

prdx_sens_df = uniform_comparison('10% increase in productivity (PRDX)',
                                  'Increased_productivity',
                                  base_county_df, prdx_county_df)

prdx4_sens_df = uniform_comparison('4% increase in productivity (PRDX)',
                                   '4percent_productivity_increase',
                                   base_county_df, prdx4_county_df)

cn_sens_df = uniform_comparison('10% increase in maximum tissue C:N ratio',
                                'Decreased_N_requirement',
                                base_county_df, cn_county_df)

drought_sens_df = uniform_comparison('10% drought tolerance increase',
                                     'Increased_drought_tolerance',
                                     base_county_df, drought_county_df)

### perform some exploratory multiple regression ###
# https://datatofish.com/multiple-linear-regression-python/
# X = drought_sens_df[['sand_base', 'precip_ave_base']]
# Y = drought_sens_df['yield_Mg_ha_change']
# X = sm.add_constant(X)  # adding a constant
# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X)
# print(model.summary())
#
# X = drought_sens_df[['sand_base', 'precip_ave_base']]
# Y = drought_sens_df['dSOC_kgC_ha_change']
# X = sm.add_constant(X)  # adding a constant
# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X)
# print(model.summary())


### Conduct strata-level sensitivity comparison for 'tiling' analysis ###
base_strata_df = pd.read_csv(results_path + 'CBI_baseline-strata_results.csv')
moisture_strata_df = pd.read_csv(results_path + 'Decreased_moisture_dependence-strata_results.csv')
allocation_strata_df = pd.read_csv(results_path + 'Increased_belowground_allocation-strata_results.csv')
prdx_strata_df = pd.read_csv(results_path + 'Increased_productivity-strata_results.csv')
prdx4_strata_df = pd.read_csv(results_path + '4percent_productivity_increase-strata_results.csv')
cn_strata_df = pd.read_csv(results_path + 'Decreased_N_requirement-strata_results.csv')
drought_strata_df = pd.read_csv(results_path + 'Increased_drought_tolerance-strata_results.csv')

tiled_county_df = tiled_comparison('drought_tolerance', 'increased_productivity',
                                   drought_strata_df, prdx4_strata_df)

tiled_sens_df = uniform_comparison('tiled drought tolerance / productivity increase',
                                     'Drought-productivity_tiling',
                                     base_county_df, tiled_county_df)
