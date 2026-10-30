"""
Functions for statistical tests for analyzing model results.
"""
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple
from itertools import combinations
from statsmodels.stats.multitest import multipletests


def get_session_mean_df(df, statistic="usage", max_syllable=40):
    """
    Compute a given mean syllable statistic grouped by groups and UUIDs.

    Args:
    df (pandas.DataFrame): dataframe that contains average syllable statistics per session (mean_df/stats_df)
    statistic (str): statistic to compute mean for, (any of the columns in input df); for example: 'usage', 'duration', 'velocity_2d_mm', etc.
    max_syllable (int): the index of the maximum number of syllables to include

    Returns:
    df_pivot (pandas.DataFrame): Mean syllable statistic per session, one column per syllable below max_syllable.
    A session that never used a syllable has a usage of 0 and a missing (NaN) value for every other statistic.
    """

    df_pivot = (
        df[df.syllable < max_syllable]
            .pivot_table(index=["group", "uuid"], columns="syllable", values=statistic)
            .reindex(columns=range(max_syllable))
    )
    if statistic == "usage":
        df_pivot = df_pivot.fillna(0)

    return df_pivot


def _at_least_as_extreme(null, observed):
    """
    Mark the permuted statistics that are at least as large as the observed one, ties included.

    A permutation that ties the observed statistic in exact arithmetic can be computed through a
    different sum and land a few units in the last place below it. As in scipy.stats.permutation_test,
    a permuted statistic within 100 machine epsilons of the observed one, relative to it, is a tie.

    Args:
    null (np.array): permuted statistics.
    observed (float or np.array): observed statistic, broadcastable against null; NaN matches nothing.

    Returns:
    (np.array): boolean array shaped like null.
    """

    gamma = 100 * np.finfo(np.float64).eps * np.abs(observed)
    return null >= observed - gamma


def bootstrap_me(usages, n_iters=10000):
    """
    Bootstrap the inputted stat data using random sampling with replacement.

    Args:
    usages (np.array): sessions x syllables; NaN marks a session that never used the syllable
    n_iters (int): Number of samples to return.

    Returns:
    boots (np.array): n_iters x syllables bootstrapped means, each syllable resampled from the sessions that used it (NaN where none did)
    """

    n_mice = usages.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        boots = np.nanmean(usages[np.random.choice(n_mice, size=(n_mice, n_iters))], axis=0)

    # a syllable some sessions never used is resampled from the sessions that did
    for s in np.flatnonzero(np.isnan(usages).any(axis=0)):
        observed = usages[~np.isnan(usages[:, s]), s]
        if observed.size == 0:
            boots[:, s] = np.nan
        else:
            boots[:, s] = observed[np.random.choice(observed.size, size=(observed.size, n_iters))].mean(0)
    return boots


def ztest_vect(d1, d2):
    """
    Perform a z-test on a pair of bootstrapped syllable statistics.

    Args:
    d1 (np.array): bootstrapped syllable stat array from group 1
    d2 (np.array): bootstrapped syllable stat array from group 2

    Returns:
    p-values (np.array): array of computed p-values of the syllables.
    """

    mu1 = d1.mean(0)
    mu2 = d2.mean(0)
    std1 = d1.std(0)
    std2 = d2.std(0)
    std = np.sqrt(std1 ** 2 + std2 ** 2)
    delta = np.abs(mu1 - mu2)
    z = np.divide(delta, std, out=np.zeros_like(delta, dtype=float), where=std > 0)
    p = np.minimum(1.0, 2 * stats.norm.cdf(-z))
    return np.where((std == 0) & (delta > 0), 0.0, p)


def bootstrap_group_means(df, group1, group2, statistic="usage", max_syllable=40):
    """
    compute boostrapped group means

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    group1 (str): Name of group 1 to compare.
    group2 (str): Name of group 2 to compare.
    statistic (str): Syllable statistic to compute bootstrap means for.
    max_syllable (int): the index of the maximum number of syllables to include

    Returns:
    boots (dictionary): dictionary of group name (keys) paired with their bootstrapped statistics numpy array.
    """

    # get separated group variables
    group_stat = get_session_mean_df(df, statistic, max_syllable)

    groups = (group1, group2)
    usages = {k: group_stat.loc[k].values for k in groups}
    boots = {k: bootstrap_me(v) for k, v in usages.items()}

    return boots


def get_tie_correction(x, N_m):
    """
   assign tied rank values to the average of the ranks they would have received if they had not been tied for Kruskal-Wallis helper function.

    Args:
    x (pd.Series): syllable usages for a single session.
    N_m (int): Number of total sessions.

    Returns:
    corrected_rank (float): average of the inputted tied rank
    """

    vc = x.value_counts()
    tie_sum = 0
    if (vc > 1).any():
        tie_sum += np.sum(vc[vc != 1] ** 3 - vc[vc != 1])
    return tie_sum / (12.0 * (N_m - 1))


def run_manual_KW_test(
        df_usage,
        merged_usages_all,
        num_groups,
        n_per_group,
        cum_group_idx,
        n_perm=10000,
        seed=0,
):
    """

    Run a manual Kruskal-Wallis test compare the results agree with the scipy.stats.s

    Args:
    df_usage (pandas.DataFrame): DataFrame containing only pre-computed syllable stats. shape = (N_m, n_syllables)
    merged_usages_all (np.array): numpy array format of the df_usage DataFrame.
    num_groups (int): Number of unique groups
    n_per_group (list): list of value counts for sessions per group. len == num_groups.
    cum_group_idx (list): list of indices for different groups. len == num_groups + 1.
    n_perm (int): Number of permuted samples to generate.
    seed (int): Random seed used to initialize the pseudo-random number generator.

    Returns:
    h_all (np.array): Array of H-stats computed for given n_syllables; shape = (n_perms, N_s)
    real_ranks (np.array): Array of syllable ranks, shape = (N_m, n_syllables)
    X_ties (np.array): 1-D list of tied ranks, where if value > 0, then rank is tied. len(X_ties) = n_syllables
    """

    N_m, N_s = merged_usages_all.shape

    # create random index array n_perm times
    rnd = np.random.RandomState(seed=seed)
    perm = rnd.rand(n_perm, N_m).argsort(-1)

    # get degrees of freedom
    dof = num_groups - 1

    real_ranks = np.apply_along_axis(stats.rankdata, 0, merged_usages_all)
    X_ties = df_usage.apply(get_tie_correction, 0, N_m=N_m).values
    KW_tie_correct = np.apply_along_axis(stats.tiecorrect, 0, real_ranks)

    # rank data
    perm_ranks = real_ranks[perm]

    # get square of sums for each group
    ssbn = np.zeros((n_perm, N_s))
    for i in range(num_groups):
        ssbn += (
                perm_ranks[:, cum_group_idx[i]: cum_group_idx[i + 1]].sum(1) ** 2
                / n_per_group[i]
        )

    # h-statistic; a syllable with one value in every session has none, and run_kruskal gives it p = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        h_all = 12.0 / (N_m * (N_m + 1)) * ssbn - 3 * (N_m + 1)
        h_all /= KW_tie_correct
        p_vals = stats.chi2.sf(h_all, df=dof)

    # check that results agree
    p_i = np.random.randint(n_perm)
    s_i = np.random.randint(N_s)
    if np.ptp(merged_usages_all[:, s_i]) > 0:
        kr = stats.kruskal(
            *np.array_split(
                merged_usages_all[perm[p_i, :], s_i], np.cumsum(n_per_group[:-1])
            )
        )
        assert (kr.statistic == h_all[p_i, s_i]) & (
                kr.pvalue == p_vals[p_i, s_i]
        ), "manual KW is incorrect"

    return h_all, real_ranks, X_ties


def _kruskal_h_permutations(values, labels, group_names, perm):
    """
    Compute the Kruskal-Wallis H of one syllable for each permutation, over the sessions that used it.

    Args:
    values (1D array): the syllable's statistic in every session; NaN where a session never used it.
    labels (1D array): each session's group.
    group_names (pd.Index): the groups, in the order their sessions appear.
    perm (2D array): n_perm x sessions permutations of all sessions; each row restricted to the sessions
        that used the syllable is a uniform permutation of them.

    Returns:
    h (1D array): one H statistic per permutation.
    """

    observed = ~np.isnan(values)
    n = int(observed.sum())
    ranks = np.full(len(values), np.nan)
    ranks[observed] = stats.rankdata(values[observed])

    counts = [int(np.sum(observed & (labels == group))) for group in group_names]
    counts = [count for count in counts if count > 0]
    edges = np.insert(np.cumsum(counts), 0, 0)

    perm_ranks = ranks[perm[observed[perm]].reshape(len(perm), n)]
    ssbn = sum(perm_ranks[:, edges[i]:edges[i + 1]].sum(1) ** 2 / counts[i] for i in range(len(counts)))
    h = 12.0 / (n * (n + 1)) * ssbn - 3 * (n + 1)
    return h / stats.tiecorrect(ranks[observed])


def plot_H_stat_significance(df_k_real, h_all, N_s):
    """
    Plot the assigned H-statistic for each syllable computed via manual KW test.

    Args:
    df_k_real (pandas.DataFrame): the dataframe that contains Kruskal-Wallis p value, H stats and whether it is significant
    h_all (np.array): Array of H-stats computed for given n_syllables; shape = (n_perms, N_s)
    N_s (int): Number of syllables to plot

    Returns:
    fig (pyplot.figure): plotted H-stats plot
    ax (pyplot.axis): plotted H-stats axis
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    pcts = np.percentile(h_all, [2.5, 97.5], axis=0)

    ax.fill_between(np.arange(N_s), pcts[0, :], pcts[1, :], alpha=0.2, color="k")

    ax.scatter(
        np.arange(N_s)[~df_k_real.is_sig],
        df_k_real.statistic[~df_k_real.is_sig],
        label="Not significant",
    )
    ax.scatter(
        np.arange(N_s)[df_k_real.is_sig],
        df_k_real.statistic[df_k_real.is_sig],
        label="Significant Syllable",
    )

    ax.set_xlabel("Syllable")
    ax.set_ylabel(f"$H$")
    ax.legend()
    sns.despine()

    return fig, ax


def run_kruskal(
        df,
        statistic="usage",
        max_syllable=40,
        n_perm=10000,
        seed=42,
        thresh=0.05,
        mc_method="fdr_bh",
        verbose=False,
):
    """
    Run Kruskal-Wallis Hypothesis test and Dunn's posthoc multiple comparisons test for syllable statistic.

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    statistic (str): statistic to compute mean for, (any of the columns in input df).
    max_syllable (int): the index of the maximum number of syllables to include
    n_perm (int): Number of permuted samples to generate.
    seed (int): Random seed used to initialize the pseudo-random number generator.
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.
    verbose (bool): indicates whether to print out the significant syllable results

    Returns:
    df_k_real (pandas.DataFrame): DataFrame of KW test results.
    dunn_results_df (pandas.DataFrame): DataFrame of Dunn's test results for permuted group pairs.
    intersect_sig_syllables (dict): dictionary containing intersecting significant syllables between KW and Dunn's tests.
    """

    rnd = np.random.RandomState(seed=seed)

    # get mean grouped data
    grouped_data = get_session_mean_df(df, statistic, max_syllable).reset_index()

    # KW Constants
    vc = grouped_data.group.value_counts().loc[grouped_data.group.unique()]
    n_per_group = vc.values
    group_names = vc.index

    cum_group_idx = np.insert(np.cumsum(n_per_group), 0, 0)
    num_groups = len(group_names)

    # NaN marks a session that never used a syllable. The vectorized permutations run on every
    # syllable with the gaps filled; syllables with gaps are then recomputed over the sessions
    # that used them, so complete syllables keep exactly the values they always had.
    values = grouped_data[range(max_syllable)].values
    group_labels = grouped_data.group.values
    incomplete = np.isnan(values).any(axis=0)
    df_only_usage = grouped_data[range(max_syllable)].fillna(0)
    merged_usages_all = df_only_usage.values

    N_m, N_s = merged_usages_all.shape

    # Run KW and return H-stats
    h_all, real_ranks, X_ties = run_manual_KW_test(
        df_usage=df_only_usage,
        merged_usages_all=merged_usages_all,
        num_groups=num_groups,
        n_per_group=n_per_group,
        cum_group_idx=cum_group_idx,
        n_perm=n_perm,
        seed=seed,
    )

    # each syllable compares the groups that used it, over the sessions that did; fewer than two
    # such groups, or one value everywhere, leaves nothing to test and gives p = 1
    kruskal_results, untestable = [], np.zeros(N_s, dtype=bool)
    for s_i in range(N_s):
        samples = [values[(group_labels == group) & ~np.isnan(values[:, s_i]), s_i] for group in group_names]
        samples = [sample for sample in samples if len(sample) > 0]
        if len(samples) < 2 or np.ptp(np.concatenate(samples)) == 0:
            untestable[s_i] = True
            kruskal_results.append((0.0, 1.0))
        else:
            kruskal_results.append(tuple(stats.kruskal(*samples)))
    df_k_real = pd.DataFrame(kruskal_results, columns=["statistic", "pvalue"])

    # a permutation p-value counts the permutations at least as extreme as the data, ties included
    p_perm = (_at_least_as_extreme(h_all, df_k_real.statistic.values).sum(0) + 1) / (n_perm + 1)
    perm = np.random.RandomState(seed=seed).rand(n_perm, N_m).argsort(-1)  # the draw run_manual_KW_test made
    for s_i in np.flatnonzero(incomplete & ~untestable):
        h_null = _kruskal_h_permutations(values[:, s_i], group_labels, group_names, perm)
        p_perm[s_i] = (_at_least_as_extreme(h_null, df_k_real.statistic[s_i]).sum() + 1) / (n_perm + 1)
    p_perm[untestable] = 1.0

    # https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    df_k_real["p_adj"] = multipletests(
        p_perm,
        alpha=thresh,
        method=mc_method,
    )[1]

    df_k_real["is_sig"] = df_k_real["p_adj"] <= thresh

    if verbose:
        print(f"Found {df_k_real['is_sig'].sum()} syllables that pass threshold {thresh} with {mc_method}")

    # Run Dunn's z-test statistics
    (
        null_zs_within_group,
        real_zs_within_group,
    ) = dunns_z_test_permute_within_group_pairs(
        grouped_data, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm, values=values
    )

    # Compute p-values from Dunn's z-score statistics
    df_pair_corrected_pvalues, _ = compute_pvalues_for_group_pairs(
        real_zs_within_group,
        null_zs_within_group,
        df_k_real,
        group_names,
        n_perm,
        thresh,
        mc_method,
    )

    # combine Dunn's test results into single DataFrame
    df_z = pd.DataFrame(real_zs_within_group)
    df_z.index = df_z.index.set_names("syllable")
    # The keys are group pairs, so the columns are a MultiIndex and reset_index
    # labels the syllable column ('syllable', ''). Selecting that full label and
    # renaming it back gives the same frame the bare string used to select:
    # columns syllable, variable_0, variable_1, value.
    flattened = df_z.reset_index()
    id_columns = [column for column in flattened.columns
                  if (column[0] if isinstance(column, tuple) else column)
                  == "syllable"]
    dunn_results_df = flattened.melt(id_vars=id_columns)
    dunn_results_df = dunn_results_df.rename(columns={id_columns[0]: "syllable"})

    if verbose:
        # take the intersection of KW and Dunn's tests
        print("permutation within group-pairs (2 groups)")
        print("intersection of Kruskal-Wallis and Dunn's tests")

    # Get intersecting significant syllables between
    intersect_sig_syllables = {}
    for pair in df_pair_corrected_pvalues.columns.tolist():
        intersect_sig_syllables[pair] = np.where(
            (df_pair_corrected_pvalues[pair] < thresh) & (df_k_real.is_sig)
        )[0]

        if verbose:
            print(pair, len(intersect_sig_syllables[pair]), intersect_sig_syllables[pair])

    return df_k_real, dunn_results_df, intersect_sig_syllables


def compute_pvalues_for_group_pairs(
        real_zs_within_group,
        null_zs,
        df_k_real,
        group_names,
        n_perm=10000,
        thresh=0.05,
        mc_method="fdr_bh",
        verbose=False
):
    """
    Adjust the p-values from Dunn's z-test statistics and computes the resulting significant syllables with the adjusted p-values.

    Args:
    real_zs_within_group (dict): dict of group pair keys paired with vector of Dunn's z-test statistics
    null_zs  (dict): dict of group pair keys paired with vector of Dunn's z-test statistics of the null hypothesis.
    df_k_real (pandas.DataFrame): DataFrame of KW test results.
    group_names (pd.Index): Index list of unique group names.
    n_perm (int): Number of permuted samples to generate.
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.
    verbose (bool): indicates whether to print out the significant syllable results

    Returns:
    df_pval_corrected (pandas.DataFrame): DataFrame containing Dunn's test results with corrected p-values.
    significant_syllables (list): List of corrected KW significant syllables (syllables with p-values < thresh)
    """

    # do empirical p-val calculation for all group permutation
    if verbose:
        print(f"Permutation across all {len(group_names)} groups")
        print(f"significant syllables for each pair with FDR < {thresh}")

    p_vals_allperm = {}
    for pair in combinations(group_names, 2):
        # permutations at least as extreme as the data, ties included
        p_vals_allperm[pair] = (
                                       _at_least_as_extreme(null_zs[pair], real_zs_within_group[pair]).sum(0) + 1
                               ) / (n_perm + 1)
        # a pair with no z for a syllable (a group never used it, or one value everywhere) cannot be compared
        p_vals_allperm[pair][np.isnan(real_zs_within_group[pair])] = 1.0

    # summarize into df
    df_pval = pd.DataFrame(p_vals_allperm)

    correct_p = lambda x: multipletests(x, alpha=thresh, method=mc_method)[1]
    df_pval_corrected = df_pval.apply(correct_p, axis=1, result_type="broadcast")

    return df_pval_corrected, ((df_pval_corrected[df_k_real.is_sig] < thresh).sum(0))


def dunns_z_test_permute_within_group_pairs(
        df_usage, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm, values=None
):
    """
    Run Dunn's z-test statistic on combinations of all group pairs, handling pre-computed tied ranks.

    Args:
    df_usage (pandas.DataFrame): DataFrame containing only pre-computed syllable stats.
    vc (pd.Series): value counts of sessions in each group.
    real_ranks (np.array): Array of syllable ranks.
    X_ties (np.array): 1-D list of tied ranks, where if value > 0, then rank is tied
    N_m (int): Number of sessions.
    group_names (pd.Index): Index list of unique group names.
    rnd (np.random.RandomState): Pseudo-random number generator.
    n_perm (int): Number of permuted samples to generate.
    values (np.array): sessions x syllables statistic with NaN where a session never used a syllable; those
        syllables are recomputed over the sessions that used them, and a pair with a group that never used
        a syllable gets a NaN z for it.

    Returns:
    null_zs_within_group (dict): dict of group pair keys paired with vector of Dunn's z-test statistics of the null hypothesis.
    real_zs_within_group (dict): dict of group pair keys paired with vector of Dunn's z-test statistics
    """

    null_zs_within_group = {}
    real_zs_within_group = {}

    A = N_m * (N_m + 1.0) / 12.0

    for (i_n, j_n) in combinations(group_names, 2):
        is_i = df_usage.group == i_n
        is_j = df_usage.group == j_n

        n_mice = is_i.sum() + is_j.sum()

        perm = rnd.rand(n_perm, n_mice).argsort(-1)
        ranks_perm = real_ranks[(is_i | is_j)][perm]
        diff = np.abs(
            ranks_perm[:, : is_i.sum(), :].mean(1)
            - ranks_perm[:, is_i.sum():, :].mean(1)
        )
        B = 1.0 / vc.loc[i_n] + 1.0 / vc.loc[j_n]

        # also do for real data
        group_ranks = real_ranks[(is_i | is_j)]
        real_diff = np.abs(
            group_ranks[: is_i.sum(), :].mean(0) - group_ranks[is_i.sum():, :].mean(0)
        )

        # add to dict
        pair = (i_n, j_n)
        with np.errstate(divide="ignore", invalid="ignore"):
            null_zs_within_group[pair] = diff / np.sqrt((A - X_ties) * B)
            real_zs_within_group[pair] = real_diff / np.sqrt((A - X_ties) * B)

        if values is None:
            continue
        # a syllable some sessions never used: ranks, sizes and ties over the sessions that did,
        # with the same permutations restricted to them
        pair_rows = np.flatnonzero((is_i | is_j).values)
        n_i_rows = int(is_i.sum())
        for s in np.flatnonzero(np.isnan(values).any(axis=0)):
            column = values[:, s]
            observed = ~np.isnan(column)
            pair_observed = observed[pair_rows]
            n_i, n_j = int(pair_observed[:n_i_rows].sum()), int(pair_observed[n_i_rows:].sum())
            n = int(observed.sum())
            if n_i == 0 or n_j == 0 or np.ptp(column[observed]) == 0:
                null_zs_within_group[pair][:, s] = np.nan
                real_zs_within_group[pair][s] = np.nan
                continue
            ranks = np.full(len(column), np.nan)
            ranks[observed] = stats.rankdata(column[observed])
            pair_ranks = ranks[pair_rows]
            scale = np.sqrt((n * (n + 1.0) / 12.0 - get_tie_correction(pd.Series(column[observed]), n))
                            * (1.0 / n_i + 1.0 / n_j))
            permuted = pair_ranks[perm[pair_observed[perm]].reshape(n_perm, n_i + n_j)]
            null_zs_within_group[pair][:, s] = np.abs(permuted[:, :n_i].mean(1) - permuted[:, n_i:].mean(1)) / scale
            kept = pair_ranks[pair_observed]
            real_zs_within_group[pair][s] = np.abs(kept[:n_i].mean() - kept[n_i:].mean()) / scale

    return null_zs_within_group, real_zs_within_group


def run_pairwise_stats(df, group1, group2, test_type="mw", verbose=False, **kwargs):
    """
    Run hypothesis testing functions: MannWhitney, Z-Test and T-Test.

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    group1 (str): Name of first group
    group2 (str): Name of second group
    test_type (str): specifying which type of statistical test
    verbose (bool): boolean flag that indicates whether to print out the significant syllable results

    Returns:
        df_pvals (pandas.DataFrame): Dataframe listing the p-values and which syllables are significant
    """
    test_types = ["mw", "z_test", "t_test"]
    if test_type not in test_types:
        raise ValueError(f"`test_type` must one of {test_types}")
    if test_type == "mw":
        return mann_whitney(df, group1, group2, verbose=verbose, **kwargs)
    elif test_type == "z_test":
        return ztest(df, group1, group2, verbose=verbose, **kwargs)
    elif test_type == "t_test":
        return ttest(df, group1, group2, verbose=verbose, **kwargs)


def _check_groups(df, *groups):
    """
    Raise a ValueError naming a group the table does not have, and the groups it does.

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    groups (str): the names of the groups to compare.
    """

    present = set(df.group.unique())
    for group in groups:
        if group not in present:
            names = ", ".join(sorted(map(str, present)))
            raise ValueError(f"Group '{group}' is not in the data. Its groups are: {names}.")


MannwhitneyuResult = namedtuple("MannwhitneyuResult", ("statistic", "pvalue"))


def _untestable(*groups):
    """
    Find the syllables a group comparison cannot evaluate.

    Args:
    groups (2D arrays): sessions x syllables for each group; NaN marks a session that never used the syllable.

    Returns:
    (1D bool array): one entry per syllable, True where a group has no session that used it or every
    observed value is the same, so there is no difference to test.
    """

    empty = np.zeros(groups[0].shape[1], dtype=bool)
    for values in groups:
        empty |= np.isnan(values).all(axis=0)
    observed = np.concatenate(groups, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanmax(observed, axis=0) - np.nanmin(observed, axis=0)
    return empty | ~(spread > 0)


def mannwhitneyu(x, y):
    """
    Run a Mann-Whitney U test, returning the smaller U and a one-sided p-value.

    Args:
    x (1D array): observations from the first sample.
    y (1D array): observations from the second sample.

    Returns:
    (MannwhitneyuResult): the smaller of the two U statistics, and the p-value
    for the larger one under a normal approximation with a continuity correction
    and a correction for ties. Named so that a frame built from a column of
    these results carries the `statistic` and `pvalue` names its readers use.

    The p-value is one-sided, and which side is decided by the data rather than
    by the caller: it is always computed for whichever sample ranks higher. That
    makes it half the two-sided value, so a result is twice as easy to call
    significant as the two-sided test everyone assumes is running. This
    reproduces the statistic every published moseq2 syllable comparison was made
    with, and changing it is a decision about the science rather than about
    which version of scipy is installed.
    """

    x, y = np.asarray(x), np.asarray(y)
    n1, n2 = len(x), len(y)
    ranked = stats.rankdata(np.concatenate((x, y)))
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(ranked[:n1], axis=0)
    u2 = n1 * n2 - u1
    tie_correction = stats.tiecorrect(ranked)
    if tie_correction == 0:
        raise ValueError("All numbers are identical in mannwhitneyu")
    sd = np.sqrt(tie_correction * n1 * n2 * (n1 + n2 + 1) / 12.0)
    # The continuity correction is applied unconditionally, including where the
    # two U statistics are equal; scipy scales it by the sign of z instead,
    # which drops it exactly at that point.
    z = (max(u1, u2) - (n1 * n2 / 2.0 + 0.5)) / sd
    return MannwhitneyuResult(min(u1, u2), stats.norm.sf(abs(z)))


def mann_whitney(df, group1, group2, statistic="usage", max_syllable=40, verbose=False, **kwargs):
    """
    Runs a Mann-Whitney hypothesis test with multiple test corrections on two given groups to find significant syllables.
    Also runs multiple corrections test to find syllables to exclude.

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    group1 (str): Name of first group
    group2 (str): Name of second group
    statistic (str): Name of statistic to compute z-test on.
    max_syllable (int): the index of the maximum number of syllables to include
    verbose (bool): indicates whether to print out the significant syllable results
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns:
    df_mw_real (pandas.DataFrame): DataFrame containing Mann-Whitney U corrected results.
    exclude_sylls (list): list of syllables that were excluded via multiple comparisons test.
    """
    _check_groups(df, group1, group2)
    # get mean grouped data
    grouped_data = get_session_mean_df(df, statistic, max_syllable).reset_index()

    values = {
        group: grouped_data.loc[grouped_data.group == group, range(max_syllable)].values
        for group in (group1, group2)
    }
    # each syllable is ranked over the sessions that used it; one a group never used,
    # or with one value everywhere, has nothing to rank and gets p = 1
    untestable = _untestable(values[group1], values[group2])
    results = []
    for i in range(max_syllable):
        x, y = values[group1][:, i], values[group2][:, i]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        results.append(MannwhitneyuResult(len(x) * len(y) / 2.0, 1.0) if untestable[i] else mannwhitneyu(x, y))
    df_mw_real = pd.DataFrame(results)
    return get_sig_syllables(df_mw_real, verbose=verbose, **kwargs)


def ztest(df, group1, group2, statistic="usage", max_syllable=40, verbose=False, **kwargs):
    """
    Computes a z hypothesis test on 2 (bootstrapped) selected groups with multiple test correction.

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    group1 (str): Name of first group
    group2 (str): Name of second group
    statistic (str): Name of statistic to compute z-test on.
    max_syllable (int): the index of the maximum number of syllables to include
    verbose (bool): boolean flag that indicates whether to print out the significant syllable results
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns:
    pvals_ztest_boots (np.array): Computed array of p-values
    syllables_to_include (list): List of significant syllables after multiple corrections.
    """
    _check_groups(df, group1, group2)
    boots = bootstrap_group_means(df, group1, group2, statistic, max_syllable)
    # do a ztest on the bootstrap distributions of your 2 conditions
    pvals_ztest_boots = ztest_vect(boots[group1], boots[group2])
    # a group with fewer than two sessions that used a syllable gives a bootstrap with no spread,
    # which says nothing about its variance, so that syllable has nothing to test and gets p = 1
    group_stat = get_session_mean_df(df, statistic, max_syllable)
    too_few = (group_stat.loc[group1].notna().sum(0).values < 2) | (group_stat.loc[group2].notna().sum(0).values < 2)
    pvals_ztest_boots[too_few] = 1.0
    df_z = pd.DataFrame(pvals_ztest_boots, columns=["pvalue"])
    return get_sig_syllables(df_z, verbose=verbose, **kwargs)


def ttest(df, group1, group2, statistic="usage", max_syllable=40, verbose=False, **kwargs):
    """
    Computes a t-hypothesis test on 2 selected groups to find significant syllables with multiple test correction.

    Args:
    df (pandas.DataFrame): dataframe that contains syllable statistics per session (mean_df/stats_df)
    group1 (str): Name of first group
    group2 (str): Name of second group
    statistic (str): Name of statistic to compute t-test on.
    max_syllable (int): the index of the maximum number of syllables to include
    verbose (bool): indicates whether to print out the significant syllable results.
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns:
    p (np.array): Computed array of p-values
    syllables_to_include (list): List of significant syllables after multiple corrections.
    """
    _check_groups(df, group1, group2)
    # get separated group variables
    group_stat = get_session_mean_df(df, statistic, max_syllable)
    groups = (group1, group2)
    usages = {k: group_stat.loc[k].values for k in groups}
    # run t-test
    # each syllable is tested over the sessions that used it; scipy warns about the
    # syllables with too few, which get p = 1 below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.ttest_ind(usages[group1], usages[group2], nan_policy="omit")
    df_t = pd.DataFrame(result, index=["t", "pvalue"]).T
    # a syllable a group never used, with one value everywhere, or too few sessions for a
    # t statistic has nothing to test, and gets p = 1
    df_t.loc[_untestable(usages[group1], usages[group2]), "pvalue"] = 1.0
    df_t["pvalue"] = df_t["pvalue"].fillna(1.0)
    return get_sig_syllables(df_t, verbose=verbose, **kwargs)


def get_sig_syllables(df_pvals, thresh=0.05, mc_method="fdr_bh", verbose=False):
    """
    Runs multiple p-value comparisons test given a set alpha threshold with mutliple test correction.

    Args:
    df_pvals (pandas.DataFrame): dataframe listing raw p-values
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.
    verbose (bool): indicates whether to print out the significant syllable results

    Returns:
    df_pvals (pandas.DataFrame): updated dataframe listing adjusted p-values
    """
    df_pvals["p_adj"] = multipletests(df_pvals.pvalue, alpha=thresh, method=mc_method)[
        1
    ]
    df_pvals["is_sig"] = df_pvals["p_adj"] <= thresh
    n_sig = df_pvals["is_sig"].sum()

    # print list of significant syllables
    sig_sylls = list(df_pvals[df_pvals["is_sig"] == True].index)

    if verbose:
        print(f"Found {n_sig} syllables that pass threshold {thresh} with {mc_method}")
        print(sig_sylls)

    return df_pvals
