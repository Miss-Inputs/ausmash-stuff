#!/usr/bin/env python3

"""Generates a ranking of active players in a region based on tournament results, and stats and summaries of those tournament results.

Active players are decided on who has attended enough events in the given season.
Can output results cosnidering ACT locals only, majors only (where tournaments are considered majors on Ausmash, which is not checked, and may be inaccurate), "regionals" only (where a regional is considered to be any tournament that has players from at least 3 different regions in attendance).

The ranking is based on giving a score to each player's result at each tournament based on how many rounds they went through, and using the mean of those scores per player.
"""

import logging
import operator
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import NamedTuple, cast

import pandas
import scipy.stats
from ausmash import (
	Game,
	Match,
	Player,
	Region,
	Result,
	Tournament,
	get_active_players,
	rounds_from_victory,
)
from tier_lister import BaseTierList, TextBoxTierList
from tqdm import tqdm

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def get_relevant_player_results(
	player: Player,
	game: Game | str,
	season_start: date | None = None,
	season_end: date | None = None,
	event_size_to_count: int = 1,
	series_to_exclude: Iterable[str] | None = None,
) -> Sequence[Result]:
	"""Gets only the results for a player that are wanted for comparison purposes

	Returns:
		List of Result, newest to oldest
	"""
	if series_to_exclude is None:
		series_to_exclude = []
	results = Result.results_for_player(player, season_start, season_end)
	# Exclude pro bracket results as we will get the pools result for the whole tournament
	if isinstance(game, str):
		game = Game(game)
	return [
		r
		for r in results
		if r.event.game == game
		and not r.event.is_side_bracket
		and not r.event.is_redemption_bracket
		and r.total_entrants >= event_size_to_count
		and r.tournament.series.name not in series_to_exclude
		and not r.tournament.previous_phase_for_event(r.event)
	]


def get_redemption_bracket_results(
	player: Player,
	game: Game | str,
	season_start: date | None = None,
	season_end: date | None = None,
	event_size_to_count: int = 1,
	series_to_exclude: Iterable[str] | None = None,
) -> Sequence[Result]:
	"""Gets only the results for a player that are for redemption bracket"""
	if series_to_exclude is None:
		series_to_exclude = []
	results = Result.results_for_player(player, season_start, season_end)
	return [
		r
		for r in results
		if r.event.game == game
		and r.event.is_redemption_bracket
		and r.total_entrants >= event_size_to_count
		and r.tournament.series.name not in series_to_exclude
	]


def _get_rows(
	players: Iterable[Player],
	game: Game | str,
	season_start: date | None = None,
	season_end: date | None = None,
	event_size_to_count: int = 1,
	excluded_series: Iterable[str] | None = None,
	*,
	redemption: bool = False,
) -> dict[Player, dict[tuple[Tournament, str], int | tuple[int, int]]]:
	rows: dict[Player, dict[tuple[Tournament, str], int | tuple[int, int]]] = {}

	with tqdm(players, 'Getting tournament results', unit='player') as t:
		for player in t:
			t.set_postfix(player=player)
			results = (
				get_redemption_bracket_results(
					player, game, season_start, season_end, event_size_to_count, excluded_series
				)
				if redemption
				else get_relevant_player_results(
					player, game, season_start, season_end, event_size_to_count, excluded_series
				)
			)

			data: dict[tuple[Tournament, str], int | tuple[int, int]] = {}
			for result in results:
				player_matches_at_event = [
					m for m in Match.matches_at_event(result.event) if player in m.players
				]
				if not player_matches_at_event:
					# Player DQd out of the whole event and so didn't really attend it
					continue

				placing = (result.real_placing, result.total_entrants)
				score = rounds_from_victory(result.total_entrants) - rounds_from_victory(placing[0])

				if score == 1 and not any(m.winner == player for m in player_matches_at_event):
					# Player went 0-2, but ended up in losers round 2 due to seeding
					# Not sure that should count
					score = 0
				elif score == 0 and any(m.winner == player for m in player_matches_at_event):
					# Player went (probably) 1-2, winning WR1 and losing WR2 and due to seeding going to LR1 and losing there
					# Kinda feel like that should be a point
					score = 1

				# Nobody would normally have a dict like this, but this is how we get pandas to make it into a MultiIndex
				data[result.tournament, 'Placing'] = placing
				data[result.tournament, 'Score'] = score

			rows[player] = data
	return rows


def expectile_nan(s: pandas.Series, alpha: float) -> float:
	return float(scipy.stats.expectile(s.dropna(), alpha=alpha))


def trimmean_nan(s: pandas.Series, proportion: float) -> float:
	return float(scipy.stats.trim_mean(s.dropna(), proportion))


def _get_stats(
	scores: pandas.DataFrame,
	placings: pandas.DataFrame,
	events_to_count: int,
	confidence_percent: float = 0.95,
	sort_column: str | None = None,
	*,
	drop_zero_score: bool = False,
) -> pandas.DataFrame:
	scores.dropna(how='all', inplace=True)
	placings.dropna(how='all', inplace=True)
	# scores: pandas.DataFrame = df.loc[:, (slice(None), 'Score')].droplevel(1, axis='columns')
	# placings: pandas.DataFrame = df.loc[:, (slice(None), 'Placing')].droplevel(1, axis='columns')

	best = scores.idxmax(axis='columns').map(_abbrev_name, 'ignore')
	worst = scores.idxmin(axis='columns').map(_abbrev_name, 'ignore')
	count = scores.count(axis='columns')
	total = scores.sum(axis='columns')
	mean = scores.mean(axis='columns')
	median = scores.median(axis='columns')
	stdev = scores.std(axis='columns')
	count_below_mean = scores[scores.lt(mean, axis='index')].count(axis='columns')
	portion_below_mean = count_below_mean / count
	count_below_median = scores[scores.lt(median, axis='index')].count(axis='columns')
	portion_below_median = count_below_median / count

	sem = scores.sem(axis='columns', skipna=True)
	raw_zscores = scipy.stats.zscore(
		scores.astype(float), nan_policy='omit'
	)  # Need nan instead of NAType
	zscores = pandas.DataFrame(raw_zscores, index=scores.index, columns=scores.columns)
	maxes = scores.max(axis='columns', skipna=True)
	mins = scores.min(axis='columns', skipna=True)
	midpoint = (maxes + mins) / 2
	count_below_midpoint = scores[scores.lt(midpoint, axis='index')].count(axis='columns')
	portion_below_midpoint = count_below_midpoint / count

	# TODO: ???? What was this supposed to do
	# if count.min() == 1:
	# 	# Whoopsie no stats involving sem for you
	# 	confidence_percent = 0
	# 	if sort_column in {'low', 'high'}:
	# 		sort_column = None
	if confidence_percent:
		z = scipy.stats.norm.ppf(1 - (1 - confidence_percent) / 2)  # Should be ~= 1.9599 for 95%
		interval_high = mean + (z * sem)
		interval_low = mean - (z * sem)
		interval_low.loc[interval_low < 0] = 0

	_first_item_getter = operator.itemgetter(0)
	wins = (placings.map(_first_item_getter, na_action='ignore') <= 1).sum(axis='columns')
	top_3s = (placings.map(_first_item_getter, na_action='ignore') <= 3).sum(axis='columns')
	top_8s = (placings.map(_first_item_getter, na_action='ignore') <= 8).sum(axis='columns')
	last_places = (scores == 0).sum(axis='columns')
	win_portion = wins / count
	top_3_portion = top_3s / count
	top_8_portion = top_8s / count
	last_place_portion = last_places / count

	cols = {
		'Best': best,
		'Worst': worst,
		'# attended': count,
		'Total score': total,
		'Mean score': mean,
		'Median score': median,
		'Standard deviation': stdev,
		'Wins': wins,
		'Top 3s': top_3s,
		'Top 8s': top_8s,
		'# last place': last_places,
		'Win %': win_portion,
		'Top 3 %': top_3_portion,
		'Top 8 %': top_8_portion,
		'Last place %': last_place_portion,
		# Some less used stats down here
		'Median': scores.where(scores.isin(median)).apply(
			pandas.Series.first_valid_index, axis='columns'
		),
		'Midpoint tournament': scores.where(scores.isin(midpoint)).apply(
			pandas.Series.first_valid_index, axis='columns'
		),
		'# below mean': count_below_mean,
		'# below median': count_below_median,
		'# below midpoint': count_below_midpoint,
		'% below mean': portion_below_mean,
		'% below median': portion_below_median,
		'% below midpoint': portion_below_midpoint,
		'Most inlier': zscores.abs().idxmin(axis=1, skipna=True).map(_abbrev_name, 'ignore'),
		'Most outlier': zscores.abs().idxmax(axis=1, skipna=True).map(_abbrev_name, 'ignore'),
		'Standout performance': zscores.idxmax(axis=1, skipna=True).map(
			_abbrev_name, 'ignore'
		),  # Usually the same as best tournament
		'Low outlier': zscores.idxmin(axis=1, skipna=True).map(
			_abbrev_name, 'ignore'
		),  # Usually the same as worst?
		'Standard error of mean': sem,
		'Kurtosis': scores.kurt(axis='columns', skipna=True),
		'Skew': scores.skew(axis='columns', skipna=True),
		'Range': maxes - mins,
		'Midpoint': midpoint,
		'Median absolute deviation': scores.subtract(median, axis='index')
		.abs()
		.median(axis='columns'),
		'Interquartile range': scipy.stats.iqr(
			scores.astype(float), axis=1, nan_policy='omit'
		),  # Need nan instead of NAType
		'25% expectile': scores.apply(expectile_nan, axis='columns', alpha=0.25),
		'75% expectile': scores.apply(expectile_nan, axis='columns', alpha=0.75),
		'Geometric mean': scipy.stats.gmean(scores.astype(float), axis=1, nan_policy='omit'),
		'Geometric mean of +1': scipy.stats.gmean(
			scores.astype(float) + 1, axis=1, nan_policy='omit'
		)
		- 1,
		'Harmonic mean': scipy.stats.hmean(scores.astype(float), axis=1, nan_policy='omit'),
		'Harmonic mean of +1': scipy.stats.hmean(
			scores.astype(float) + 1, axis=1, nan_policy='omit'
		)
		- 1,
		'10% trimmed mean': scores.apply(trimmean_nan, proportion=0.1, axis='columns'),
		'Coefficient of variation': stdev / mean,
	}

	df = pandas.DataFrame(cols)

	if confidence_percent:
		df.insert(
			df.columns.get_loc('Mean score') + 1,
			f'{confidence_percent:.0%} confidence interval low',
			interval_low,
		)
		df.insert(
			df.columns.get_loc('Mean score') + 2,
			f'{confidence_percent:.0%} confidence interval high',
			interval_high,
		)

	df.drop(index=count[count < events_to_count].index, inplace=True)
	if drop_zero_score:
		df.drop(
			index=total[total == 0].index, inplace=True, errors='ignore'
		)  # Ignore rows that were dropped for inactivity anyway

	if sort_column is None:
		sort_column = 'Mean score'
	elif confidence_percent and sort_column == 'low':
		sort_column = f'{confidence_percent:.0%} confidence interval low'
	elif confidence_percent and sort_column == 'high':
		sort_column = f'{confidence_percent:.0%} confidence interval high'

	df.insert(0, sort_column, df.pop(sort_column))

	df.insert(0, 'Rank', df[sort_column].rank(ascending=False, method='min').astype(int))
	df.sort_values(sort_column, ascending=False, inplace=True)

	diff = df[sort_column].diff(-1)
	df.insert(2, 'Difference to next', diff)

	return df.dropna(axis='columns', how='all')


def _format_placing_tuple(placing: tuple[int, int]) -> str:
	return f'{placing[0]}/{placing[1]}'


class ScoresAndStats(NamedTuple):
	scores: pandas.DataFrame
	placings: pandas.DataFrame
	stats: pandas.DataFrame


def get_scores_and_stats(
	active_players: Iterable[Player],
	game: Game | str,
	season_start: date | None,
	season_end: date | None,
	event_size_to_count: int,
	excluded_series: Iterable[str],
	minimum_events_to_count: int,
	*,
	drop_zero_score: bool = False,
	redemption: bool = False,
	confidence_percent: float = 0.95,
	sort_column: str | None = None,
) -> ScoresAndStats:
	rows = _get_rows(
		active_players,
		game,
		season_start,
		season_end,
		event_size_to_count,
		excluded_series,
		redemption=redemption,
	)

	df = pandas.DataFrame.from_dict(rows, orient='index')
	df = df.reindex(
		columns=pandas.Index(
			sorted(
				cast('pandas.Index[tuple[Tournament, str]]', df.columns),
				key=lambda t: t[0].date,
				reverse=True,
			)
		)
	).convert_dtypes()
	df.index.name = 'Player'
	if not df.empty:
		df.columns = df.columns.set_names('Tournament', level=0)

	scores: pandas.DataFrame = df.loc[:, (slice(None), 'Score')].droplevel(1, axis='columns')
	placings: pandas.DataFrame = df.loc[:, (slice(None), 'Placing')].droplevel(1, axis='columns')
	stats = _get_stats(
		scores,
		placings,
		minimum_events_to_count,
		confidence_percent,
		sort_column,
		drop_zero_score=drop_zero_score,
	)
	return ScoresAndStats(scores, placings, stats)


def _abbrev_name(t: Tournament) -> str:
	return t.abbrev_name


def output_stats(
	output_folder: Path | None,
	active_players: Iterable[Player],
	game: Game | str,
	region: Region | None = None,
	season_start: date | None = None,
	season_end: date | None = None,
	event_size_to_count: int = 1,
	excluded_series: Iterable[str] | None = [],
	minimum_events_to_count: int = 3,
	output_prefix: str = 'Tournament result',
	*,
	drop_zero_score: bool = False,
	output_dates: bool = False,  # TODO: Forgot to implement that whoops!
	redemption: bool = False,
	confidence_percent: float = 0.95,
	sort_column: str | None = None,
):
	scores, placings, stats = get_scores_and_stats(
		active_players,
		game,
		season_start,
		season_end,
		event_size_to_count,
		excluded_series or (),
		minimum_events_to_count,
		drop_zero_score=drop_zero_score,
		redemption=redemption,
		confidence_percent=confidence_percent,
		sort_column=sort_column,
	)

	suffix = f' {region.short_name}' if region else ''
	if redemption:
		suffix += ' redemption'

	logger.info(stats)
	stats.to_csv(
		output_folder / f'{output_prefix} stats{suffix}.csv' if output_folder else sys.stdout
	)
	if output_folder:
		# Reindexing by stats.index is just to sort it
		scores.reindex(index=stats.index).rename(columns=_abbrev_name).to_csv(
			output_folder / f'{output_prefix} scores{suffix}.csv'
		)
		placings.rename(columns=_abbrev_name).map(
			_format_placing_tuple, na_action='ignore'
		).reindex(index=stats.index).to_csv(output_folder / f'{output_prefix} placings{suffix}.csv')
		tier_list: BaseTierList[Player] = TextBoxTierList.from_items(
			stats['Mean score'], append_minmax_to_tier_titles=True, score_formatter='.4g'
		)
		tier_list.to_image(max_images_per_row=5, show_scores=True).save(
			output_folder / 'Tier lists' / f'{output_prefix}{suffix} tiered by mean.png'
		)
		if confidence_percent:
			tier_list = TextBoxTierList.from_items(
				stats[f'{confidence_percent:.0%} confidence interval low'],
				append_minmax_to_tier_titles=True,
				score_formatter='.4g',
			)
			tier_list.to_image(max_images_per_row=5, show_scores=True).save(
				output_folder
				/ 'Tier lists'
				/ f'{output_prefix}{suffix} tiered by confidence interval low.png'
			)
			tier_list = TextBoxTierList.from_items(
				stats[f'{confidence_percent:.0%} confidence interval high'],
				append_minmax_to_tier_titles=True,
				score_formatter='.4g',
			)
			tier_list.to_image(max_images_per_row=5, show_scores=True).save(
				output_folder
				/ 'Tier lists'
				/ f'{output_prefix}{suffix} tiered by confidence interval high.png'
			)

	if output_folder:
		# TODO: This should all be optional via arguments

		# TODO: Do this for other regions too, or some kind of only_include_series option
		if not region or region.short_name == 'ACT':
			locals_columns = scores.columns[
				scores.columns.get_level_values(0)
				.map(_abbrev_name, na_action='ignore')
				.str.contains('UPOV|Sauce|DB', regex=True)
			]

			if not locals_columns.empty:
				locals_only_scores = scores.loc[:, locals_columns]
				locals_only_placings = placings.loc[:, locals_columns]
				locals_only_stats = _get_stats(
					locals_only_scores,
					locals_only_placings,
					minimum_events_to_count,
					confidence_percent,
					sort_column,
					drop_zero_score=drop_zero_score,
				)
				logger.info('Locals only:')
				logger.info(locals_only_stats)
				locals_only_stats.to_csv(
					output_folder / 'Tournament result ACT locals only stats.csv'
				)
				locals_only_scores.rename(columns=_abbrev_name).reindex(
					index=locals_only_stats.index
				).to_csv(output_folder / 'Tournament result ACT locals only scores.csv')

		majors_columns = scores.columns.map(lambda t: t.is_major)
		majors_only_scores = scores.loc[:, majors_columns].copy()
		if not majors_only_scores.isna().all().all():
			majors_only_placings = placings.loc[:, majors_columns].copy()
			majors_only_stats = _get_stats(
				majors_only_scores,
				majors_only_placings,
				1,
				confidence_percent,
				sort_column,
				drop_zero_score=drop_zero_score,
			)
			logger.info('Majors only:')
			logger.info(majors_only_stats)
			majors_only_stats.to_csv(
				output_folder / f'{output_prefix} stats{suffix} majors only.csv'
			)
			majors_only_scores.rename(columns=_abbrev_name).reindex(
				index=majors_only_stats.index
			).to_csv(output_folder / f'{output_prefix} scores{suffix} majors only.csv')

		minimum_for_regional = 3  # TODO This should probably be an argument too
		regionals_columns = scores.apply(
			#FIXME: This means that it will only count big tournaments as regionals if there is 3 regions out of active_players, and not out of everyone who attended, which is often wrong/nonsense when active_players is some specific subset of players
			lambda column: (
				column.groupby(lambda player: player.region, sort=False).count() > 0
			).sum()
			>= minimum_for_regional,
			axis='index',
		)

		regionals_only_scores = scores.loc[:, regionals_columns].copy()
		if not regionals_only_scores.isna().all().all():
			regionals_only_placings = placings.loc[:, regionals_columns].copy()
			regionals_only_stats = _get_stats(
				regionals_only_scores,
				regionals_only_placings,
				minimum_events_to_count,
				confidence_percent,
				sort_column,
				drop_zero_score=drop_zero_score,
			)
			logger.info('Regionals only:')
			logger.info(regionals_only_stats)
			regionals_only_stats.to_csv(
				output_folder / f'{output_prefix} stats{suffix} regionals only.csv'
			)
			regionals_only_scores.rename(columns=_abbrev_name).reindex(
				index=regionals_only_stats.index
			).to_csv(output_folder / f'{output_prefix} scores{suffix} regionals only.csv')


def main():
	argparser = ArgumentParser(description=__doc__)

	# TODO: Catch errors while getting this, so that even if ausmash lib is not set up with an API key yet, this still prints basic usage
	game_choices = Game.all()
	argparser.add_argument(
		'--game',
		type=Game.from_name,
		help='Name or acronym of game to generate ranking for, by default, SSBU',
		default=Game('SSBU'),
		choices=game_choices,
		metavar=str({g.short_name for g in game_choices}),
	)
	region_choices = Region.all()
	argparser.add_argument(
		'--region',
		type=Region.from_name,
		help='Name or acronym of region to generate ranking for, or if not specified, generate ranking for every player on Ausmash',
		default=None,
		choices=region_choices,
		metavar='<region>',
	)
	argparser.add_argument(
		'--season-start',
		type=date.fromisoformat,
		help='Start date for this ranking season, in ISO format, or consider all tournaments if not specified',
		metavar='YYYY-MM-DD',
		default=None,
	)
	argparser.add_argument(
		'--season-end',
		type=date.fromisoformat,
		help='End date for this season, in ISO format, or today if not specified',
		metavar='YYYY-MM-DD',
		default=None,
	)
	argparser.add_argument(
		'--minimum-events-to-count',
		type=int,
		help='Players must have attended this amount of tournaments (that are counted for the ranking, see --event-size-to-count) in this season to be eligible for the rankings. 3 by default',
		default=3,
	)
	argparser.add_argument(
		'--event-size-to-count',
		type=int,
		help="Events must have this amount of entrants to count for the ranking (to reduce the effect of smaller tournaments where a player's performance is mostly dependent on the distribution of other players attending). 16 by default",
		default=16,
	)
	argparser.add_argument(
		'--excluded-series',
		nargs='*',
		help='Tournaments in this series (as defined by Ausmash) do not count towards the rankings',
	)
	argparser.add_argument(
		'--drop-zero-score',
		action=BooleanOptionalAction,
		help="Don't show players with a total of zero score across all tournaments, to avoid any perception that 0-2er friends are being picked on :) (and also because those stats would be less meaningful or useful) but by default all players are listed",
		default=False,
	)
	argparser.add_argument(
		'--output-dates',
		action=BooleanOptionalAction,
		help='Output the dates that tournaments happened on, instead of the tournament name, so it can be used as a time series. If two tournaments happen on the same day, might break, uh oh',
		default=False,
	)
	argparser.add_argument(
		'--redemption',
		action='store_true',
		help='Output stats for redemption brackets instead of main bracket',
	)
	argparser.add_argument(
		'--confidence-percent',
		type=float,
		help='Calculate a confidence interval for the mean score at this confidence level, default is 0.95, specify 0 to not do this',
		default=0.95,
	)
	argparser.add_argument(
		'--sort-column',
		help='Sort by and add ranks and tiers for this column if specied, "low" to use the lower bound of the confidence interval, or "high" for the high bound, otherwise mean score',
		default=None,
	)

	argparser.add_argument(
		'out_path',
		type=Path,
		help='Path to save csv files to, if not set, will output to stdout, and only the scores for all relevant tournaments',
		nargs='?',
		default=None,
	)

	if 'debugpy' in sys.modules:
		# Evil hack, setting default args for inside VS Code
		argparser.print_help()
		args = argparser.parse_args(
			[
				'--region',
				'ACT',
				'--season-start',
				'2023-07-01',
				'--season-end',
				'2023-10-03',
				'--excluded-series',
				'Epic Games Night',
				'--drop-zero-score',
				'--minimum-events-to-count',
				'5',
				'--sort-column',
				'low',
				'/media/Shared/Datasets/Smash',
			]
		)
		logger.info(args)
	else:
		args = argparser.parse_args()

	out_path: Path | None = args.out_path
	game: Game = args.game
	region: Region | None = args.region
	season_start: date | None = args.season_start
	season_end: date | None = args.season_end
	minimum_events_to_count: int = args.minimum_events_to_count
	event_size_to_count: int = args.event_size_to_count
	excluded_series: list[str] = args.excluded_series or []
	drop_zero_score: bool = args.drop_zero_score
	output_dates: bool = args.output_dates
	redemption: bool = args.redemption
	confidence_percent: float = args.confidence_percent
	sort_column: str | None = args.sort_column

	# TODO: Arguments should have a way to override active_players

	active_players = sorted(
		get_active_players(
			game,
			region,
			season_start,
			season_end,
			minimum_events_to_count,
			only_count_locals=False,
			only_count_main_bracket=False,
		),
		key=lambda p: p.name,
	)
	if not active_players:
		raise ValueError('Nobody is active for this game in this region for this season')
	logger.info('%d active players in this region for this season', len(active_players))
	logger.info([str(p) for p in active_players])

	output_stats(
		out_path,
		active_players,
		game,
		region,
		season_start,
		season_end,
		event_size_to_count,
		excluded_series,
		minimum_events_to_count,
		output_dates=output_dates,
		drop_zero_score=drop_zero_score,
		redemption=redemption,
		confidence_percent=confidence_percent,
		sort_column=sort_column,
	)


if __name__ == '__main__':
	main()
