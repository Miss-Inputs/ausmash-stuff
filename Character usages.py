#!/usr/bin/env python3

import itertools
import logging
import pickle
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import cast

from ausmash import BracketStyle, Character, Match, Region, combine_echo_fighters
from matplotlib import pyplot
from pyzstd import ZstdFile
from tqdm import tqdm

from tier_list import CharacterTierList, TieredItem

logger = logging.getLogger(__file__)

ssbu_chars = {combine_echo_fighters(char) for char in Character.characters_in_game('SSBU')}


def character_usages(
	matches: Sequence[Match], tqdm_desc='Character usages'
) -> tuple[float, list[TieredItem[Character]]]:
	usages: defaultdict[Character, int] = defaultdict(int)
	matches_with_character_data = 0
	for match in tqdm(matches, tqdm_desc):
		chars_used = frozenset(
			combine_echo_fighters(char) for char in match.winner_characters
		) | frozenset(combine_echo_fighters(char) for char in match.loser_characters)
		if not chars_used:
			continue
		matches_with_character_data += 1
		for char in chars_used:
			usages[char] += 1

	if matches[0].game.short_name == 'SSBU':
		for char in ssbu_chars:
			usages[char]  # pylint: disable=pointless-statement #nuh uhhh nerd it's literally a defaultdict

	return matches_with_character_data / len(matches), [
		TieredItem(char, usage / len(matches)) for char, usage in usages.items()
	]


def months_between(d1: date, d2: date) -> int:
	return ((d1.year - d2.year) * 12) + (d1.month - d2.month)


def _region_name(match: Match) -> str:
	return match.tournament.region.short_name


def _region(match: Match) -> Region:
	return match.tournament.region


colourmap = pyplot.get_cmap('Spectral')


def output_tier_list(
	region: Region | None, matches: Sequence[Match], image_path: Path, *additional_title_lines: str
):
	ratio, usages = character_usages(matches, f'Character usage for {image_path.stem}')
	match_dates = [m.date for m in matches]
	earliest, latest = min(match_dates), max(match_dates)
	title_lines = [region.name if region else 'Australia + New Zealand']
	if additional_title_lines:
		title_lines += additional_title_lines
	title_lines += [
		f'From {earliest} to {latest}',
		f'{ratio:%} of {len(matches)} matches with character data',
	]
	if len([usage for usage in usages if usage.score > 0]) < 3:
		logger.info('nope')
		return
	tier_list = CharacterTierList(
		usages, title='\n'.join(title_lines), score_formatter='%', scale_factor=3
	)
	if tier_list.num_tiers == 2:
		tier_list.tier_names = dict(enumerate(('Used', 'Less used')))
	elif tier_list.num_tiers == 3:
		tier_list.tier_names = dict(enumerate('SAB'))
	tier_list.to_image(
		colourmap, show_scores=True, title_background=region.colour_string if region else None
	).save(image_path)
	image_path.with_suffix('.txt').write_text(tier_list.to_text(show_scores=True), encoding='utf8')


def _is_in_grands(match: Match) -> bool:
	if match.event not in match.tournament.events:
		logger.error('what the hellll %s %s %s', match, match.event, match.tournament)
		return False
	return (
		match.round_name in {'GF', 'GF2'}
		and not match.event.is_redemption_bracket
		and match.tournament.final_phase_for_event(match.event) == match.event
	)


def _is_in_wr1_or_2(match: Match) -> bool:
	# TODO: What I really want is "is in starting round" but I'd have to look at start.gg to see who starts in WR1 and who starts in WR2 (or calculate that from seeding which needs looking at start.gg anyway)
	if match.event not in match.tournament.events:
		logger.error('what the hellll %s %s %s', match, match.event, match.tournament)
		return False
	return (
		(match.round_name in {'W1', 'W2'} or match.event.bracket_style == BracketStyle.RoundRobin)
		and not match.event.is_redemption_bracket
		and match.tournament.start_phase_for_event(match.event) == match.event
	)


def main() -> None:
	base_dir = Path('/media/Shared/Datasets/Smash/')
	out_dir = base_dir / 'Tier lists'
	with ZstdFile(max(base_dir.glob('Matches/All matches (*).pickle.zst')).open('rb')) as z:
		matches = cast(Sequence[Match], pickle.load(z))

	ult_matches = sorted(
		(match for match in matches if match.game.short_name == 'SSBU'), key=_region_name
	)
	output_tier_list(None, ult_matches, out_dir / 'Characters by usage.png')
	output_tier_list(
		None,
		[m for m in ult_matches if _is_in_wr1_or_2(m)],
		out_dir / 'Characters by usage in WR1 or WR2.png',
		'Winners round 1 or 2',
	)
	output_tier_list(
		None,
		[m for m in ult_matches if _is_in_grands(m)],
		out_dir / 'Characters by usage in grands.png',
		'Grand finals',
	)
	output_tier_list(
		None,
		[match for match in matches if match.game.short_name == 'SSBM'],
		out_dir / 'SSBM characters by usage.png',
	)
	recent_matches = [m for m in ult_matches if months_between(date.today(), m.date) <= 3]
	output_tier_list(None, recent_matches, out_dir / 'Characters by usage in last 3 months.png')
	output_tier_list(
		None,
		[m for m in recent_matches if _is_in_wr1_or_2(m)],
		out_dir / 'Characters by usage in WR1 or WR2 in last 3 months.png',
		'Winners round 1 or 2',
	)
	output_tier_list(
		None,
		[m for m in recent_matches if _is_in_grands(m)],
		out_dir / 'Characters by usage in grands in last 3 months.png',
		'Grand finals',
	)

	for region, group in itertools.groupby(ult_matches, _region):
		logger.info(region.short_name)
		matches = list(group)
		output_tier_list(
			region, matches, out_dir / f'Characters by usage in {region.short_name}.png'
		)
		output_tier_list(
			region,
			[m for m in matches if _is_in_wr1_or_2(m)],
			out_dir / f'Characters by usage in WR1 or WR2 in {region.short_name}.png',
			'Winners round 1 or 2',
		)
		output_tier_list(
			region,
			[m for m in matches if _is_in_grands(m)],
			out_dir / f'Characters by usage in grands in {region.short_name}.png',
			'Grand finals',
		)

		recent_matches = [m for m in matches if months_between(date.today(), m.date) <= 3]
		if len(recent_matches) <= 3:
			logger.info(
				"wank dicks balls piss %s has only %d active matches, that won't do",
				region.name,
				len(recent_matches),
			)
			continue
		output_tier_list(
			region,
			recent_matches,
			out_dir / f'Characters by usage in {region.short_name} in last 3 months.png',
		)
		output_tier_list(
			region,
			[m for m in recent_matches if _is_in_wr1_or_2(m)],
			out_dir
			/ f'Characters by usage in WR1 or WR2 in {region.short_name} in last 3 months.png',
			'Winners round 1 or 2',
		)
		output_tier_list(
			region,
			[m for m in recent_matches if _is_in_grands(m)],
			out_dir / f'Characters by usage in grands in {region.short_name} in last 3 months.png',
			'Grand finals',
		)


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	main()
