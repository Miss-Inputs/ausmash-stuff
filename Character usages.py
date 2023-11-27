#!/usr/bin/env python3

import itertools
import logging
import pickle
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import cast

from ausmash import Character, Match, combine_echo_fighters
from matplotlib import pyplot
from pyzstd import ZstdFile
from tqdm import tqdm

from tier_list import CharacterTierList, TieredItem

logger = logging.getLogger(__file__)

ssbu_chars = {combine_echo_fighters(char) for char in Character.characters_in_game('SSBU')}


def character_usages(matches: Sequence[Match]):
	usages: defaultdict[Character, int] = defaultdict(int)
	matches_with_character_data = 0
	for match in tqdm(matches):
		chars_used = frozenset(
			combine_echo_fighters(char) for char in match.winner_characters
		) | frozenset(combine_echo_fighters(char) for char in match.loser_characters)
		if not chars_used:
			continue
		matches_with_character_data += 1
		for char in chars_used:
			usages[char] += 1

	logger.info(
		'Ratio of matches with character data: %g %g',
		matches_with_character_data / len(matches),
		len(matches) / matches_with_character_data,
	)

	for char in ssbu_chars:
		usages.__getitem__(char)

	return (TieredItem(char, usage / len(matches)) for char, usage in usages.items())


def months_between(d1: date, d2: date) -> int:
	return ((d1.year - d2.year) * 12) + (d1.month - d2.month)


def _region(match: Match) -> str:
	return match.tournament.region.short_name


colourmap = pyplot.get_cmap('Spectral')


def output_tier_list(matches: Sequence[Match], image_path: Path):
	usages = character_usages(matches)
	tier_list = CharacterTierList(
		usages,
		score_formatter='%',
		scale_factor=3,
	)
	if tier_list.num_tiers == 3:
		tier_list.tier_names = dict(enumerate('SAB'))
	tier_list.to_image(colourmap, show_scores=True).save(image_path)
	image_path.with_suffix('.txt').write_text(tier_list.to_text(show_scores=True), encoding='utf8')


def main() -> None:
	base_dir = Path('/media/Shared/Datasets/Smash/')
	with ZstdFile(max(base_dir.glob('All matches (*).pickle.zst')).open('rb')) as z:
		matches = cast(Sequence[Match], pickle.load(z))

	ult_matches = sorted(
		(match for match in matches if match.game.short_name == 'SSBU'), key=_region
	)
	output_tier_list(ult_matches, base_dir / 'Characters by usage.png')
	output_tier_list(
		[match for match in matches if match.game.short_name == 'SSBM'],
		base_dir / 'SSBM characters by usage.png',
	)
	output_tier_list(
		[m for m in ult_matches if months_between(date.today(), m.date) <= 3],
		base_dir / 'Characters by usage in last 3 months.png',
	)

	for region, group in itertools.groupby(ult_matches, _region):
		logger.info(region)
		matches = list(group)
		output_tier_list(matches, base_dir / f'Characters by usage in {region}.png')

		recent_matches = [m for m in matches if months_between(date.today(), m.date) <= 3]
		if not recent_matches:
			logger.info('wank dicks balls piss')
			continue
		output_tier_list(
			recent_matches, base_dir / f'Characters by usage in {region} in last 3 months.png'
		)


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	main()
