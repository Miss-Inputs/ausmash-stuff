#!/usr/bin/env python3

from itertools import starmap

import numpy
import pandas

import ausmash
from tier_list import CharacterTierList, TieredItem


def main() -> None:
	# Exclude national rankings because they're borked right now, 500 error when attempting to access any ranks
	rankings = [
		r for r in ausmash.Ranking.all() if r.game.short_name == 'SSBU' and r.region
	]

	char_ranks: dict[ausmash.Character, list[float]] = {}
	act_char_ranks: dict[ausmash.Character, list[float]] = {}

	for ranking in rankings:
		size = len(ranking.ranks) + 1
		chars_in_ranking = set()
		for rank in ranking.ranks:
			for character in rank.characters:
				character = ausmash.combine_echo_fighters(character)
				score = 1 - (rank.rank / size)
				char_ranks.setdefault(character, []).append(score)
				if ranking.region and ranking.region.short_name == 'ACT':
					act_char_ranks.setdefault(character, []).append(score)
				chars_in_ranking.add(character)
		for character in ausmash.Character.characters_in_game(ranking.game):
			character = ausmash.combine_echo_fighters(character)
			if character not in chars_in_ranking:
				char_ranks.setdefault(character, []).append(0)
				if ranking.region and ranking.region.short_name == 'ACT':
					act_char_ranks.setdefault(character, []).append(0)

	rows = [
		[
			char,
			len([rank for rank in ranks if rank != 0]),
			numpy.sum(ranks),
			numpy.mean(ranks),
			numpy.max(ranks),
		]
		for char, ranks in char_ranks.items()
	]
	df = pandas.DataFrame(rows, columns=['char', 'count', 'sum', 'mean', 'max'])
	df.set_index('char', inplace=True)

	tl = CharacterTierList(
		[TieredItem(row.Index, row.mean) for row in df.itertuples()],
		append_minmax_to_tier_titles=True,
		score_formatter='.4g',
		scale_factor=3,
	)
	tl.to_image('Spectral').save(
		'/media/Shared/Datasets/Smash/Character showcase positions.png'
	)

	tl = CharacterTierList(
		[
			TieredItem(char, numpy.mean(scores))
			for char, scores in act_char_ranks.items()
			if any(scores)
		],
		append_minmax_to_tier_titles=True,
		score_formatter='.4g',
		scale_factor=3,
	)
	tl.to_image('Spectral').save(
		'/media/Shared/Datasets/Smash/Character showcase positions ACT.png'
	)

	tl = CharacterTierList(
		starmap(
			TieredItem,
			zip(
				sorted(df.index, key=lambda char: char.name),
				numpy.linspace(1, 0, df.index.size),
				strict=True,
			),
		),
		append_minmax_to_tier_titles=False,
		score_formatter='%',
	)
	tl.to_image('Spectral').save(
		'/media/Shared/Datasets/Smash/Characters tiered alphabetically.png'
	)

	tl = CharacterTierList(
		[TieredItem(char, len(char.name)) for char in df.index],
		append_minmax_to_tier_titles=True,
		score_formatter=',',
	)
	tl.to_image('Spectral').save(
		'/media/Shared/Datasets/Smash/Characters tiered by name length.png'
	)


if __name__ == '__main__':
	main()
