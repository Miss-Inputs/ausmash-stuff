#!/usr/bin/env python3

import ausmash
import numpy
import pandas
from matplotlib import pyplot

from tier_list import CharacterTierList, TieredItem


def main() -> None:
	# Exclude national rankings because they're borked right now, 500 error when attempting to access any ranks
	rankings = [r for r in ausmash.Ranking.all() if r.game.short_name == 'SSBU' and r.region]

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

	spectral = pyplot.get_cmap('Spectral')

	tl = CharacterTierList.from_items(
		df['mean'], title='Australia + NZ', score_formatter='.4g', scale_factor=3
	)
	tl.to_image(spectral, show_scores=True, title_background='white').save(
		'/home/megan/Pictures/Tier lists/SSBU tier lists/Character showcase positions.png'
	)
	print(tl.to_text())

	tl = CharacterTierList(
		[
			TieredItem(char, numpy.mean(scores))
			for char, scores in act_char_ranks.items()
			if any(scores)
		],
		title='ACT',
		score_formatter='.4g',
		scale_factor=3,
	)
	tl.to_image(spectral, show_scores=True, title_background='white').save(
		'/home/megan/Pictures/Tier lists/SSBU tier lists/Character showcase positions ACT.png'
	)
	print(tl.to_text())


if __name__ == '__main__':
	main()
