#!/usr/bin/env python3

import numpy
import pandas

import ausmash
from tier_list import CharacterTierList, TieredItem


def main() -> None:
	#Exclude national rankings because they're borked right now, 500 error when attempting to access any ranks
	rankings = [r for r in ausmash.Ranking.all() if r.game.short_name == 'SSBU' and r.region]
	
	char_ranks: dict[ausmash.Character, list[float]] = {}

	for ranking in rankings:
		size = len(ranking.ranks) + 1
		chars_in_ranking = set()
		for rank in ranking.ranks:
			for character in rank.characters:
				character = ausmash.combine_echo_fighters(character)
				char_ranks.setdefault(character, []).append(1 - (rank.rank / size))
				chars_in_ranking.add(character)
		for character in ausmash.Character.characters_in_game(ranking.game):
			character = ausmash.combine_echo_fighters(character)
			if character not in chars_in_ranking:
				char_ranks.setdefault(character, []).append(0)
				
	rows = [[char, len([rank for rank in ranks if rank != 0]), numpy.sum(ranks), numpy.mean(ranks), numpy.max(ranks)] for char, ranks in char_ranks.items()]
	df = pandas.DataFrame(rows, columns=['char', 'count', 'sum', 'mean', 'max'])
	df.set_index('char', inplace=True)

	tl = CharacterTierList([TieredItem(row.Index, row.mean) for row in df.itertuples()])
	tl.to_image('Spectral').save('/media/Shared/Datasets/Smash/Character showcase positions.png')

	tl = CharacterTierList([TieredItem(char, score) for char, score in zip(sorted(df.index, key=lambda char: char.name), numpy.linspace(1, 0, df.index.size))])
	tl.to_image('Spectral').save('/media/Shared/Datasets/Smash/Characters tiered alphabetically.png')

	tl = CharacterTierList([TieredItem(char, len(char.name)) for char in df.index])
	tl.to_image('Spectral').save('/media/Shared/Datasets/Smash/Characters tiered by name length.png')
	

if __name__ == '__main__':
	main()