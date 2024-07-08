#!/usr/bin/env python3

from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from io import BytesIO
from typing import Literal

import numpy
import requests
from ausmash import Character, Elo, combine_echo_fighters
from ausmash.classes.character import CombinedCharacter
from PIL import Image, ImageOps
from tier_lister import AutoTierer, BaseTierList, TextBoxTierList, TieredItem, image_utils
from tier_lister.kmeans_tierer import kmeans_tierer
from tier_lister.tiers import quantile_tierer

try:
	from requests_cache import CachedSession

	have_requests_cache = True
except ImportError:
	have_requests_cache = False


@cache
def get_character_image(char: Character) -> Image.Image:
	"""Yoink character select screen image for a character as a Pillow image."""
	url = char.character_select_screen_pic_url
	with (
		CachedSession('character_images', use_cache_dir=True)
		if have_requests_cache
		else requests.session() as session
	):
		response = session.get(
			url, timeout=10
		)  # stream=True doesn't work with CachedSession I think
		response.raise_for_status()
		return Image.open(BytesIO(response.content))


class CharacterTierList(BaseTierList[Character]):
	"""Tier list specifically for characters, which would be one of the most likely use cases"""

	def __init__(
		self,
		items: Iterable[TieredItem[Character]],
		tiers: 'Sequence[int] | AutoTierer' = kmeans_tierer,
		num_tiers: int | Literal['auto'] = 'auto',
		tier_names: Sequence[str] | Mapping[int, str] | None = None,
		title: str | None = None,
		*,
		append_minmax_to_tier_titles: bool = False,
		score_formatter: str = '',
		scale_factor: float | None = None,
		resampling: Image.Resampling = Image.Resampling.LANCZOS,
	) -> None:
		self.scale_factor = scale_factor
		self.resampling = resampling
		super().__init__(
			items,
			tiers,
			num_tiers,
			tier_names,
			title,
			append_minmax_to_tier_titles=append_minmax_to_tier_titles,
			score_formatter=score_formatter,
		)

	def get_item_image(self, item: Character) -> Image.Image:
		if isinstance(item, CombinedCharacter):
			return self.get_combined_char_image(item)
		image = get_character_image(item)
		if self.scale_factor:
			image = ImageOps.scale(image, self.scale_factor, self.resampling)
		return image

	def get_combined_char_image(self, character: CombinedCharacter) -> Image.Image:
		if len(character.chars) == 2:
			# Divvy it up into two diagonally, I dunno how to make it look the least shite
			first, second = character.chars
			first_image = self.get_item_image(first)
			second_image = self.get_item_image(second)
			return image_utils.combine_images_diagonally(first_image, second_image)

		# Just merge them together if we have a combined character with 3 or more
		images = [numpy.array(self.get_item_image(char)) for char in character.chars]
		return Image.fromarray(numpy.mean(images, axis=(0)).astype('uint8'))


def main() -> None:
	# TODO: Proper command line interfacey
	print('testing for now')
	miis = set()
	chars: set[Character] = set()
	for char in Character.characters_in_game('SSBU'):
		if char.name.startswith('Mii'):
			miis.add(char)
		else:
			chars.add(combine_echo_fighters(char))
	chars.add(CombinedCharacter('Mii Fighters', miis))
	scores = [TieredItem(char, len(char.name)) for char in chars]
	tierlist = CharacterTierList(
		scores,
		title='Test tier list\nCharacter name lengths',
		append_minmax_to_tier_titles=True,
		# tiers=quantile_tierer,
	)
	# print(
	# 	tierlist.tiers.inertia,
	# 	tierlist.tiers.kmeans_iterations,
	# 	tierlist.tiers.tier_ids.nunique(),
	# )
	print(tierlist.tiers.centroids)
	print(tierlist.to_text())
	tierlist.to_image('rainbow_r', show_scores=True).show()

	players = [p for p in Elo.for_game('SSBU', 'ACT') if p.is_active]
	listy = TextBoxTierList(
		[TieredItem(player.player, player.elo) for player in players],
		title='Active ACT players by Elo',
		append_minmax_to_tier_titles=True,
		score_formatter=',',
		# tiers=quantile_tierer,
	)
	listy.to_image('rainbow_r', show_scores=True).show()

	test_list = TextBoxTierList(
		[
			TieredItem('🐈', 10),
			TieredItem('Good Thing', 7.5),
			TieredItem('okay', 6),
			TieredItem('meh', 4),
			TieredItem('Windows Vista', -1),
		],
		[0, 0, 1, 1, 2],
		tier_names=['pretty cool', 'alright', 'bleh'],
		append_minmax_to_tier_titles=True,
	)
	print(test_list.to_text())
	test_list.to_image(show_scores=True).show()


if __name__ == '__main__':
	main()
