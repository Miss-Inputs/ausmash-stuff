#!/usr/bin/env python3

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeVar

import numpy
import pandas
import requests
from matplotlib import pyplot  # just for colour maps lol
from PIL import Image, ImageFilter, ImageFont
from PIL.ImageDraw import ImageDraw
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import minmax_scale

from ausmash import Character, Elo, combine_echo_fighters
from ausmash.models.character import CombinedCharacter

if TYPE_CHECKING:
	from pandas.core.groupby.generic import DataFrameGroupBy


class Tiers(NamedTuple):
	tiers: 'pandas.Series[int]'  # Index = the same one that was passed in for scores
	centroids: Mapping[int, float]  # For each tier number
	inertia: float
	kmeans_iterations: int


def find_best_clusters(scores: 'pandas.Series[float]') -> Tiers:
	best_score = -numpy.inf
	best = None
	for i in range(2, scores.nunique()):
		tiers = _get_clusters(scores, i)
		score = -tiers.inertia
		if best_score >= score:
			continue
		if tiers.kmeans_iterations == 300:
			continue
		if tiers.tiers.nunique() < len(tiers.centroids):
			continue
		best = tiers
		best_score = score
	if not best:
		raise RuntimeError('oh no')
	return best


def _get_clusters(
	scores: 'pandas.Series[float]', n_clusters: int | Literal['auto']
) -> Tiers:
	# Returns tiers for each row, centroids
	if n_clusters == 'auto':
		return find_best_clusters(scores)
	kmeans = KMeans(n_clusters, n_init='auto', random_state=0)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', ConvergenceWarning)
		labels = kmeans.fit_predict(scores.to_numpy().reshape(-1, 1))
	raw_tiers = pandas.Series(labels, index=scores.index, name='tiers')
	# The numbers in raw_tiers are just random values for the sake of being distinct, we are already sorted by score, so have ascending tiers instead
	mapping = {c: i for i, c in enumerate(raw_tiers.unique())}
	tiers = raw_tiers.map(mapping)
	centroids = (
		pandas.Series(kmeans.cluster_centers_.squeeze(), name='centroids')
		.rename(mapping)
		.sort_index()
		.to_dict()
	)
	return Tiers(tiers, centroids, kmeans.inertia_, kmeans.n_iter_)


def generate_background(width: int, height: int) -> Image.Image:
	"""Generate some nice pretty rainbow clouds"""
	rng = numpy.random.default_rng()
	noise = rng.integers(0, (128, 128, 128), (height, width, 3), 'uint8', True)
	# Could also have a fourth dim with max=255 and then layer multiple transparent clouds on top of each other
	image = (
		Image.fromarray(noise)
		.filter(ImageFilter.ModeFilter(100))
		.filter(ImageFilter.GaussianBlur(50))
	)
	return image


def pad_image(
	image: Image.Image, width: int, height: int, colour: Any, centred: bool = False
) -> Image.Image:
	"""Return expanded version of image with blank space to ensure a certain size.

	Don't forget to call ImageDraw again"""
	new_image = Image.new(image.mode, (width, height), colour)
	if image.palette:
		palette = image.getpalette()
		assert palette, 'image.getpalette() should not return None since we have already checked image.palette'
		new_image.putpalette(palette)
	if centred:
		x = (width - image.width) // 2
		y = (height - image.height) // 2
	else:
		x = y = 0
	new_image.paste(image, (x, y))
	return new_image


def draw_box(image: Image.Image, colour: Any = 'black', width: int = 2) -> Image.Image:
	"""Modifies image in-place and returns it with a border around the sides"""
	draw = ImageDraw(image)
	draw.rectangle((0, 0, image.width, image.height), outline=colour, width=width)
	return image


T = TypeVar('T')


@dataclass
class TieredItem(Generic[T]):
	"""An item that has a score associated with it used for ranking.

	item should not be None, and score should be a higher number for better items."""

	item: T
	score: float


class BaseTierList(Generic[T], ABC):
	def __init__(
		self,
		items: Iterable[TieredItem[T]],
		num_tiers: int | Literal['auto'] = 7,
		append_minmax_to_tier_titles: bool = False,
		score_formatter: str = '',
	) -> None:
		""":param num_tiers: Number of tiers to separate scores into. If "auto", finds the biggest number of tiers before it would stop making sense, but that often doesn't work very well and is slower to calculate, so don't bother."""
		self.data = pandas.DataFrame(list(items), columns=['item', 'score'])
		self.data.sort_values('score', ascending=False, inplace=True)
		self.data['rank'] = numpy.arange(1, self.data.index.size + 1)

		self.tiers = _get_clusters(self.data['score'], num_tiers)
		self.data['tier'] = self.tiers.tiers

		self.append_minmax_to_tier_titles = append_minmax_to_tier_titles
		self.score_formatter = score_formatter
		tier_letters = 'SABCDEFGHIJKLZ'
		# TODO: Option to provide existing tiers (would need to calculate centroids manually I guess)
		# TODO: Option to provide custom images
		self.tier_names: Mapping[int, str] = dict(enumerate(tier_letters))

	def to_text(self) -> str:
		lines: list[str] = []
		for tier_number, group in self._groupby:
			lines.extend(
				(
					'=' * 20,
					self.displayed_tier_text(tier_number, group),
					'-' * 10,
					*(f'{row.rank}: {row.item}' for row in group.itertuples()),
					'',
				)
			)
		return '\n'.join(lines)

	@cached_property
	def _groupby(self) -> 'DataFrameGroupBy[int]':
		return self.data.groupby('tier')

	@cached_property
	def _tier_texts(self) -> Mapping[int, str]:
		"""Based off tier names, not necessarily using all of them if there aren't as many tiers, falling back to a default name if needed"""
		return {
			tier_number: self.tier_names.get(tier_number, f'Tier {tier_number}')
			for tier_number in self._groupby.groups
		}

	def displayed_tier_text(
		self, tier_number: int, group: pandas.DataFrame | None = None
	):
		""":param group: If you are already iterating through ._groupby, you can pass each group so you don't have to call get_group"""
		text = self._tier_texts[tier_number]
		if self.append_minmax_to_tier_titles:
			if group is None:
				group = self._groupby.get_group(tier_number)
			min_ = group['score'].min()
			max_ = group['score'].max()
			if numpy.isclose(min_, max_):
				return f'{text} ({min_:{self.score_formatter}})'
			return f'{text} ({min_:{self.score_formatter}} to {max_:{self.score_formatter}})'
		return text

	@staticmethod
	@abstractmethod
	def get_item_image(item: T) -> Image.Image:
		...

	@cached_property
	def scaled_centroids(self) -> Mapping[int, float]:
		"""Scale centroids between 0.0 and 1.0"""
		# Don't worry, it still works on 1D arrays even if it says it wants a MatrixLike in the type hint
		# If it stops working in some future version use reshape(-1, 1)
		values = minmax_scale(list(self.tiers.centroids.values()))
		return {k: values[k] for k in self.tiers.centroids}

	@cached_property
	def images(self) -> Mapping[T, Image.Image]:
		return {item: self.get_item_image(item) for item in self.data['item']}

	def to_image(
		self, colourmap_name: str | None = None, max_images_per_row: int | None = 8
	) -> Image.Image:
		"""Render the tier list as an image.

		This doesn't look too great if the images are of uneven size, but that's allowed."""
		max_image_width = max(im.width for im in self.images.values())
		max_image_height = max(im.height for im in self.images.values())
		tier_texts = {i: self.displayed_tier_text(i) for i in self._tier_texts}

		cmap = pyplot.get_cmap(colourmap_name)

		font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None
		textbox_width = 0

		vertical_padding = 10
		horizontal_padding = 10
		# Find the largest font size we can use inside the tier name box to fit the available height
		# font_size is points and not pixels, but it'll do as a starting point
		font_size = max_image_height * 2
		for text in tier_texts.values():
			size: None | tuple[int, int, int, int] = None
			while (
				size is None or (size[3] + vertical_padding) > max_image_height
			) and font_size > 0:
				font = (
					font.font_variant(size=font_size)
					if isinstance(font, ImageFont.FreeTypeFont)
					else ImageFont.load_default(font_size)
				)
				size = font.getbbox(text)
				font_size -= 1
			assert font, 'how did my font end up being None :('
			assert size, 'meowwww'  # FIXME I think this can happen if max image size is <= the smallest size of the default font? (I think even if font_size = 0 it's not smaller than like 8 or whichever)
			length = size[2] + horizontal_padding
			if length > textbox_width:
				textbox_width = length

		height = self._groupby.ngroups * max_image_height
		if not max_images_per_row:
			max_images_per_row = self._groupby.size().max()
		width = (
			min(self._groupby.size().max(), max_images_per_row) * max_image_width
			+ textbox_width
		)

		trans = (0, 0, 0, 0)
		image = Image.new('RGBA', (width, height), trans)
		draw = ImageDraw(image)

		actual_width = width
		next_line_y = 0
		for tier_number, group in self._groupby:
			tier_text = tier_texts[tier_number]

			row_height = max_image_height * (
				((group.index.size - 1) // max_images_per_row) + 1
			)

			box_end = next_line_y + row_height
			if box_end > image.height:
				image = pad_image(image, image.width, box_end, trans)
				draw = ImageDraw(image)
			if textbox_width > image.width:  # This probably doesn't happen
				image = pad_image(image, textbox_width, image.height, trans)
				draw = ImageDraw(image)

			colour = cmap(self.scaled_centroids[tier_number])
			colour_as_int = tuple(int(v * 255) for v in colour)
			# Am I stupid, or is there actually nothing in standard library or matplotlib that does this
			# Well colorsys.rgb_to_yiv would also potentially work
			luminance = (
				(colour[0] * 0.2126) + (colour[1] * 0.7152) + (colour[2] * 0.0722)
			)
			text_colour = 'white' if luminance <= 0.5 else 'black'
			draw.rectangle((0, next_line_y, 0, box_end - 1), fill='black')
			draw.rectangle(
				(textbox_width, next_line_y, textbox_width, box_end - 1), fill='black'
			)
			draw.rectangle(
				(1, next_line_y, textbox_width - 1, box_end - 1),
				fill=colour_as_int,
				outline='black',
				width=1,
			)
			draw.text(
				(textbox_width / 2, (next_line_y + box_end) / 2),
				tier_text,
				anchor='mm',
				fill=text_colour,
				font=font,
			)

			next_image_x = textbox_width + 1  # Account for border
			for i, item in enumerate(group['item']):
				item_image = self.images[item]
				image_row, image_col = divmod(i, max_images_per_row)
				if not image_col:
					next_image_x = textbox_width + 1

				image_y = next_line_y + (max_image_height * image_row)
				if next_image_x + item_image.width > image.width:
					image = pad_image(
						image, next_image_x + item_image.width, image.height, trans
					)
					draw = ImageDraw(image)
					# TODO: Optionally draw the score or name below each character (maybe that's better off with an overriten get_item_image, or maybe get_item_image_with_score)
				image.paste(item_image, (next_image_x, image_y))
				next_image_x += item_image.width
				actual_width = max(actual_width, next_image_x)
			next_line_y = box_end

		# Uneven images can result in calculating too much space to the side
		image = image.crop((0, 0, actual_width, next_line_y))

		background = generate_background(image.width, image.height)
		background.paste(image, mask=image)
		return background


class TierList(BaseTierList[T]):
	"""Default implementation of TierList that just displays text as images"""

	# Default size of load_default 10, so that sucks and we won't do that
	_default_font = ImageFont.load_default(20)
	_reg = re.compile(r'\s+?(?=\b\w{5,})')  # Space followed by at least 5-letter word

	@staticmethod
	def get_item_image(item: T) -> Image.Image:
		text = getattr(item, 'name', str(item))
		if ' ' in text:
			text = TierList._reg.sub('\n', text)
			width, height = ImageDraw(Image.new('1', (1, 1))).multiline_textbbox(
				(0, 0), text, font=TierList._default_font
			)[2:]
		else:
			width, height = TierList._default_font.getbbox(text)[2:]
		image = Image.new('RGBA', (width + 1, height + 1))
		draw = ImageDraw(image)
		draw.multiline_text((0, 0), text, font=TierList._default_font, align='center')
		return image


class TextBoxTierList(TierList[T]):
	"""Pads out the images from the default implementation of get_item_image, so they are all evenly spaced.

	Hopefully looks a bit less bad."""

	@cached_property
	def images(self) -> Mapping[T, Image.Image]:
		unscaled = super().images
		max_height = max(im.height for im in unscaled.values())
		max_width = max(im.width for im in unscaled.values())
		return {
			item: draw_box(
				pad_image(image, max_width + 2, max_height + 2, (0, 0, 0, 0), True)
			)
			for item, image in unscaled.items()
		}


class CharacterTierList(BaseTierList[Character]):
	@staticmethod
	def get_item_image(item: Character) -> Image.Image:
		if isinstance(item, CombinedCharacter):
			return CharacterTierList.get_combined_char_image(item)
		url = item.character_select_screen_pic_url
		response = requests.get(url, stream=True, timeout=10)
		response.raise_for_status()
		image = Image.open(response.raw)

		return image

	@staticmethod
	def get_combined_char_image(character: CombinedCharacter) -> Image.Image:
		if len(character.chars) == 2:
			# Divvy it up into two diagonally, I dunno how to make it look the least shite
			first, second = character.chars
			first_image = CharacterTierList.get_item_image(first)
			second_image = CharacterTierList.get_item_image(second)
			if first_image.size != second_image.size:
				second_image = second_image.resize(first_image.size)
			# numpy.triu/tril won't work nicely on non-square rectangles
			orig_size = first_image.size
			max_dim = max(orig_size)
			square_size = max_dim, max_dim
			a = numpy.array(first_image.resize(square_size))
			b = numpy.array(second_image.resize(square_size))
			upper_right = numpy.triu(a.swapaxes(0, 2)).swapaxes(0, 2)
			lower_left = numpy.tril(b.swapaxes(0, 2)).swapaxes(0, 2)
			return Image.fromarray(upper_right + lower_left).resize(orig_size)

		# Just merge them together if we have a combined character with 3 or more
		images = [
			numpy.array(CharacterTierList.get_item_image(char))
			for char in character.chars
		]
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
	tierlist = CharacterTierList(scores, 7, append_minmax_to_tier_titles=True)
	print(
		tierlist.tiers.inertia,
		tierlist.tiers.kmeans_iterations,
		tierlist.tiers.tiers.nunique(),
		len(tierlist.tiers.centroids),
	)
	print(tierlist.tiers.centroids)
	print(tierlist.to_text())

	players = [p for p in Elo.for_game('SSBU', 'ACT') if p.is_active]
	listy = TextBoxTierList(
		[TieredItem(player.player, player.elo) for player in players],
		append_minmax_to_tier_titles=True,
		score_formatter=',',
	)
	listy.to_image('rainbow_r').show()


if __name__ == '__main__':
	main()
