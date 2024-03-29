#!/usr/bin/env python3

import itertools
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cache, cached_property
from io import BytesIO
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, SupportsFloat, TypeVar

import numpy
import pandas
import requests
from ausmash import Character, Elo, combine_echo_fighters
from ausmash.classes.character import CombinedCharacter
from matplotlib import pyplot  # just for colour maps lol
from PIL import Image, ImageColor, ImageFilter, ImageFont, ImageOps
from PIL.ImageDraw import ImageDraw

try:
	from requests_cache import CachedSession

	have_requests_cache = True
except ImportError:
	have_requests_cache = False
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import minmax_scale

if TYPE_CHECKING:
	from matplotlib.colors import Colormap
	from pandas.core.groupby.generic import DataFrameGroupBy
	from typing_extensions import Self

Floatable = SupportsFloat | pandas.Timestamp | timedelta


class Tiers(NamedTuple):
	tiers: 'pandas.Series[int]'  # Index = the same one that was passed in for scores
	centroids: Mapping[int, float]  # For each tier number
	inertia: float
	kmeans_iterations: int


def _cluster_loss(tiers: Tiers, desired_size: float | None) -> float:
	"""Loss function that gives a number closer to 0 for more balanced clusters (containing similar number of elements), kinda"""
	sizes = tiers.tiers.value_counts()
	if not desired_size:
		desired_size = sizes.mean()
	diffs = sizes - desired_size
	diffs[diffs < 0] = (
		diffs[diffs < 0] * 2
	)  # This should penalize harder results that end up with small clusters such as 1 item (I think) so you don't end up with 9999 tiers (I think)
	return float(
		(diffs**2).sum()
	)  # Technically return value is numpy.float64 or whatever (but type hinted as Any) and the mypy warning was bugging me


def find_best_clusters(scores: 'pandas.Series[float]') -> Tiers:
	"""Tries to find a value for n_clusters that gives cluster sizes close to sqrt(number of tier items)
	:raises RuntimeError: if it somehow doesn't find anything"""
	best_loss = numpy.inf
	best = None
	# KMeans is invalid with <2 clusters, and wouldn't really make sense with more than the number of unique values
	for i in range(2, scores.nunique()):
		tiers = _get_clusters(scores, i)
		loss = _cluster_loss(tiers, numpy.sqrt(scores.size))
		if loss > best_loss:
			continue
		if tiers.kmeans_iterations == 300:
			# KMeans didn't like this and cooked too hard, give up
			continue
		if tiers.tiers.nunique() < len(tiers.centroids):
			# KMeans gave us weird results, give up
			continue
		best = tiers
		best_loss = loss
	if not best:
		raise RuntimeError('oh no')
	return best


def _get_clusters(scores: 'pandas.Series[float]', n_clusters: int | Literal['auto']) -> Tiers:
	"""Separate scores into tiers with k-means clustering. Ensures tier numbers are monotonically increasing."""
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


def generate_background(width: int, height: int, radius: int = 50) -> Image.Image:
	"""Generate some nice pretty rainbow clouds"""
	rng = numpy.random.default_rng()
	noise = rng.integers(0, (128, 128, 128), (height, width, 3), 'uint8', endpoint=True)
	# Could also have a fourth dim with max=255 and then layer multiple transparent clouds on top of each other
	return (
		Image.fromarray(noise)
		.filter(ImageFilter.ModeFilter(radius * 2))
		.filter(ImageFilter.GaussianBlur(radius))
	)


def pad_image(
	image: Image.Image, width: int, height: int, colour: Any, *, centred: bool = False
) -> Image.Image:
	"""Return expanded version of image with blank space to ensure a certain size.

	Don't forget to call ImageDraw again"""
	new_image = Image.new(image.mode, (width, height), colour)
	if image.palette:
		palette = image.getpalette()
		assert (
			palette
		), 'image.getpalette() should not return None since we have already checked image.palette'
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


def fit_font(
	font: ImageFont.FreeTypeFont | None,
	width: int | None,
	height: int | None,
	text: str | Iterable[str],
	vertical_padding: int = 10,
	horizontal_padding: int = 10,
) -> tuple[ImageFont.FreeTypeFont, int, int]:
	"""Find the largest font that will fit into a box of a given size, or just height or width to also find how big the other dimension of the box needs to be

	Returns the font, height, or width

	:param font: Font to use, or load the default font if None
	:param width: Max width in pixels or None
	:param height: Max height in pixels or None
	:param text: Text to measure, or iterable of texts to find what fits all of them
	:param vertical_padding: Amount of vertical space in pixels to add to the box
	:param horizontal_padding: Amount of horizontal space in pixels to add to the box
	:raises RuntimeError: if font is None and default font can't be resized
	:raises ValueError: if it can't find a font at all somehow
	"""
	# font_size is points and not pixels, but it'll do as a starting point
	font_size = font.size if font else (100 if height is None else height * 2)

	if isinstance(text, str):
		while True:
			if isinstance(font, ImageFont.FreeTypeFont):
				font = font.font_variant(size=font_size)
			else:
				default_font = ImageFont.load_default(font_size)
				if not isinstance(default_font, ImageFont.FreeTypeFont):
					raise RuntimeError('Uh oh, you need FreeType and Pillow >= 10.1.10')  # noqa: TRY004
				font = default_font
			if '\n' in text:
				# This is more annoying then, because there is no ImageFont.getmultilinebbox
				# And I don't feel like doing all the line calculations myself to make it work (or do I) (would need to call getbbox for each line and then add the line spacing)
				# This isn't even necessarily a good idea if it wasn't for me never fucking with the spacing parameter of multiline_text when I use it or anything like that
				size = ImageDraw(Image.new('RGBA', (width, height))).multiline_textbbox(
					(0, 0), text, font
				)[2:]
			else:
				size = font.getbbox(text)[2:]
			if font_size == 1 or (
				(height is None or (size[1] + vertical_padding) <= height)
				and (width is None or (size[0] + horizontal_padding) <= width)
			):
				break
			font_size -= 1

		assert font, 'how did my font end up being None :('
		return font, size[0] + horizontal_padding, size[1] + vertical_padding

	out_width = 0
	out_height = 0
	out_font = font
	for t in text:
		t_font, t_width, t_height = fit_font(
			font, width, height, t, vertical_padding, horizontal_padding
		)
		if t_width > out_width:
			out_width = t_width
		if t_height > out_height:
			out_height = t_height
		out_font = t_font if out_font is None or t_font.size < out_font.size else out_font
	if not out_font:
		raise ValueError(
			'out_font ended up being None, maybe you provided an empty iterable for text'
		)
	return out_font, out_width, out_height


def combine_images_diagonally(first_image: Image.Image, second_image: Image.Image) -> Image.Image:
	"""Return a new image of the size of the first image with one diagonal half being from the first image, and the second half being from the second image"""
	if first_image.size != second_image.size:
		second_image = second_image.resize(first_image.size)
	# numpy.triu/tril won't work nicely on non-square rectangles
	orig_size = first_image.size
	max_dim = max(orig_size)
	square_size = max_dim, max_dim
	a = numpy.array(first_image.resize(square_size))
	b = numpy.array(second_image.resize(square_size))
	# triu/tril works on the last two axes, so we want those to be height and width
	upper_right = numpy.triu(a.swapaxes(0, 2)).swapaxes(0, 2)
	lower_left = numpy.tril(b.swapaxes(0, 2)).swapaxes(0, 2)
	return Image.fromarray(upper_right + lower_left).resize(orig_size)


def draw_centred_textbox(
	draw: ImageDraw,
	background_colour: tuple[float, float, float, float] | str,
	left: int,
	top: int,
	right: int,
	bottom: int,
	text: str,
	font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
):
	"""Draw a box with text in the centre with a 1 pixel border and the specified background colour, selecting white or black text colour as appropriate for readability"""
	if isinstance(background_colour, str):
		colour_as_int = ImageColor.getrgb(background_colour)
		background_colour = tuple(v / 255 for v in colour_as_int)
		assert background_colour, 'Mypy, why would it be a str at this point'
	else:
		colour_as_int = tuple(int(v * 255) for v in background_colour)
	# Am I stupid, or is there actually nothing in standard library or matplotlib that does this
	# Well colorsys.rgb_to_yiv would also potentially work
	luminance = (
		(background_colour[0] * 0.2126)
		+ (background_colour[1] * 0.7152)
		+ (background_colour[2] * 0.0722)
	)
	text_colour = 'white' if luminance <= 0.5 else 'black'
	draw.rectangle((left, top, left, bottom - 1), fill='black')
	draw.rectangle((right, top, right, bottom - 1), fill='black')
	draw.rectangle(
		(left + 1, top, right - 1, bottom - 1), fill=colour_as_int, outline='black', width=1
	)
	draw.text(
		((left + right) // 2, (top + bottom) // 2),
		text,
		anchor='mm',
		fill=text_colour,
		font=font,
		align='center',
	)


@cache
def get_character_image(char: Character) -> Image.Image:
	"""Yoink character select screen image for a character as a Pillow image."""
	url = char.character_select_screen_pic_url
	with CachedSession(
		'character_images', use_cache_dir=True
	) if have_requests_cache else requests.session() as session:
		response = session.get(
			url, timeout=10
		)  # stream=True doesn't work with CachedSession I think
		response.raise_for_status()
		return Image.open(BytesIO(response.content))


T = TypeVar('T')


@dataclass
class TieredItem(Generic[T]):
	"""An item that has a score associated with it used for ranking.

	item should not be None, and score should be a higher number for better items.
	"""

	item: T
	score: Floatable


class BaseTierList(Generic[T], ABC):
	"""Abstract tier list where all the logic is"""

	def __init__(
		self,
		items: Iterable[TieredItem[T]],
		tiers: int | Literal['auto'] | Sequence[int] = 'auto',
		tier_names: Sequence[str] | Mapping[int, str] | None = None,
		title: str | None = None,
		*,
		append_minmax_to_tier_titles: bool = False,
		score_formatter: str = '',
	) -> None:
		""":param tiers: Number of tiers to separate scores into. If "auto", tries to find a number of tiers that balances the size of each tier, but I made up that algorithm myself and I don't strictly speaking know what I'm doing so maybe it doesn't work. If a sequence, pre-computed tiers"""
		self.data = pandas.DataFrame(list(items), columns=['item', 'score'])
		self.data.sort_values('score', ascending=False, inplace=True)
		self.data['rank'] = numpy.arange(1, self.data.index.size + 1)

		if isinstance(tiers, Sequence) and tiers != 'auto':
			self.data['tier'] = tiers
			self.num_tiers = self._groupby.ngroups
			centroids = self._groupby['score'].mean().to_dict()
			self.tiers = Tiers(self.data['tier'], centroids, 0, 0)
		else:
			self.tiers = _get_clusters(self.data['score'], tiers)
			self.num_tiers = len(
				self.tiers.centroids
			)  # Might not be the same as tiers, esp if tiers is 'auto'
			self.data['tier'] = self.tiers.tiers

		self.append_minmax_to_tier_titles = append_minmax_to_tier_titles
		self.score_formatter = score_formatter
		# TODO: Option to provide custom images
		if not tier_names:
			tier_names = self.default_tier_names(self.num_tiers)
		if isinstance(tier_names, Sequence):
			tier_names = dict(enumerate(tier_names))
		self.tier_names = tier_names

		self.title = title

	@classmethod
	def from_items(
		cls,
		s: 'Mapping[T, Floatable] | pandas.Series[float]',
		tiers: int | Sequence[int] | Literal['auto'] = 'auto',
		tier_names: Sequence[str] | Mapping[int, str] | None = None,
		title: str | None = None,
		**kwargs,
	) -> 'Self':
		"""Create a new tier list from scores in a dict or pandas Series.
		If Series, assumes s is indexed by the items to be tiered."""
		return cls(itertools.starmap(TieredItem, s.items()), tiers, tier_names, title, **kwargs)

	@staticmethod
	def default_tier_names(length: int) -> Mapping[int, str]:
		"""Default tier names named after letters as is usually the case, starting with S at the top"""
		if length == 2:
			return {0: 'Good', 1: 'Bad'}
		if length == 3:
			return {0: 'Good', 1: 'Okay', 2: 'Bad'}
		if length == 6:
			# Just think it looks a bit weird to have it end at E tier
			return dict(enumerate('SABCDF'))
		tier_letters: Sequence[str] = 'SABCDEFGHIJKLMNOPQRSTUVXY'[:length]
		if length > 9:
			# Once you go past H it looks weird, so have the last one be Z
			tier_letters = list(tier_letters)
			tier_letters[-1] = 'Z'
		return dict(enumerate(tier_letters))

	def to_text(self, *, show_scores: bool = False) -> str:
		"""Return this tier list displayed as plain text"""
		text = '\n'.join(
			itertools.chain.from_iterable(
				(
					'=' * 20,
					self.displayed_tier_text(tier_number, group),
					'-' * 10,
					*(
						f'{row.rank}: {row.item}'
						+ (f' ({row.score:{self.score_formatter}})' if show_scores else '')
						for row in group.itertuples()
					),
					'',
				)
				for tier_number, group in self._groupby
			)
		)
		if self.title is not None:
			text = f'{self.title}\n' + text
		return text

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

	def displayed_tier_text(self, tier_number: int, group: pandas.DataFrame | None = None):
		"""
		:param tier_number: Tier number
		:param group: If you are already iterating through ._groupby, you can pass each group so you don't have to call get_group"""
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

	@abstractmethod
	def get_item_image(self, item: T) -> Image.Image:
		"""Return an image that represents item."""

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
		self,
		colourmap: 'str | Colormap | None' = None,
		max_images_per_row: int | None = 8,
		*,
		show_scores: bool = False,
		score_height: float = 0.5,
		title_background: tuple[int, int, int, int] | str | None = None,
	) -> Image.Image:
		"""Render the tier list as an image.

		This doesn't look too great if the images are of uneven size, but that's allowed."""
		max_image_width = max(im.width for im in self.images.values())
		max_image_height = max(im.height for im in self.images.values())
		if show_scores:
			scores = {
				item: f'{score:{self.score_formatter}}'
				for item, score in zip(self.data['item'], self.data['score'], strict=True)
			}
			score_font, _, score_height = fit_font(
				None,
				max_image_width,
				int(score_height if score_height > 1 else max_image_height * score_height),
				scores.values(),
				vertical_padding=0,
			)
			max_image_height += score_height

		tier_texts = {i: self.displayed_tier_text(i) for i in self._tier_texts}

		if colourmap is None or isinstance(colourmap, str):
			colourmap = pyplot.get_cmap(colourmap)

		font: ImageFont.FreeTypeFont | None = None
		font, textbox_width, _ = fit_font(
			font,
			None,
			max_image_height,
			tier_texts.values(),
			horizontal_padding=max_image_height,  # Not a typo, I just think that looks better
		)

		height = self._groupby.ngroups * max_image_height
		if self.title is not None:
			height += max_image_height
		if not max_images_per_row:
			max_images_per_row = self._groupby.size().max()
		width = (
			min(self._groupby.size().max(), max_images_per_row) * max_image_width + textbox_width
		)

		trans = (0, 0, 0, 0)
		image = Image.new('RGBA', (width, height), trans)
		draw = ImageDraw(image)

		next_line_y = 0
		if self.title is not None:
			next_line_y = max_image_height
			title_font = fit_font(None, width, max_image_height, self.title)[0]
			draw_centred_textbox(
				draw,
				title_background if title_background else trans,
				0,
				0,
				width,
				max_image_height,
				self.title,
				title_font,
			)

		actual_width = width
		for tier_number, group in self._groupby:
			tier_text = tier_texts[tier_number]

			row_height = max_image_height * (((group.index.size - 1) // max_images_per_row) + 1)

			box_end = next_line_y + row_height
			if box_end > image.height:
				image = pad_image(image, image.width, box_end, trans)
				draw = ImageDraw(image)
			if textbox_width > image.width:  # This probably doesn't happen
				image = pad_image(image, textbox_width, image.height, trans)
				draw = ImageDraw(image)

			colour = colourmap(self.scaled_centroids[tier_number])
			draw_centred_textbox(
				draw, colour, 0, next_line_y, textbox_width, box_end, tier_text, font
			)

			next_image_x = textbox_width + 1  # Account for border
			for i, item in enumerate(group['item']):
				item_image = self.images[item]
				if show_scores:
					orig_height = item_image.height
					item_image = pad_image(
						item_image, item_image.width, orig_height + score_height, trans
					)
					draw_centred_textbox(
						ImageDraw(item_image),
						'black',
						0,
						orig_height,
						item_image.width,
						item_image.height,
						scores[item],
						score_font,
					)

				image_row, image_col = divmod(i, max_images_per_row)
				if not image_col:
					next_image_x = textbox_width + 1

				image_y = next_line_y + (max_image_height * image_row)
				if next_image_x + item_image.width > image.width:
					image = pad_image(image, next_image_x + item_image.width, image.height, trans)
					draw = ImageDraw(image)
					# TODO: Optionally draw the score or name below each character (maybe that's better off with an overriten get_item_image, or maybe get_item_image_with_score)
				image.paste(item_image, (next_image_x, image_y))
				next_image_x += item_image.width
				actual_width = max(actual_width, next_image_x)
			next_line_y = box_end

		# Uneven images can result in calculating too much space to the side
		image = image.crop((0, 0, actual_width, next_line_y))

		# TODO: Allow for custom background either via parameter or property supplied via constructor
		background = generate_background(image.width, image.height)
		background.paste(image, mask=image)
		return background


class TierList(BaseTierList[T]):
	"""Default implementation of TierList that just displays text as images"""

	# Default size of load_default 10, so that sucks and we won't do that
	_default_font = ImageFont.load_default(20)
	_reg = re.compile(r'\s+?(?=\b\w{5,})')  # Space followed by at least 5-letter word

	def get_item_image(self, item: T) -> Image.Image:
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
				pad_image(image, max_width + 2, max_height + 2, (0, 0, 0, 0), centred=True)
			)
			for item, image in unscaled.items()
		}


class CharacterTierList(BaseTierList[Character]):
	"""Tier list specifically for characters, which would be one of the most likely use cases"""

	def __init__(
		self,
		items: Iterable[TieredItem[Character]],
		tiers: int | Literal['auto'] | Sequence[int] = 'auto',
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
			return combine_images_diagonally(first_image, second_image)

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
		scores, title='Test tier list\nCharacter name lengths', append_minmax_to_tier_titles=True
	)
	print(
		tierlist.tiers.inertia,
		tierlist.tiers.kmeans_iterations,
		tierlist.tiers.tiers.nunique(),
		len(tierlist.tiers.centroids),
	)
	print(tierlist.tiers.centroids)
	print(tierlist.to_text())
	tierlist.to_image('rainbow_r', show_scores=True).show()

	players = [p for p in Elo.for_game('SSBU', 'ACT') if p.is_active]
	listy = TextBoxTierList(
		[TieredItem(player.player, player.elo) for player in players],
		title='Active ACT players by Elo',
		append_minmax_to_tier_titles=True,
		score_formatter=',',
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
