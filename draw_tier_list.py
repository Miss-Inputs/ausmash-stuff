#!/usr/bin/env python3

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy
import pandas
import requests
from matplotlib import pyplot  #just for colour maps lol
from PIL import Image, ImageFilter, ImageFont
from PIL.ImageDraw import ImageDraw
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

from ausmash import Character, combine_echo_fighters
from ausmash.models.character import CombinedCharacter

if TYPE_CHECKING:
	from pandas.core.groupby.generic import DataFrameGroupBy


def _get_tiers(scores: 'pandas.Series[float]', n_clusters: int) -> tuple[pandas.Series, Mapping[int, float]]:
	#Returns tiers for each row, centroids
	kmeans = KMeans(n_clusters, n_init='auto', random_state=0)
	labels = kmeans.fit_predict(scores.to_numpy().reshape(-1, 1))
	raw_tiers = pandas.Series(labels, index=scores.index, name='tiers')
	#The numbers in raw_tiers are just random values for the sake of being distinct, we are already sorted by score, so have ascending tiers instead
	mapping = {c: i for i, c in enumerate(raw_tiers.unique())}
	tiers = raw_tiers.map(mapping)
	centroids = pandas.Series(kmeans.cluster_centers_.squeeze(), name='centroids').rename(mapping).sort_index().to_dict()
	return tiers, centroids

def generate_background(width: int, height: int) -> Image.Image:
	#Generate some weird clouds
	rng = numpy.random.default_rng()
	noise = rng.integers(0, (128, 128, 128), (height, width, 3), 'uint8', True)
	#Could also have a fourth dim with max=255 and then layer multiple transparent clouds on top of each other
	image = Image.fromarray(noise).filter(ImageFilter.ModeFilter(100)).filter(ImageFilter.GaussianBlur(50))
	return image

T = TypeVar('T')
class TierList(Generic[T], ABC):
	def __init__(self, items: Iterable[T], scores: Iterable[float], num_tiers: int=7) -> None:
		self.df = pandas.DataFrame(list(zip(items, scores)), columns=['item', 'score'])
		self.df.sort_values('score', ascending=False, inplace=True)
		self.df['tier'], self.centroids = _get_tiers(self.df['score'], num_tiers)
		self.df['rank'] = numpy.arange(1, self.df.index.size + 1)
		
		tier_letters = list('SABCDEFGHIJKLZ')
		#TODO: Argument to provide tier names, or append min/max of each tier to _tier_texts, etc
		self.tier_names = dict(enumerate(tier_letters))

	def to_text(self) -> str:
		lines = []
		for tier_number, group in self._groupby:
			lines.append('=' * 20)
			lines.append(f'{self.tier_names.get(tier_number, tier_number)}: {group.score.min()} to {group.score.max()}')
			lines.append('-' * 10)

			lines += (f'{row.rank}: {row.item}' for row in group.itertuples())
			lines.append('')
		return '\n'.join(lines)
	
	@cached_property
	def _groupby(self) -> 'DataFrameGroupBy[int]':
		return self.df.groupby('tier')
	
	@cached_property
	def _tier_texts(self) -> Mapping[int, str]:
		"""Based off tier names, not necessarily using all of them if there aren't as many tiers, falling back to a default name if needed"""
		return {tier_number: self.tier_names.get(tier_number, f'Tier {tier_number}') for tier_number in self._groupby.groups}

	@staticmethod
	@abstractmethod
	def get_item_image(item: T) -> Image.Image: ...

	@cached_property
	def scaled_centroids(self) -> Mapping[int, float]:
		"""Scale self.centroids between 0.0 and 1.0"""
		#Don't worry, it still works on 1D arrays even if it says it wants a MatrixLike in the type hint
		#If it stops working in some future version use reshape(-1, 1)
		values = minmax_scale(list(self.centroids.values()))
		return {k: values[k] for k in self.centroids}

	def to_image(self, colourmap_name: str | None=None, max_images_per_row: int=8) -> Image.Image:
		images = {char: self.get_item_image(char) for char in self.df['item']}
		max_image_width = max(im.width for im in images.values())
		max_image_height = max(im.height for im in images.values())
		#Need to start off with some image size
		#We start off with enough to get all the images, but this won't actually be enough, because of the text boxes and such

		cmap = pyplot.get_cmap(colourmap_name)

		font = None
		textbox_width = 0

		vertical_padding = 10
		horizontal_padding = 10
		#Find the largest font size we can use inside the tier name box to fit the available height
		font_size = max_image_height #font_size is points, but we can assume n points is >= n pixels, so it'll do as a starting point
		for text in self._tier_texts.values():
			size: None | tuple[int, int, int, int] = None
			while (size is None or (size[3] + vertical_padding) > max_image_height) and font_size > 0:
				font = ImageFont.load_default(font_size)
				size = font.getbbox(text)
				font_size -= 1
			assert font, 'how did my font end up being null :('
			assert size, 'meowwww'
			length = size[2] + horizontal_padding
			if length > textbox_width:
				textbox_width = length

		height = self._groupby.ngroups * max_image_height
		if not max_images_per_row:
			max_images_per_row = self._groupby.size().max()
		width = min(self._groupby.size().max(), max_images_per_row) * max_image_width + textbox_width
		
		trans = (0,0,0,0)
		image = Image.new('RGBA', (width, height), trans)
		draw = ImageDraw(image)

		next_line_y = 0
		for tier_number, group in self._groupby:
			tier_text = self._tier_texts[tier_number]

			row_height = max_image_height * (((group.index.size - 1) // max_images_per_row) + 1)
		
			box_end = next_line_y + row_height
			if box_end > image.height:
				new_image = Image.new('RGBA', (image.width, box_end), trans)
				new_image.paste(image)
				image = new_image
				draw = ImageDraw(image)
			if textbox_width > image.width: #This probably doesn't happen
				new_image = Image.new('RGBA', (textbox_width, image.height), trans)
				new_image.paste(image)
				image = new_image
				draw = ImageDraw(image)

			colour = cmap(self.scaled_centroids[tier_number])
			colour_as_int = tuple(int(v * 255) for v in colour)
			#Am I stupid, or is there actually nothing in standard library or matplotlib that does this
			#Well colorsys.rgb_to_yiv would also potentially work
			luminance = (colour[0] * 0.2126) + (colour[1] * 0.7152) + (colour[2] * 0.0722)
			text_colour = 'white' if luminance <= 0.5 else 'black'
			draw.rectangle((0, next_line_y, 0, box_end - 1), fill='black')
			draw.rectangle((textbox_width, next_line_y, textbox_width, box_end - 1), fill='black')
			draw.rectangle((1, next_line_y, textbox_width - 1, box_end - 1), fill=colour_as_int, outline='black', width=1)
			draw.text((textbox_width / 2, (next_line_y + box_end) / 2), tier_text, anchor='mm', fill=text_colour, font=font)
			
			next_image_x = textbox_width + 1 #Account for border
			for i, char in enumerate(group['item']):
				char_image = images[char]
				image_row, image_col = divmod(i, max_images_per_row)
				if not image_col:
					next_image_x = textbox_width + 1

				image_y = next_line_y + (max_image_height * image_row)
				if next_image_x + char_image.width > image.width:
					new_image = Image.new('RGBA', (next_image_x + char_image.width, image.height), trans)
					new_image.paste(image)
					image = new_image
					draw = ImageDraw(image)
					#TODO: Optionally draw the score or name below each character
				image.paste(char_image, (next_image_x, image_y))
				next_image_x += char_image.width
			next_line_y = box_end

		background = generate_background(image.width, image.height)
		background.paste(image, mask=image)
		return background
	
class CharacterTierList(TierList[Character]):

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
			#Divvy it up into two diagonally, I dunno how to make it look the least shite
			first, second = character.chars
			first_image = CharacterTierList.get_item_image(first)
			second_image = CharacterTierList.get_item_image(second)
			if first_image.size != second_image.size:
				second_image = second_image.resize(first_image.size)
			#numpy.triu/tril won't work nicely on non-square rectangles
			orig_size = first_image.size
			max_dim = max(orig_size)
			square_size = max_dim, max_dim
			a = numpy.array(first_image.resize(square_size))
			b = numpy.array(second_image.resize(square_size))
			return Image.fromarray(numpy.triu(a.swapaxes(0, 2)).swapaxes(0, 2) + numpy.tril(b.swapaxes(0, 2)).swapaxes(0, 2)).resize(orig_size)

		#Just merge them together if we have a combined character with 3 or more	
		images = [numpy.array(CharacterTierList.get_item_image(char)) for char in character.chars]
		return Image.fromarray(numpy.mean(images, axis=(0)).astype('uint8'))
	
def main() -> None:
	#TODO: Proper command line interfacey
	print('testing for now')
	miis = set()
	chars: set[Character] = set()
	for char in Character.characters_in_game('SSBU'):
		if char.name.startswith('Mii'):
			miis.add(char)
		else:
			chars.add(combine_echo_fighters(char))
	chars.add(CombinedCharacter('Mii Fighters', miis))
	tierlist = CharacterTierList(chars, [len(char.name) for char in chars], 7)
	tierlist.to_image('Spectral').show()

if __name__ == '__main__':
	main()
