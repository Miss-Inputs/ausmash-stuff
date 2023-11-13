#!/usr/bin/env python3

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy
import pandas
import requests
from matplotlib import pyplot  #just for colour maps lol
from PIL import Image, ImageFilter, ImageFont
from PIL.ImageDraw import ImageDraw
from sklearn.cluster import KMeans

from ausmash import Character
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

def _tier_list_to_text(df: pandas.DataFrame, tier_names: Mapping[int, str]) -> str:
	lines = []
	for tier_number, group in df.groupby('tier'):
		lines.append('=' * 20)
		lines.append(f'{tier_names.get(tier_number, tier_number)}: {group.score.min()} to {group.score.max()}')
		lines.append('-' * 10)

		lines += (f'{row.n}: {row.character}' for row in group.itertuples())
		lines.append('')
	return '\n'.join(lines)

def get_combined_char_image(character: CombinedCharacter) -> Image.Image:
	if len(character.chars) == 2:
		#Divvy it up into two diagonally, I dunno how to make it look the least shite
		first, second = character.chars
		first_image = get_char_image(first)
		second_image = get_char_image(second)
		if first_image.size != second_image.size:
			second_image = second_image.resize(first_image.size)
		#numpy.triu/tril won't work nicely on non-square rectangles
		orig_size = first_image.size
		max_dim = max(orig_size)
		square_size = max_dim, max_dim
		a = numpy.array(first_image.resize(square_size))
		b = numpy.array(second_image.resize(square_size))
		return Image.fromarray(numpy.triu(a.swapaxes(0, 2)).swapaxes(0, 2) + numpy.tril(b.swapaxes(0, 2)).swapaxes(0, 2)).resize(orig_size)
	
	images = [numpy.array(get_char_image(char)) for char in character.chars]
	return Image.fromarray(numpy.mean(images, axis=(0)).astype('uint8'))

def get_char_image(character: Character) -> Image.Image:
	if isinstance(character, CombinedCharacter):
		return get_combined_char_image(character)
	url = character.character_select_screen_pic_url
	response = requests.get(url, stream=True, timeout=10)
	response.raise_for_status()
	image = Image.open(response.raw)

	# if image.mode != 'RGBA': #Don't do this
	# 	alpha = (numpy.array(image) != character.colour).all(axis=2)
	# 	image.putalpha(Image.fromarray(alpha))
	return image

def generate_background(width: int, height: int) -> Image.Image:
	#Generate some weird clouds
	rng = numpy.random.default_rng()
	noise = rng.integers(0, (128, 128, 128), (height, width, 3), 'uint8', True)
	#Could also have a fourth dim with max=255 and then layer multiple transparent clouds on top of each other
	image = Image.fromarray(noise).filter(ImageFilter.ModeFilter(100)).filter(ImageFilter.GaussianBlur(50))
	return image

def _tier_list_to_image(df: pandas.DataFrame, centroids: Mapping[int, float], tier_names: Mapping[int, str], colourmap_name: str | None=None) -> Image.Image:
	groupby = cast('DataFrameGroupBy[int]', df.groupby('tier'))
	images = {char: get_char_image(char) for char in df['character']}
	max_image_width = max(im.width for im in images.values())
	max_image_height = max(im.height for im in images.values())
	#Need to start off with some image size
	#We start off with enough to get all the images, but this won't actually be enough, because of the text boxes and such

	max_centroid = max(centroids.values())
	#If max_centroid is 0 we'd divide by zero so fall back to just using colours indexed by tier number
	cmap = pyplot.get_cmap(colourmap_name, lut=None if max_centroid else groupby.ngroups)

	tier_texts = {tier_number: tier_names.get(tier_number, f'Tier {tier_number}') for tier_number in groupby.groups}

	font = None
	textbox_width = 0

	vertical_padding = 10
	horizontal_padding = 10
	font_size = 100
	for text in tier_texts.values():
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

	height = groupby.ngroups * max_image_height
	width = groupby.size().max() * max_image_width + textbox_width

	#TODO: Maybe you could have a max images per row, and then row height = max_image_height * (max number of images in each tier // max images per row)
	
	trans = (0,0,0,0)
	image = Image.new('RGBA', (width, height), trans)
	draw = ImageDraw(image)

	next_line_y = 0
	for tier_number, group in groupby:
		tier_text = tier_texts[tier_number]
	
		box_end = next_line_y + max_image_height
		if box_end > image.height or textbox_width > image.width:
			new_image = Image.new('RGBA', (max(image.width, textbox_width), max(image.height, box_end)), trans)
			new_image.paste(image)
			image = new_image
			draw = ImageDraw(image)

		if max_centroid:
			#Scale to [0.0, 1.0] interval
			colour = cmap(centroids[tier_number] / max_centroid)
		else:
			colour = cmap(tier_number)
		colour_as_int = tuple(int(v * 255) for v in colour)
		#Am I stupid, or is there actually nothing in standard library or matplotlib that does this
		#Well colorsys.rgb_to_yiv would also potentially work
		luminance = (colour[0] * 0.2126) + (colour[1] * 0.7152) + (colour[2] * 0.0722)
		text_colour = 'white' if luminance <= 0.5 else 'black'
		draw.rectangle((0, next_line_y, textbox_width, box_end), fill=colour_as_int, outline='black', width=2)
		draw.text((textbox_width / 2, (next_line_y + box_end) / 2), tier_text, anchor='mm', fill=text_colour, font=font)
		
		next_image_x = textbox_width + 1 #Account for border
		for char in group['character']:
			char_image = images[char]
			if next_image_x + char_image.width > image.width:
				new_image = Image.new('RGBA', (next_image_x + char_image.width, image.height), trans)
				new_image.paste(image)
				image = new_image
				draw = ImageDraw(image)
				#TODO: Optionally draw the score or name below each character
			image.paste(char_image, (next_image_x, next_line_y))
			next_image_x += char_image.width
		# next_text_x = bbox[0]
		next_line_y = box_end

	background = generate_background(image.width, image.height)
	background.paste(image, mask=image)
	return background		

@overload
def draw_tier_list(characters: Iterable[Character], scores: Iterable[float], output_type: Literal['image'], num_tiers: int=7, colourmap_name: str | None=None) -> Image.Image: ...

@overload
def draw_tier_list(characters: Iterable[Character], scores: Iterable[float], output_type: Literal['str'], num_tiers: int=7, colourmap_name: str | None=None) -> str: ...

@overload
def draw_tier_list(characters: Iterable[Character], scores: Iterable[float], output_type: Literal['frame'], num_tiers: int=7, colourmap_name: str | None=None) -> tuple[pandas.DataFrame, Mapping[int, float]]: ...

def draw_tier_list(characters: Iterable[Character], scores: Iterable[float], output_type: Literal['str', 'image', 'frame']='frame', num_tiers: int=7, colourmap_name: str | None=None) -> tuple[pandas.DataFrame, Mapping[int, float]] | Image.Image | str:
	df = pandas.DataFrame({'character': characters, 'score': scores})
	df.sort_values('score', ascending=False, inplace=True)
	df['tier'], centroids = _get_tiers(df['score'], num_tiers)
	df['n'] = numpy.arange(1, df.index.size + 1)
	
	tier_letters = list('SABCDEFGHIJKLZ')
	#TODO: Argument to provide tier names, or use centroid/min to max of each tier instead, etc
	tier_names = dict(enumerate(tier_letters))

	if output_type == 'frame':
		return df, centroids
	if output_type == 'str':
		return _tier_list_to_text(df, tier_names)
	if output_type == 'image':
		# print(df.groupby('tier', as_index=False, group_keys=False).score.agg(['min', 'max']))
		return _tier_list_to_image(df, centroids, tier_names, colourmap_name)
	raise ValueError(output_type)
	
def main() -> None:
	#TODO: Proper command line interfacey
	print('zzzz')

if __name__ == '__main__':
	main()