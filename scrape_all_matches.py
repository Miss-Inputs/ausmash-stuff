#!/usr/bin/env python3

"""Save all the matches to a pickle file and then just use that file everywhere else you wanna use a whole lot of match data"""

import pickle
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import NamedTuple

from ausmash import (
	Event,
	Match,
	Player,
	Result,
	Tournament,
)
from pyzstd import ZstdFile
from tqdm import tqdm


class MatchEtc(NamedTuple):
	"""Return everything all at once so we don't have to look it up again"""

	tournament: Tournament
	event: Event
	results: Mapping[Player | str, Result]
	match: Match


def iter_matches() -> Iterator[MatchEtc]:
	with tqdm(
		Tournament.all_with_results(),
		desc='Getting matches from all tournaments',
		unit='tournament',
	) as t:
		for tournament in t:
			t.set_postfix(tournament=tournament)
			try:
				events = tournament.events
				for event in events:
					results = {
						result.player or result.player_name: result
						for result in Result.results_for_event(event)
					}
					matches = Match.matches_at_event(event)
					for match in matches:
						yield MatchEtc(tournament, event, results, match)
			except KeyboardInterrupt:
				break


def get_all_matches() -> tuple[Path, list[MatchEtc]]:
	base_path = Path(
		f'/media/Shared/Datasets/Smash/Matches/All tournaments + events + results + matches ({date.today().isoformat()})'
	)
	pickle_path = base_path.with_suffix('.pickle.zst')
	try:
		with ZstdFile(pickle_path) as z:
			matches = pickle.load(z)
	except FileNotFoundError:
		matches = list(iter_matches())
		with ZstdFile(pickle_path, 'w') as z:
			pickle.dump(matches, z)
		just_matches = [match for _, _, _, match in matches]
		with ZstdFile(
			pickle_path.with_stem(
				pickle_path.stem.replace(
					'tournaments + events + results + matches', 'matches'
				)
			),
			'w',
		) as z:
			pickle.dump(just_matches, z)

	return base_path, matches

if __name__ == '__main__':
	get_all_matches()
