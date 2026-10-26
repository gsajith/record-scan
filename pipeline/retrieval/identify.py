"""Identify a cover photo against a built index.

    python -m pipeline.retrieval.identify photo.jpg --model NAME [--index DIR] [--k 5] [--views 1|3]

--index defaults to config.EMB_DIR/<model>. Pass it to query an index built in another
shard folder. The 2025-11 clip-vit-b-32-aug13 shards used plain GELU ViT-B-32 rather than
this QuickGELU baseline, so true matches against them score about 0.95, not 1.0: fine for a
smoke test, never for numbers.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
from PIL import Image

from pipeline import config
from pipeline.metadata.catalog import Catalog
from pipeline.retrieval.embedders import EMBEDDER_NAMES, Embedder, load_embedder, query_views


@dataclass
class Match:
    mbid: str
    title: str
    artists: tuple[str, ...]
    score: float  # cosine similarity, max over query views and over the releases in the group


class Identifier:
    def __init__(
        self,
        embedder: Embedder,
        index_dir: Path,
        catalog: Catalog,
        clusters: dict[str, int] | None = None,
        ef: int = 128,
    ):
        self.embedder = embedder
        self.catalog = catalog
        self.clusters = clusters
        self.ef = ef
        self.labels = np.load(index_dir / "labels.npy")
        self.index = hnswlib.Index(space="cosine", dim=embedder.dim)
        self.index.load_index(str(index_dir / "index.hnsw"))
        # build_index swaps the two files one after the other, so a crash between can pair them wrong
        if self.index.get_current_count() != len(self.labels):
            raise SystemExit(f"{index_dir} has an index and labels of different sizes. Rerun build_index.")

    def identify(self, img: Image.Image, k: int = 5, views: int = 3) -> list[Match]:
        """Top k albums: k*10 neighbours per view, best score per group, one match per group.

        A group is the duplicate cluster when clusters are given, else the album (title plus
        primary artist), so reissues of one album take a single slot. When one big group fills
        the neighbours and fewer than k groups come back, the search widens 4x, at most three
        times, and returns what it found.
        """
        vectors = self.embedder.embed(query_views(img)[:views])
        for widening in range(4):
            neighbours = min(k * 10 * 4**widening, len(self.labels))
            best = self._best_per_group(vectors, neighbours)
            if len(best) >= k or neighbours == len(self.labels):
                break
        top = sorted(best.values(), reverse=True)[:k]
        return [self._match(mbid, score) for score, mbid in top]

    def _best_per_group(self, vectors: np.ndarray, neighbours: int) -> dict[str | int, tuple[float, str]]:
        """Group key -> (best score, its mbid) over the nearest neighbours of every view."""
        self.index.set_ef(max(self.ef, neighbours))  # hnswlib needs ef >= the neighbour count
        ids, distances = self.index.knn_query(vectors, k=neighbours)
        best: dict[str | int, tuple[float, str]] = {}
        for mbid, score in zip(self.labels[ids.ravel()], 1 - distances.ravel()):
            group = self._group(str(mbid))
            if group not in best or score > best[group][0]:
                best[group] = (float(score), str(mbid))
        return best

    def _group(self, mbid: str) -> str | int:
        if self.clusters is not None:
            return self.clusters.get(mbid, mbid)
        try:
            key = self.catalog.album_key(mbid)
        except KeyError:  # a cover without a catalog row stands alone
            return mbid
        # "+" and "÷" both normalize to an empty title, and unknown artists to an empty name
        title, artist = key.split("|", 1)
        return key if title and artist else mbid

    def _match(self, mbid: str, score: float) -> Match:
        release = self.catalog.release(mbid)
        if release is None:
            return Match(mbid, "", (), score)
        return Match(mbid, release.title, release.artists, score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--model", required=True, choices=EMBEDDER_NAMES)
    parser.add_argument("--index", type=Path, help="index folder, default EMB_DIR/<model>")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--views", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()

    index_dir = args.index or config.EMB_DIR / args.model
    if not (index_dir / "index.hnsw").is_file():
        raise SystemExit(f"No index in {index_dir}. Build one with: python -m pipeline.retrieval.build_index")
    identifier = Identifier(load_embedder(args.model), index_dir, Catalog())
    for rank, m in enumerate(identifier.identify(Image.open(args.photo), args.k, args.views), 1):
        print(f"{rank}. {m.score:.3f}  {', '.join(m.artists) or '?'} - {m.title or '?'}  {m.mbid}")


if __name__ == "__main__":
    main()
