import hashlib
import json
import logging
import pickle
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .export import EXPORT_FORMAT_VERSION
from .forest import PackedForest

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def _compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_string_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _parse_graphs_file(graphs_file: str) -> dict[str, str]:
    """graph_id -> hash of the graph's lines, starting at its @id line."""
    hashes = {}
    graph_id = None
    lines = []
    with open(graphs_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.startswith('@'):
                lines.append(line)
                continue
            if graph_id is not None:
                hashes[graph_id] = _compute_string_hash('\n'.join(lines))
            graph_id = line[1:].strip()
            lines = [line]
    if graph_id is not None:
        hashes[graph_id] = _compute_string_hash('\n'.join(lines))
    return hashes


def _sanitize_graph_id(graph_id: str) -> str:
    for ch in '/\\: ':
        graph_id = graph_id.replace(ch, '_')
    return graph_id[:100]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    load_time: float = 0.0
    save_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class ForestCache:
    cache_dir: Path
    grammar_hash: str
    graph_hashes: dict[str, str] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        # Forests from different exporter versions get separate directories; old ones stay.
        suffix = "" if EXPORT_FORMAT_VERSION == 1 else f".v{EXPORT_FORMAT_VERSION}"
        self._grammar_dir = self.cache_dir / f"{self.grammar_hash[:16]}{suffix}"
        self._forests_dir = self._grammar_dir / "forests"
        self._metadata_file = self._grammar_dir / "metadata.json"
        self._batch_file = self._grammar_dir / "batch.pkl"

    @classmethod
    def from_files(cls, cache_dir: str, grammar_file: str, graphs_file: str) -> 'ForestCache':
        cache = cls(
            cache_dir=Path(cache_dir),
            grammar_hash=_compute_file_hash(grammar_file),
            graph_hashes=_parse_graphs_file(graphs_file),
        )
        cache._forests_dir.mkdir(parents=True, exist_ok=True)
        cache._save_metadata(grammar_file, graphs_file)
        return cache

    def _save_metadata(self, grammar_file, graphs_file):
        metadata = {
            "version": CACHE_VERSION,
            "grammar_file": str(grammar_file),
            "graphs_file": str(graphs_file),
            "grammar_hash": self.grammar_hash,
            "num_graphs": len(self.graph_hashes),
            "created": time.time(),
        }
        with open(self._metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _forest_path(self, graph_id: str) -> Path:
        graph_hash = self.graph_hashes.get(graph_id, "")
        return self._forests_dir / f"{_sanitize_graph_id(graph_id)}.{graph_hash[:8]}.pkl"

    def is_valid(self, graph_id: str) -> bool:
        # The current graph hash is in the file name, so existence is the whole check.
        return self._forest_path(graph_id).exists()

    def save_forest(self, graph_id: str, forest: PackedForest, cleanup_stale: bool = False) -> None:
        start = time.time()
        path = self._forest_path(graph_id)
        if cleanup_stale:
            for old in self._forests_dir.glob(f"{_sanitize_graph_id(graph_id)}.*.pkl"):
                if old != path:
                    old.unlink()
        with open(path, 'wb') as f:
            pickle.dump(forest, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.stats.save_time += time.time() - start

    def load_forest(self, graph_id: str) -> PackedForest | None:
        start = time.time()
        if not self.is_valid(graph_id):
            self.stats.misses += 1
            return None
        path = self._forest_path(graph_id)
        try:
            with open(path, 'rb') as f:
                forest = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached forest for {graph_id}: {e}")
            self.stats.misses += 1
            self.stats.invalidations += 1
            path.unlink(missing_ok=True)
            return None
        self.stats.hits += 1
        self.stats.load_time += time.time() - start
        return forest

    def has_all(self, graph_ids: list[str]) -> bool:
        return all(self.is_valid(gid) for gid in graph_ids)

    def _batch_matches(self, batch) -> bool:
        return (batch.get("version") == CACHE_VERSION
                and batch.get("grammar_hash") == self.grammar_hash
                and batch.get("graph_hashes") == self.graph_hashes)

    def save_batch(self, forests: dict[str, PackedForest]) -> None:
        start = time.time()
        for graph_id, forest in forests.items():
            self.save_forest(graph_id, forest)
        batch = {
            "version": CACHE_VERSION,
            "grammar_hash": self.grammar_hash,
            "graph_hashes": self.graph_hashes,
            "forests": forests,
        }
        with open(self._batch_file, 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.stats.save_time += time.time() - start
        logger.info(f"Saved batch of {len(forests)} forests")

    def load_batch(self) -> dict[str, PackedForest]:
        """Falls back to the per-graph files when the batch file is missing or stale."""
        start = time.time()
        if self._batch_file.exists():
            try:
                with open(self._batch_file, 'rb') as f:
                    batch = pickle.load(f)
                if self._batch_matches(batch):
                    forests = batch["forests"]
                    self.stats.hits += len(forests)
                    self.stats.load_time += time.time() - start
                    logger.info(f"Loaded batch of {len(forests)} forests")
                    return forests
                logger.info("Batch file stale, loading individual files")
                self.stats.invalidations += 1
            except Exception as e:
                logger.warning(f"Failed to load batch file: {e}")

        forests = {}
        for graph_id in self.graph_hashes:
            forest = self.load_forest(graph_id)
            if forest is not None:
                forests[graph_id] = forest
        self.stats.load_time += time.time() - start
        return forests

    def use_batch_mode(self) -> bool:
        if not self._batch_file.exists():
            return False
        try:
            with open(self._batch_file, 'rb') as f:
                return self._batch_matches(pickle.load(f))
        except Exception:
            return False

    def clear(self) -> None:
        if self._grammar_dir.exists():
            shutil.rmtree(self._grammar_dir)
        self.stats = CacheStats()
        logger.info(f"Cleared cache for grammar {self.grammar_hash[:16]}")

    def get_stats(self) -> dict:
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "invalidations": self.stats.invalidations,
            "hit_rate": self.stats.hit_rate,
            "load_time": self.stats.load_time,
            "save_time": self.stats.save_time,
            "num_cached": sum(1 for gid in self.graph_hashes if self.is_valid(gid)),
            "num_graphs": len(self.graph_hashes),
        }


def parse_with_hard_timeout(grammar_file: str, graphs_file: str, index: int, seconds: float):
    """Parse one graph in a child process, killed after `seconds`.

    Returns the parser's error name ("kNone" on success), or None on timeout.
    """
    import multiprocessing

    def _worker(queue):
        import pyshrg
        pyshrg.initialize()
        manager = pyshrg.get_manager()
        manager.load_grammars(grammar_file)
        manager.load_graphs(graphs_file)
        manager.allocate(num_contexts=1)
        manager.init_all(type="linear", verbose=False)
        queue.put(manager.get_context(0).parse(manager.get_graph(index)).name)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_worker, args=(queue,))
    process.start()
    process.join(seconds)
    if process.is_alive():
        process.terminate()
        process.join(2)
        return None
    return queue.get() if not queue.empty() else None


class ParseTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ParseTimeout("Parse timed out")


def _empty_forest(num_rules, graph_id):
    return PackedForest(nodes=[], hyperedges=[], roots=[], num_rules=num_rules, graph_id=graph_id)


def _load_ordered_batch(cache, graph_ids):
    cached = cache.load_batch()
    if cached and len(cached) >= len(graph_ids) and all(gid in cached for gid in graph_ids):
        return [cached[gid] for gid in graph_ids]
    return None


def load_forests_with_cache(
    grammar_file: str,
    graphs_file: str,
    cache_dir: str | None = None,
    timeout: int = 10,
    batch_cache: bool = True,
    verbose: bool = True,
    parser_type: str = "linear",
) -> tuple:
    """Returns (forests, graph_ids, sentences, num_rules, cache); cache is None without cache_dir."""
    import signal
    from .export import export_forest
    from .evaluation.bleu import reference_from_graph

    try:
        import pyshrg
    except ImportError:
        raise ImportError("pyshrg not found. Make sure the C++ parser is built.")

    pyshrg.initialize()
    manager = pyshrg.get_manager()
    manager.load_grammars(grammar_file)
    manager.load_graphs(graphs_file)
    manager.allocate(num_contexts=1)
    manager.init_all(type=parser_type, verbose=False)
    context = manager.get_context(0)
    # Forest rule ids are HRG indices (item.grammar_index), so size by hrg_size, not shrg_size.
    num_rules = manager.hrg_size

    graph_ids = []
    sentences = []
    for i in range(manager.graph_size):
        graph = manager.get_graph(i)
        graph_ids.append(graph.sentence_id)
        sentences.append(reference_from_graph(graph))

    cache = None
    if cache_dir:
        try:
            cache = ForestCache.from_files(cache_dir, grammar_file, graphs_file)
        except Exception as e:
            if verbose:
                logger.warning(f"Cache initialization failed: {e}")

    if cache and batch_cache and cache.use_batch_mode():
        forests = _load_ordered_batch(cache, graph_ids)
        if forests is not None:
            if verbose:
                logger.info(f"Loaded {len(forests)} forests from batch cache")
            return forests, graph_ids, sentences, num_rules, cache

    forests = []
    cache_hits = parse_count = timeout_count = fail_count = 0
    # SIGALRM cannot interrupt the C++ parser, so this only labels slow graphs after the
    # fact; parse_with_hard_timeout is the real bound.
    use_timeout = timeout > 0 and hasattr(signal, 'SIGALRM')
    if use_timeout:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        for i in range(manager.graph_size):
            graph = manager.get_graph(i)
            graph_id = graph.sentence_id

            if cache is not None:
                forest = cache.load_forest(graph_id)
                if forest is not None:
                    forests.append(forest)
                    cache_hits += 1
                    continue

            try:
                if use_timeout:
                    signal.alarm(timeout)
                error = context.parse(graph)
                if error.name == 'kNone':
                    forest = export_forest(context, graph)
                    forests.append(forest)
                    if cache is not None:
                        cache.save_forest(graph_id, forest)
                    parse_count += 1
                else:
                    if verbose:
                        logger.debug(f"Parse failed for graph {i}: {error.name}")
                    forests.append(_empty_forest(num_rules, graph_id))
                    fail_count += 1
                if use_timeout:
                    signal.alarm(0)
            except ParseTimeout:
                timeout_count += 1
                if verbose:
                    logger.debug(f"Parse timeout for graph {i} (>{timeout}s)")
                forests.append(_empty_forest(num_rules, graph_id))
            except Exception as e:
                if use_timeout:
                    signal.alarm(0)
                if verbose:
                    logger.warning(f"Failed to export forest for graph {i}: {e}")
                forests.append(_empty_forest(num_rules, graph_id))
                fail_count += 1
            finally:
                # Without this the C++ chart of every graph leaks.
                context.release_memory()
    finally:
        if use_timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if verbose:
        if cache is not None:
            logger.info(f"Cache: {cache_hits} hits, {parse_count} parses, {timeout_count} timeouts")
        else:
            logger.info(f"Parsed {parse_count} graphs, {fail_count} failures, {timeout_count} timeouts")

    if cache is not None and batch_cache and parse_count > 0:
        cache.save_batch(dict(zip(graph_ids, forests)))

    return forests, graph_ids, sentences, num_rules, cache


def load_forests_with_cache_legacy(cache, context, manager, num_rules, verbose=True):
    from .export import export_forest

    forests = []
    cache_hits = parse_count = 0
    for i in range(manager.graph_size):
        graph = manager.get_graph(i)
        graph_id = graph.sentence_id

        if cache is not None:
            forest = cache.load_forest(graph_id)
            if forest is not None:
                forests.append(forest)
                cache_hits += 1
                continue

        error = context.parse(graph)
        if error.name != 'kNone':
            if verbose:
                logger.debug(f"Parse failed for graph {i}: {error.name}")
            forests.append(_empty_forest(num_rules, graph_id))
            continue
        try:
            forest = export_forest(context, graph)
            forests.append(forest)
            if cache is not None:
                cache.save_forest(graph_id, forest)
            parse_count += 1
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to export forest for graph {i}: {e}")
            forests.append(_empty_forest(num_rules, graph_id))

    if verbose and cache is not None:
        logger.info(f"Cache: {cache_hits} hits, {parse_count} parses")
    return forests
