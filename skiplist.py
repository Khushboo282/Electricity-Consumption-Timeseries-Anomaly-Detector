"""
skiplist.py -- A Skip List implemented from scratch (no external libraries).

WHAT IS A SKIP LIST?
--------------------
A skip list is a sorted linked list with extra "express lanes" on top.

    Level 3:  HEAD ---------------------------> 40 ------------------> None
    Level 2:  HEAD ---------> 10 -------------> 40 ------> 70 -------> None
    Level 1:  HEAD --> 5 ---> 10 --> 25 ------> 40 --> 55 > 70 --> 90 > None
    (Level 1 is the bottom level and contains EVERY key, in sorted order.)

To find a key we start at the top-left and move right while the next key is
still smaller than the target; when we cannot go right any more we drop one
level down. This skips most nodes, so a search takes O(log n) steps on average
instead of the O(n) steps a plain linked list or a full scan would need.

How tall each node's "tower" is gets decided by a coin flip when the node is
inserted (it gets one more level with probability p, default 0.5). This
randomness keeps the structure balanced on average without any rebalancing
code, which is the main appeal of skip lists compared with balanced trees.

Reference: W. Pugh, "Skip Lists: A Probabilistic Alternative to Balanced
Trees", Communications of the ACM 33(6), 1990.

COMPLEXITY (n = number of stored keys, k = number of keys returned)
------------------------------------------------------------------
    insert        O(log n) expected
    search / get  O(log n) expected
    range_query   O(log n + k) expected   <- find the start, then walk right
    memory        O(n) expected (about 1 / (1 - p) = 2 pointers per node)

The worst case is O(n) (all coin flips unlucky) but its probability is
negligible for realistic n.

KEYS
----
Keys can be any mutually comparable Python objects. In this project the key is
a tuple ``(building_id, epoch_seconds)``; Python compares tuples element by
element, so all readings of one building sit next to each other in time order.
"""

from __future__ import annotations

import random
from typing import Any, Iterator, List, Optional, Tuple


class _Node:
    """One node of the skip list.

    Attributes:
        key:     the sort key (``None`` only for the head sentinel).
        value:   the payload stored with the key.
        forward: list of "next" pointers; ``forward[i]`` is the next node on
                 level ``i`` (level 0 is the bottom level holding every key).
                 The length of this list is the height of the node's tower.
    """

    __slots__ = ("key", "value", "forward")

    def __init__(self, key: Any, value: Any, height: int) -> None:
        self.key = key
        self.value = value
        self.forward: List[Optional["_Node"]] = [None] * height


class SkipList:
    """A sorted map from keys to values with O(log n) search and range query.

    Args:
        max_level: maximum tower height. Rule of thumb: about log2(n). The
            default of 20 comfortably covers over a million keys.
        p: probability that a node's tower grows one more level (0 < p < 1).
            0.5 is the textbook choice.
        seed: seed for the internal random generator. Pass an integer to make
            the structure (and therefore any benchmark) reproducible.

    Example:
        >>> sl = SkipList(seed=1)
        >>> for k in (30, 10, 20):
        ...     _ = sl.insert(k, str(k))
        >>> sl.range_query(10, 20)
        [(10, '10'), (20, '20')]
    """

    def __init__(self, max_level: int = 20, p: float = 0.5,
                 seed: Optional[int] = None) -> None:
        if max_level < 1:
            raise ValueError("max_level must be at least 1")
        if not 0.0 < p < 1.0:
            raise ValueError("p must be strictly between 0 and 1")
        self._max_level = max_level
        self._p = p
        self._rng = random.Random(seed)
        # The head is a sentinel that has a pointer on every possible level.
        self._head = _Node(None, None, max_level)
        self._level = 1   # number of levels currently in use (>= 1)
        self._size = 0    # number of stored keys

    # ------------------------------------------------------------------ #
    # Basic container behaviour
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        node = self._first_at_or_after(key)
        return node is not None and node.key == key

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """Iterate over (key, value) pairs in ascending key order."""
        node = self._head.forward[0]
        while node is not None:
            yield node.key, node.value
            node = node.forward[0]

    @property
    def height(self) -> int:
        """Number of levels currently in use (useful for the report)."""
        return self._level

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _random_level(self) -> int:
        """Flip coins to choose a new node's tower height (geometric dist.)."""
        level = 1
        while level < self._max_level and self._rng.random() < self._p:
            level += 1
        return level

    def _first_at_or_after(self, key: Any) -> Optional[_Node]:
        """Return the first node whose key is >= ``key`` (or None).

        This is THE search routine; ``search`` and ``range_query`` both build
        on it. Start at the top level, move right while the next key is still
        smaller than ``key``, then drop down a level and repeat. On reaching
        level 0, the node to the right is the first one that is not smaller.
        """
        node = self._head
        for level in range(self._level - 1, -1, -1):
            nxt = node.forward[level]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[level]
        return node.forward[0]

    # ------------------------------------------------------------------ #
    # Public operations
    # ------------------------------------------------------------------ #
    def insert(self, key: Any, value: Any) -> bool:
        """Insert ``key`` -> ``value``; if the key exists, replace its value.

        Returns:
            True if a new key was added, False if an existing key was updated.
        """
        # update[i] = the last node on level i that is strictly smaller than
        # key. Those are exactly the nodes whose pointers must be rewired.
        # Levels above the current height default to the head.
        update: List[_Node] = [self._head] * self._max_level
        node = self._head
        for level in range(self._level - 1, -1, -1):
            nxt = node.forward[level]
            while nxt is not None and nxt.key < key:
                node = nxt
                nxt = node.forward[level]
            update[level] = node

        candidate = node.forward[0]
        if candidate is not None and candidate.key == key:
            candidate.value = value          # duplicate key: overwrite
            return False

        height = self._random_level()
        if height > self._level:
            self._level = height             # new levels start at the head

        new_node = _Node(key, value, height)
        for level in range(height):          # splice into every level
            new_node.forward[level] = update[level].forward[level]
            update[level].forward[level] = new_node
        self._size += 1
        return True

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value stored under ``key`` or ``default`` if absent."""
        node = self._first_at_or_after(key)
        if node is not None and node.key == key:
            return node.value
        return default

    def search(self, key: Any) -> Tuple[bool, Any]:
        """Look up ``key``. Returns ``(found, value)``; value is None if absent.

        Unlike ``get`` this can tell "absent" apart from "stored value None".
        """
        node = self._first_at_or_after(key)
        if node is not None and node.key == key:
            return True, node.value
        return False, None

    def range_query(self, lo: Any, hi: Any) -> List[Tuple[Any, Any]]:
        """Return all ``(key, value)`` pairs with ``lo <= key <= hi``.

        Both ends are INCLUSIVE and the result is sorted by key. If
        ``lo > hi`` or no key falls in the range, an empty list is returned.

        How it works (O(log n + k)):
          1. Descend the express lanes to the first key >= ``lo``  (O(log n)).
          2. Walk along the bottom level collecting keys until one exceeds
             ``hi``                                                 (O(k)).
        """
        if lo > hi:
            return []
        result: List[Tuple[Any, Any]] = []
        node = self._first_at_or_after(lo)
        while node is not None and node.key <= hi:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    # ------------------------------------------------------------------ #
    # Self-check used by the tests
    # ------------------------------------------------------------------ #
    def check_invariants(self) -> bool:
        """Verify the structural rules of a skip list; raise if any is broken.

        Rules checked:
          * the bottom level holds exactly ``len(self)`` keys, strictly
            increasing (so sorted, with no duplicates);
          * on every higher level keys are strictly increasing, and every key
            there also appears one level below (each level is a "sub-list" of
            the level beneath it);
          * ``height`` is at least the tallest tower actually present.
        """
        previous_level_keys = None
        for level in range(0, self._max_level):
            keys = []
            node = self._head.forward[level]
            while node is not None:
                keys.append(node.key)
                node = node.forward[level]
            for a, b in zip(keys, keys[1:]):
                if not a < b:
                    raise AssertionError(
                        f"level {level} is not strictly increasing: {a!r}, {b!r}")
            if level == 0:
                if len(keys) != self._size:
                    raise AssertionError(
                        f"level 0 has {len(keys)} keys but size is {self._size}")
            else:
                lower = set(previous_level_keys)
                for key in keys:
                    if key not in lower:
                        raise AssertionError(
                            f"key {key!r} on level {level} missing from level {level - 1}")
            if keys and level >= self._level:
                raise AssertionError(
                    f"level {level} is in use but height is {self._level}")
            previous_level_keys = keys
            if not keys:
                break
        return True
