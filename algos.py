import collections
import functools
import itertools
import operator

import queue
import heapq
import re


# basic algo stuff

class DisjointSet:
    def __init__(self, n):
        self.ns = list(range(n))
        self.sz = [1] * n

    def find(self, k):
        while self.ns[k] != k:
            self.ns[k] = self.ns[self.ns[k]]
            k = self.ns[k]
        return k

    def union(self, j, k):
        if self.sz[j] < self.sz[k]:
            self.ns[j] = self.ns[k]
            self.sz[k] += self.sz[j]
        else:
            self.ns[k] = self.ns[j]
            self.sz[j] += self.sz[k]

# graph traversal

def dfs(sf, n):
    visited = set([n])
    stack = collections.deque(sf(n))
    while stack:
        cn = stack.pop()
        if cn in visited:
            continue
        visited.add(cn)
        stack.extend(sf(cn))
        yield cn


def bfs(sf, n):
    visited = set([n])
    queue = collections.deque(sf(n))
    while queue:
        cn = queue.popleft()
        if cn in visited:
            continue
        visited.add(cn)
        queue.extend(sf(cn))
        yield cn

