/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 */

#include "disjoint_set.hpp"

DisjointSet::DisjointSet(int n) {
  // https://stackoverflow.com/questions/2204176/how-to-initialise-memory-with-new-operator-in-c
  parent = new int[n]();
  rank = new int[n]();
  this->n = n;
}

int DisjointSet::find(int x) {
  if (parent[x] != x) {
    parent[x] = find(parent[x]);
  }

  return parent[x];
}

void DisjointSet::make_union(int x, int y) {
  int xroot = find(x);
  int yroot = find(y);

  if (xroot == yroot) {
    return;
  }

  if (rank[xroot] > rank[yroot]) {
    parent[yroot] = xroot;
  } else {
    parent[xroot] = yroot;
    if (rank[xroot] == rank[yroot]) {
      rank[yroot]++;
    }
  }
}