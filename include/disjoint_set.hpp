/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 */

#ifndef DISJOINT_SET_HPP
#define DISJOINT_SET_HPP

/**
 * Class for a disjoint set data structure. source:
 * https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
 */
class DisjointSet {
 public:
  /**
   * Constructor for DisjointSet. Given n items, initializes n disjoint sets.
   *
   * @param n the number of items
   */
  DisjointSet(int n);

  /**
   * Finds the set of the given item by following parent pointers until the
   * parent is itself (i.e. the root is found).
   *
   * @param i the item whose set will be found
   *
   * @return the root of the set containing the given item
   */
  int find(int i);

  /**
   * Makes a union of the sets containing the given items. The union is done by
   * rank, i.e. the set with the smaller rank is made a child of the set with
   * the larger rank.
   *
   * @param x the first item
   * @param y the second item
   */
  void make_union(int x, int y);

 private:
  int* parent;
  int* rank;
  int n;
};

#endif  // DISJOINT_SET_HPP