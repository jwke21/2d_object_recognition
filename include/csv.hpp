/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 *
 * These utility functions are derived from Bruce Maxwell's csv_util.h
 */

#include <string>
#include <vector>

#ifndef CSV_H
#define CSV_H

namespace csv {

/**
 * Writes the given row to the given file. Will overwrite the file if
 * reset_file is true.
 * 
 * @param filename The name of the file to write to.
 * @param row The row to write to the file.
 * @param reset_file Whether or not to overwrite the file.
 * 
 * @returns 0 on success, -1 on failure.
 */
int write_row(const std::string filename, const std::vector<std::string> row,
              const bool reset_file = false);

/**
 * Writes the given rows to the given file. Will overwrite the file if
 * reset_file is true.
 *
 * @param filename The name of the file to write to.
 * @param rows The rows to write to the file.
 * @param reset_file Whether or not to overwrite the file.
 *
 * @returns 0 on success, -1 on failure.
 */
int write_rows(const std::string filename,
               const std::vector<std::vector<std::string>> rows,
               const bool reset_file = false);

/**
 * Reads all rows from the given file into the given vector.
 *
 * @param filename The name of the file to read from.
 * @param dst The vector to read the rows into.
 * @param echo_file Whether or not to echo the file to stdout.
 *
 * @returns 0 on success, -1 on failure.
 */
int read_all_rows(const std::string filename,
                  std::vector<std::vector<std::string>>& dst,
                  const bool echo_file = false);

}  // namespace csv

#endif  // CSV_H