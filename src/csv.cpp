/**
 * Jake Van Meter
 * Fall 2023
 * CS 5330
 */

#include "csv.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace csv {

int write_row(const std::string filename, const std::vector<std::string> row,
              const bool reset_file) {
    if (filename.empty()) {
        std::cout << "[write_row] Error: filename is empty" << std::endl;
        return -1;
    }

    std::fstream fs;

    // https://cplusplus.com/reference/fstream/fstream/open/
    if (reset_file) {
        // reset the file
        fs.open(filename, std::ios::out | std::ios::trunc);
    } else {
        // append to the file
        fs.open(filename, std::ios::out | std::ios::app);
    }

    if (!fs.is_open()) {
        std::cout << "[write_row] Error: could not open file " << filename
                  << std::endl;
        return -1;
    }

    for (int i = 0; i < row.size() - 1; i++) {
        fs << row[i] << ',';
    }
    
    fs << row.back() + '\n';

    fs.close();

    return 0;
}

int write_rows(const std::string filename,
               const std::vector<std::vector<std::string>> rows,
               const bool reset_file) {
    if (filename.empty()) {
        std::cout << "[write_rows] Error: filename is empty" << std::endl;
        return -1;
    }

    std::fstream fs;

    // https://cplusplus.com/reference/fstream/fstream/open/
    if (reset_file) {
        // reset the file
        fs.open(filename, std::ios::out | std::ios::trunc);
    } else {
        // append to the file
        fs.open(filename, std::ios::out | std::ios::app);
    }

    if (!fs.is_open()) {
        std::cout << "Error: could not open file " << filename << std::endl;
        return -1;
    }

    for (int i = 0; i < rows.size(); i++) {
        for (int j = 0; j < rows[0].size(); j++) {
            fs << rows[i][j] << ',';
        }
        fs << '\n';
    }

    // fs.close();

    std::cout << "Wrote " << rows.size() << " rows to " << filename
              << std::endl;

    return 0;
}

int read_all_rows(const std::string filename,
                  std::vector<std::vector<std::string>>& dst,
                  const bool echo_file) {
    if (filename.empty()) {
        std::cout << "Error: filename is empty" << std::endl;
        return -1;
    }

    std::fstream fs;

    fs.open(filename, std::ios::in);

    if (!fs.is_open()) {
        std::cout << "Error: could not open file " << filename << std::endl;
        return -1;
    }

    // read each row into the dst vector
    while (!fs.eof()) {
        std::string line;
        std::getline(fs, line);

        // skip empty lines (e.g. last row of some csv)
        if (line.empty()) {
            continue;
        }

        std::istringstream line_stream(line);

        // get each column in the row
        std::vector<std::string> row;
        std::string col;

        // read each column into the row vector
        while (std::getline(line_stream, col, ',')) {
            row.push_back(col);
        }

        // add the row to dst vector
        dst.push_back(row);
    }

    if (echo_file) {
        for (const auto& row : dst) {
            for (const auto& col : row) {
                std::cout << col << ' ';
            }
            std::cout << std::endl;
        }
    }

    return 0;
}

}  // namespace csv