//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_SHARED_UTILS_HPP
#define COMP4221_2019Q1_A3_SHARED_UTILS_HPP
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "xml_archive.hpp"

using namespace std;

class frequent_token_collector {
public:

  /**
   * +1 to a token's occurence
   * \param token
   */
  void add_occurence(const std::string &token) {
    if (counts.count(token) > 0) {
      counts[token]++;
    } else {
      counts[token] = 1;
    }
  }

  /**
   * get the X most frequent tokens.
   * if the tokens seens are less than X, all tokens will be returned.
   * \param size top X
   * \return the list of tokens, ordered from most frequent to most infrequent.
   */
  std::vector<std::string> list_frequent_tokens(unsigned size) const {
    std::vector<std::pair<std::string, unsigned>> pairs;
    for (const auto &p:counts) {
      pairs.emplace_back(p);
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto &x, const auto &y) { return x.second > y.second; });
    if (pairs.size() > size) pairs.resize(size);
    std::vector<std::string> ret;
    ret.reserve(pairs.size());
    for (const auto &p:pairs) {
      ret.push_back(p.first);
    }
    return ret;
  }

  void print_summary() const {
    std::vector<std::pair<std::string, unsigned>> pairs;
    for (const auto &p:counts) {
      pairs.emplace_back(p);
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto &x, const auto &y) { return x.second > y.second; });
    unsigned i=0;
    for(const auto &[token, count]:pairs) {
      cout << i << ". " << token << " "<< count <<endl;
      i++;
    }
  }

private:
  std::unordered_map<std::string, unsigned> counts;
};


/**
 * a helper function to collect vocabulary from training data
 * collect distinct symbols from a list of symbols that may contain duplicates
 * \param symbols a list of symbols
 * \param max_vocab_size size of the vocabulary
 * \return the distinct list
 */
inline vector<string> collect_vocab(const vector<string> &symbols, unsigned max_vocab_size) {
  frequent_token_collector collector;
  for(const auto &symbol:symbols) {
    collector.add_occurence(symbol);
  }
  return collector.list_frequent_tokens(max_vocab_size);
}

/**
 * a helper function to collect vocabulary from training data
 * collect distinct symbols from a matrix of symbols that may contain duplicates
 * \param symbol_matrix a matrix of symbols
 * \param max_vocab_size size of the vocabulary
 * \return the distinct list
 */
inline vector<string> collect_vocab_from_symbol_matrix(const vector<vector<string>> &symbol_matrix, unsigned max_vocab_size) {
  vector<string> symbols;
  for (const auto &row:symbol_matrix) {
    copy(row.begin(), row.end(), back_inserter(symbols));
  }
  return collect_vocab(symbols, max_vocab_size);
}

inline vector<vector<string>> read_dataset(const string &path) {
  ifstream ifs(path);
  if(!ifs.is_open()) throw std::runtime_error("cannot open file " + path);
  cereal::hltc_xml_input_archive oa(ifs);
  vector<vector<string>> ret;
  oa.nest([&]() {
    while(oa.hasNextChild()) {
      oa.nest("sent", [&]() {
        vector<string> sent;
        while(oa.hasNextChild()) {
          string token;
          oa(cereal::make_nvp("token", token));
          sent.push_back(token);
        }
        ret.push_back(sent);
      });
    }
  });
  return ret;
}

inline void save_prediction(const vector<vector<string>> &sentences, const vector<double> &distances, const string &path) {
  if(sentences.size() != distances.size()) throw std::runtime_error("save prediction: sentences and distances should be the same size");
  ofstream ofs(path);
  if(!ofs.is_open()) throw std::runtime_error("cannot open file for output: "+path);
  cereal::hltc_xml_output_archive oa(ofs);
  oa.nest("dataset", [&]() {
    for(unsigned i=0; i<sentences.size(); ++i) {
      const auto &sentence = sentences.at(i);
      const auto &distance = distances.at(i);
      oa.attribute("distance", std::to_string(distance));
      oa.nest("sent", [&]() {
        for(const auto &token:sentence) {
          oa(cereal::make_nvp("token", token));
        }
      });
    }
  });
}

template<typename PARAM_TYPE, typename RET_TYPE>
vector<RET_TYPE> array_map(const vector<PARAM_TYPE> &xs, std::function<RET_TYPE(const PARAM_TYPE&)> fn) {
  vector<RET_TYPE> ret;
  ret.reserve(xs.size());
  for(const auto &x:xs) {
    ret.push_back(fn(x));
  }
  return ret;
}

#endif //COMP4221_2019Q1_A3_SHARED_UTILS_HPP
