//
// Created by Dekai WU and YAN Yuchen on 20190424.
//

#ifndef COMP3221_2019Q1_A3_ASSIGNMENT_HPP
#define COMP3221_2019Q1_A3_ASSIGNMENT_HPP
#include <make_transducer.hpp>
using namespace std;
using namespace tg;

namespace part_a {

  extern unsigned const NUM_EPOCHS;

  void init(const vector<sentence_t> &training_set);

  transducer_t make_rnn_recognizer_4_inputs();

  vector<feature_t> get_features(const vector<string> &sentence);
}
#endif //COMP3221_2019Q1_A3_ASSIGNMENT_HPP
