//
// Created by Dekai WU and YAN Yuchen on 20190424.
//
#include <shared_utils.hpp>
#include <make_transducer.hpp>
#include "assignment.hpp"
#include <numeric>
#include "../your_itsc.hpp"

using namespace std;
using namespace tg;

int main() {
  if(your_itsc().empty()) {throw std::runtime_error("please fill in your ITSC");}
  try{tg::send_get_request("start/" + your_itsc());}
  catch(...){}
  // read training data
  auto training_data = read_dataset("/project/cl/httpd/htdocs/COMP3211_2019Q1_a3/res/traindata_length_3.xml");

  // call the get feature function to transform the data into desired features
  auto training_examples = array_map<vector<string>, vector<feature_t>>(training_data, part_demo::get_features);

  // define oracles. oracles are trivially 0s
  vector<feature_t> training_oracles(training_examples.size(), feature_t(0));

  // get the RNN
  part_demo::init(training_data);
  auto rnn = part_demo::make_rnn_recognizer_3_inputs();

  // train the RNN
  cout << "training" << endl;
  try{tg::send_get_request(your_itsc());}
  catch(...){}
  rnn.train(training_examples, training_oracles, part_demo::NUM_EPOCHS);

  // read test data
  auto dev_data = read_dataset("/project/cl/httpd/htdocs/COMP3211_2019Q1_a3/res/devdata_length_3.xml");

  // preprocess the test data
  auto dev_examples = array_map<vector<string>, vector<feature_t>>(dev_data, part_demo::get_features);

  // use the model to score the sentence
  auto sentence_scores = array_map<vector<feature_t>, scalar_t>(rnn.transduce_many(dev_examples),
                                                                [](const vector<feature_t> &fs) {
                                                                  return get<vector<scalar_t>>(fs[0])[0];
                                                                });

  // write prediction
  save_prediction(dev_data, sentence_scores, "predict.xml");
  cout << "prediction saved to predict.xml" << endl;

  // calculate score
  auto total_distance = std::accumulate(sentence_scores.begin(), sentence_scores.end(), (double) 0);
  cout << "average distance per token: " << total_distance/(dev_data.size()*3) << endl;
}
