//
// Created by Dekai WU and YAN Yuchen on 20190424.
//
#include <shared_utils.hpp>
#include <make_transducer.hpp>
#include "assignment.hpp"
#include <numeric>
#include <unordered_map>
#include "../your_itsc.hpp"
using namespace std;
using namespace tg;

int main() {
  if(your_itsc().empty()) {throw std::runtime_error("please fill in your ITSC");}
  try{tg::send_get_request("start/" + your_itsc());}
  catch(...){}
  const unsigned MAX_SENTENCE_LENGTH = 16;

  // read training data
  auto training_data = read_dataset("/project/cl/httpd/htdocs/COMP3211_2019Q1_a3/res/traindata.xml");

  // filter out data that is too long
  for(auto itr = training_data.begin(); itr != training_data.end(); ) {
    if(itr->empty()) {
      itr = training_data.erase(itr);
    }
    else if(itr->size() > MAX_SENTENCE_LENGTH) {
      itr = training_data.erase(itr);
    }
    else {
      ++itr;
    }
  }


  // get the RNN
  part_b::init(training_data);

  // organize all possible sentences by their lengths
  unordered_map<unsigned, vector<vector<token_t>>> training_data_group_by_length;
  for (const auto &training_datum:training_data) {
    training_data_group_by_length[training_datum.size()].push_back(training_datum);
  }

  // create all possible RNNs from every possible lengths
  vector<transducer_t> transducers{transducer_t()};
  transducers.reserve(MAX_SENTENCE_LENGTH);

  for (unsigned i = 1; i <= MAX_SENTENCE_LENGTH; ++i) {
    cout << "constructing RNN of length " << i <<endl;
    transducers.push_back(part_b::make_rnn_recognizer_any_inputs(i));
  }


  // train the RNN
  cout << "training" << endl;
  try{tg::send_get_request(your_itsc());}
  catch(...){}
  for (auto &[sentence_size, partial_training_data]:training_data_group_by_length) {

    transducer_t &rnn = transducers[sentence_size];
    // define oracles. oracles are trivially 0s
    vector<feature_t> training_oracles(partial_training_data.size(), feature_t(0));

    auto partial_training_examples = array_map<vector<string>, vector<feature_t>>(partial_training_data, part_b::get_features);

    rnn.train(partial_training_examples, training_oracles, part_b::NUM_EPOCHS);

  }


  // read test data
  auto dev_data = read_dataset("/project/cl/httpd/htdocs/COMP3211_2019Q1_a3/res/devdata.xml");

  // filter out data that is too long
  for(auto itr = dev_data.begin(); itr != dev_data.end(); ) {
    if(itr->empty()) {
      itr = dev_data.erase(itr);
    }
    else if(itr->size() > MAX_SENTENCE_LENGTH) {
      itr = dev_data.erase(itr);
    }
    else {
      ++itr;
    }
  }


  // test the RNN
  cout << "testing" << endl;
  vector<scalar_t> sentence_scores;
  unsigned accumulated_num_tokens = 0;
  for (const auto &dev_datum:dev_data) {
    accumulated_num_tokens += dev_datum.size();
    auto dev_example = part_b::get_features(dev_datum);
    sentence_scores.push_back(get<vector<scalar_t>>(transducers[dev_datum.size()].transduce(dev_example)[0])[0]);
  }

  // write prediction
  save_prediction(dev_data, sentence_scores, "predict.xml");
  cout << "prediction saved to predict.xml" << endl;

  // calculate score
  auto total_distance = std::accumulate(sentence_scores.begin(), sentence_scores.end(), (double) 0);
  cout << "average distance per token: " << total_distance/accumulated_num_tokens << endl;
}
