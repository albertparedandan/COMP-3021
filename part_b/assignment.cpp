//
// Created by Dekai WU and YAN Yuchen on 20190424.
//

#include "assignment.hpp"
#include <make_transducer.hpp>
#include <shared_utils.hpp>
using namespace tg;
using namespace std;

const char* STUDENT_ID = "20373300";

namespace part_b {

  unsigned const NUM_EPOCHS = 4;

  static const unsigned EMBEDDING_SIZE = 32;

  // create some custom layers that will be used to constuct your RNN
  // TODO: you can add more custom layers
  transducer_t embedding_layer;
  transducer_t one_hot;
  transducer_t dense_feedfwd_softmax;
  transducer_t dense_feedfwd_tanh;

  // initialize those custom layer
  // TODO: initialize more layers if you have defined them globally
  void init(const vector<sentence_t> &training_set) {
    // first you need to assemble the vocab you need
    // in this simple model, the vocab is the top 1000 most frequent tokens in training set
    // we provide a frequent_token_collector utility,
    // that can count token frequencies and collect the top X most frequent tokens
    // all out-of-vocabulary tokens will be treated as "unknown token"
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    //Here we are going to define our custom layers that we are going to use in each RNN layer
    //Since each RNN hidden layer is identical we only need to define each layer once.
    //You might see that we are not defining the layers dot_product, -log and sum. Because they are already
    //available in our library and doesn't require a customization
    //since all layers are defined here we can make our rnn by declaring topology over the layers
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);
    embedding_layer = make_embedding_lookup(EMBEDDING_SIZE, vocab); //stands for EMB layer in the figure
    one_hot = make_onehot(vocab);  //stands for 1-hot layer in the figure
    dense_feedfwd_softmax = make_dense_feedfwd(vocab.size() + 1, make_log_softmax());// stands for dense 1 log softmax in the figure
    dense_feedfwd_tanh = make_dense_feedfwd(EMBEDDING_SIZE, make_tanh()); // stands for dense 0 tanh
  }

  transducer_t make_timestep_T(transducer_t initial_state, transducer_t start_of_sentence, transducer_t cell, transducer_t copy, unsigned input_length) {
    transducer_t result = compose(group(initial_state, compose(start_of_sentence, embedding_layer)), cell, copy);
    
    if (input_length > 1) {
      for (int i = 1; i < input_length; ++i) {
        if (i + 1 == input_length) {
          result = compose(group(result, embedding_layer), group(make_identities(i), cell));
          continue;
        }
        else if (i == 1) {
          result = compose(group(result, embedding_layer), group(make_identity(), compose(cell, copy)));
          continue;
        }
        else {
          result = compose(group(result, embedding_layer), group(make_identities(i), compose(cell, copy)));
          continue;
        }
      }
      return result;
    }
    else {
      return result;
    }
  }

  transducer_t make_group_readout_loss(unsigned input_length) {
    transducer_t result;
    auto readout_loss = compose(
            group(compose(dense_feedfwd_softmax, make_tensor_neg()),
                  one_hot), make_dot_product());

    if (input_length > 0) {
      result = readout_loss;
    }

    if (input_length > 1) {
      for (int i = 0; i < input_length - 1; ++i) {
          result = group(result, readout_loss);
      }
    }

    return result;
  }

  vector<unsigned> make_input_pick(unsigned input_length) {
    vector<unsigned> result;
    for (int i = 0; i < input_length - 1; ++i) {
      result.emplace_back(i);
    }

    for (int i = 0; i < input_length; ++i) {
      result.emplace_back(i);
    }    
    return result;
  }

  vector<unsigned> make_output_pick(unsigned input_length) {
    vector<unsigned> result;
    
    for (int i = 0; i < input_length; ++i) {
      result.emplace_back(i);
      result.emplace_back(i+input_length);
    }

    return result;
  }

  /**
   * construct an RNN that will evaluate the degree of goodness of a sentence
   * \return an RNN model
   */
  transducer_t make_rnn_recognizer_any_inputs(unsigned input_length) {
    if(input_length==0) throw std::runtime_error("input length cannot be zero");
    // TODO: construct your RNN

    auto initial_state = make_const(vector<scalar_t>(EMBEDDING_SIZE, 0));
    auto start_of_sentence = make_const(token_t("<s>"));

    auto cell = compose(make_concatenate(2), dense_feedfwd_tanh);
    auto copy = make_copy(2);

    // MAKE THE INPUT PICK
    auto input_pick = make_pick(make_input_pick(input_length));
    
    // MAKE OUTPUT PICK
    auto output_pick = make_pick(make_output_pick(input_length));

    // MAKE READOUT LOSS
    auto grouped_readout_loss = make_group_readout_loss(input_length);

    if (input_length == 1) {
      auto timestep_0 = compose(group(initial_state, compose(start_of_sentence, embedding_layer)), cell);
      input_pick = make_pick({0});
      auto output_pick = make_pick({0, 1});
      auto readout_loss = compose(
              group(compose(dense_feedfwd_softmax, make_tensor_neg()),
                    one_hot), make_dot_product());
      return compose(input_pick, group(timestep_0, make_identity()), output_pick,
                     readout_loss);
    }

    // CALL THE RECURSION
    auto timestep_0_to_T = make_timestep_T(initial_state, start_of_sentence, cell, copy, input_length);
    
    // MAKE RETURN
    return compose(input_pick, group(timestep_0_to_T, make_identities(input_length)), output_pick,
                     grouped_readout_loss, make_tensor_add(input_length));
  }

  /**
   * transform a sentence into features that your RNN model expects
   * some pre-processing can be done here, like transforming uppercase into lowercase
   * \param sentence the sentence to transform
   * \return the features that would be feed to your RNN
   */
  vector<feature_t> get_features(const vector<string> &sentence) {
    // in this starting code we didn't do any pre-processing of the sentence
    // just directly passing it to the RNN

    vector<feature_t> ret;
    ret.reserve(sentence.size());
    for(const auto &token:sentence) {
      char test = (char) token[0];
      if (isalnum(test)) {
        ret.emplace_back(token);
      }
      else {
        auto dummy = "<s>";
        ret.emplace_back(dummy);
      }
    }
    return ret;
  }

}
