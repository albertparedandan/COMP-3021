//
// Created by Dekai WU and YAN Yuchen on 20190424.
//

#include "assignment.hpp"
#include <make_transducer.hpp>
#include <shared_utils.hpp>

using namespace tg;
using namespace std;

const char* STUDENT_ID = "20373300";
namespace part_demo {

    unsigned const NUM_EPOCHS = 4;

    static const unsigned EMBEDDING_SIZE = 32;
    transducer_t embedding_layer;
    transducer_t one_hot;
    transducer_t dense_feedfwd_softmax;
    transducer_t dense_feedfwd_tanh;

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

    /**
     * construct an RNN that will evaluate the degree of goodness of a sentence
     * in this demo part, it evaluate the degree of goodness of the first 3 tokens of the sentence
     * \param training_set the set of all training tokens, will be used to construct a vocabulary
     * \return an RNN model
     */
    transducer_t make_rnn_recognizer_3_inputs() {
      auto initial_state = make_const(vector<scalar_t>(EMBEDDING_SIZE, 0));
      auto start_of_sentence = make_const(token_t("<s>"));

      auto cell = compose(make_concatenate(2), dense_feedfwd_tanh);
      auto copy = make_copy(2);

      auto input_pick = make_pick({0, 1, 0, 1, 2});

      auto timestep_0 = compose(group(initial_state, compose(start_of_sentence, embedding_layer)), cell, copy);
      auto timestep_0_to_1 = compose(group(timestep_0, embedding_layer), group(make_identity(), compose(cell, copy)));
      auto timestep_0_to_2 = compose(group(timestep_0_to_1, embedding_layer), group(make_identities(2), cell));

      auto output_pick = make_pick({0, 3, 1, 4, 2, 5});

      auto readout_loss = compose(
              group(compose(dense_feedfwd_softmax, make_tensor_neg()),
                    one_hot), make_dot_product());

      return compose(input_pick, group(timestep_0_to_2, make_identities(3)), output_pick,
                     group(readout_loss, readout_loss, readout_loss), make_tensor_add(3));
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
      // because the RNN expects 3 tokens
      if(sentence.size() == 3) {
        return vector<feature_t>{sentence[0], sentence[1], sentence[2]};
      }
      else {
        throw std::runtime_error("sentences should be in length 3");
      }
    }

}
