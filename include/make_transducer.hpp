//
// Created by Dekai WU and YAN Yuchen on 20190402.
//

#ifndef SRC_MAKE_TRANSDUCER_HPP
#define SRC_MAKE_TRANSDUCER_HPP
#include <functional>
#include <memory>
#include <variant>
#include <cereal/cereal.hpp>
using namespace std;
namespace tg {
  using symbol_t = string;
  using token_t = symbol_t;
  using postag_t = symbol_t;
  using scalar_t = double;
  using sentence_t = vector<token_t>;

  class dim_t {
  public:
    std::vector<unsigned> ds;
    unsigned batch{1};

    template<typename Archive>
    void serialize(Archive &ar) {
      ar(cereal::make_nvp("ds", ds));
      ar(cereal::make_nvp("batch", batch));
    }

    dim_t() = default;
    dim_t(const dim_t&) = default;
    dim_t(dim_t&&) noexcept = default;
    dim_t &operator=(const dim_t&) = default;
    dim_t &operator=(dim_t&&) noexcept = default;
    dim_t(std::initializer_list<unsigned int> ds): ds(ds) {}
    unsigned operator[](unsigned index) const {return ds[index];}

  };

  class tensor_t {
  public:
    dim_t dim;
    std::vector<float> data;

    template<typename Archive>
    void serialize(Archive &ar) {
      bool valid = true;
      ar(cereal::make_nvp("valid", valid));
      if(!valid) return;
      ar(cereal::make_nvp("dim", dim));
      ar(cereal::make_nvp("data", data));
    }

    tensor_t() = default;
    tensor_t(const tensor_t&) = default;
    tensor_t(tensor_t&&) noexcept = default;
    tensor_t &operator=(const tensor_t&) = default;
    tensor_t &operator=(tensor_t&&) noexcept = default;
    tensor_t(std::initializer_list<float> values): dim({(unsigned)values.size()}), data(values) {}
    explicit tensor_t(std::vector<float> values): dim({(unsigned)values.size()}), data(move(values)) {}

  };

  using feature_t = std::variant<monostate, tensor_t, string, scalar_t, std::vector<scalar_t>>;

  class object_id {
  public:
    string id;

    template<typename Archive>
    void serialize(Archive &ar) {
      ar(cereal::make_nvp("objectid", id));
    }

    object_id() = default;
    object_id(const object_id&) = default;
    object_id(object_id&&) noexcept = default;
    object_id &operator=(const object_id&) = default;
    object_id &operator=(object_id&&) noexcept = default;
    explicit object_id(string id):id(move(id)) {}

  };

  /**
   * a transducer
   */
  class transducer_t {
  public:
    object_id id_m;

    transducer_t() = default;
    transducer_t(const transducer_t&) = default;
    transducer_t(transducer_t&&) noexcept = default;
    transducer_t &operator=(const transducer_t&) = default;
    transducer_t &operator=(transducer_t&&) noexcept = default;

    explicit transducer_t(const object_id &id);

    inline vector<feature_t> operator()(const vector<feature_t> &x) const {
      return transduce(x);
    }

    vector<feature_t> transduce(const vector<feature_t> &x) const;

    vector<vector<feature_t>> transduce_many(const vector<vector<feature_t>> &xs) const;

    unsigned num_inputs() const;

    unsigned num_outputs() const;

    void
    train(const vector<vector<feature_t>> &examples, const vector<vector<feature_t>> &oracles, unsigned num_epochs);

    void
    train(const vector<vector<feature_t>> &examples, const vector<feature_t> &oracles, unsigned num_epochs);

    void
    train(const vector<feature_t> &examples, const vector<vector<feature_t>> &oracles, unsigned num_epochs);

    void
    train(const vector<feature_t> &examples, const vector<feature_t> &oracles, unsigned num_epochs);

    void save(std::ostream &os) const;

    void load(std::istream &is);
  };

  /**
   * create an identity transducer
   * identity transducer let its input passes through
   * it takes 1 input of any type and returns exactly that input
   * \return an identity transducer
   */
  transducer_t make_identity();

  /**
   * create an identity transducer that takes multiple inputs
   * this transducer takes multiple inputs, and returns exactly those inputs
   * this is a more efficient equivalent to group(make_identity(), make_identity(), ...)
   * \param num the number of inputs
   * \return an identity transducer that takes multiple inputs
   */
  transducer_t make_identities(unsigned num);

  /**
   * create a pick transducer that picks (and re-order) its inputs
   * \param num_inputs number of inputs
   * \param input_positions each element of this vector represents the position of inputs to pick.
   * the size of this vector equals the number of outputs
   * for example, make_pick(2, {1, 0}) takes 2 inputs and swap them.
   * make_pick(3, {2, 1}) takes 3 inputs, discard the first one and swap the other two.
   * \return the pick transducer
   */
  transducer_t make_pick(unsigned num_inputs, const vector<unsigned> &input_positions);

  /**
   * create a pick transducer that picks (and re-order) its inputs
   * this overload automatically infers the num_inputs, by looking at the maximum value of input_positions
   */
  transducer_t make_pick(const vector<unsigned> &input_positions);

  /**
   * create a transducer that always outputs a constant value
   * it takes 0 inputs, and returns one output (equal to the provided value)
   * \param value the constant value to output
   * \return the transducer that outputs the provided value
   */
  transducer_t make_const(const feature_t &value);

  /**
   * create a copy transducer
   * a copy transducer make multiple copies of the input
   * it takes 1 input of any type and returns multiple copies of the that input
   * \param num_copies number of copies to make
   * \return
   */
  transducer_t make_copy(unsigned num_copies);

  /**
   * compose two transducers into one transducer, by sequentially applying them
   * the output of the first transducer must match the input of the second transducer
   * \param f0 the first transducer
   * \param f1 the second transducer
   * \return
   */
  transducer_t compose(transducer_t f0, transducer_t f1);

  /**
   * group two transducers input one transducer, by applying them in parallel
   * (# of input of the grouped transducer) = (# of input of the first transducer) + (# of input of the second transducer)
   * (# of output of the grouped transducer) = (# of output of the first transducer) + (# of output of the second transducer)
   * \param f0 the first transducer
   * \param f1 the second transducer
   * \return
   */
  transducer_t group(transducer_t f0, transducer_t f1);

  /**
   * a KNN classifier as introduced in the class
   * this transducer takes some symbolic features and predicts a symbolic class
   * \param k the number of nearest neighbors to look at
   * \param num_inputs number of input symbolic features
   * \param output_classes the list of all possible classes as classification output
   * \return
   */
  transducer_t make_symbolic_k_nearest_neighbors_classifier(unsigned k, unsigned num_inputs, const vector<symbol_t> &output_classes);

  /**
   * a onehot transducer converts a token into onehot representation
   * it takes a token, and outputs a rank 1 tensor whose dimension is (vocab size + 1)
   * +1 because the UNK token will be automatically inserted into the vocabulary.
   * what is UNK? if the token to convert is outside of the vocabulary, this token will be treated as UNK.
   * \param vocab the list of vocabulary
   * \return a onehot transducer
   */
  transducer_t make_onehot(const vector<symbol_t> &vocab);


  /**
   * an embedding lookup transducer converts a token into rank 1 tensor
   * this is done by keeping a lookup table. For every possible token (including UNK), the table records its corresponding tensor.
   * note that this transducer is mathematically equivalent to compose(make_onehot(vocab), dense(embedding_size, make_identity()))
   * \param embedding_size the dimension of the output tensor
   * \param vocab the list of vocabulary
   * \return an embedding transducer
   */
  transducer_t make_embedding_lookup(unsigned embedding_size, const vector<symbol_t> &vocab);

  /**
   * a onehot inverse transducer is the inverse transducer of onehot
   * it takes a rank 1 onehot tensor, and returns a token
   * actually the input tensor doesn't have to be exactly onehot tensor, but it must be a probability distribution.
   * (note that onehot tensor is also a probability distribution)
   * \param vocab the list of vocabulary
   * \return a onehot inverse transducer
   */
  transducer_t make_onehot_inverse(const vector<symbol_t> &vocab);

  /**
   * a tanh transducer can perform tanh operation
   * it takes a rank 1 tensor, and returns a tensor of the same dimension, with each element applying tanh operation
   * \return a tanh transducer
   */
  transducer_t make_tanh();

  /**
   * a softmax transducer can perform softmax operation
   * it takes a rank 1 tensor, and apply softmax function. It returns a tensor of the same dimension.
   * softmax function will always return a probability distribution
   * you can check softmax function on wikipedia
   * \return a softmax transducer
   */
  transducer_t make_softmax();

  /**
   * a log softmax transducer performs softmax operation followed by log operation
   * it takes a rank 1 tensor, and apply softmax function and log function. It returns a tensor of the same dimension.
   * this combined version is numerically stable (doesn't cause NaN) compared with applying softmax and log separately
   * \return a log softmax transducer
   */
  transducer_t make_log_softmax();

  /**
   * a concatenate transducer takes multiple rank 1 tensors, and concatenate them into a single rank 1 tensor
   * \param num_inputs number of tensors to concatenate
   * \return a concatenate transducer
   */
  transducer_t make_concatenate(unsigned num_inputs);

  /**
   * a standard dense feedforward layer
   * it takes a rank 1 tensor x, performs f(Wx+b). where f is the activation function
   * \param output_dim the desired output dimension
   * \param activation the activation function to apply
   * \return
   */
  transducer_t make_dense_feedfwd(unsigned output_dim, transducer_t activation);

  /**
   * compute Euclidean distance between two rank 1 tensors
   * it takes two rank 1 tensors of the same dimension, and returns a rank 1 tensor with 1 dimension
   * \return a Euclidean distance transducer
   */
  transducer_t make_l2_distance();

  /**
   * compute the dot product of two rank 1 tensors
   * it takes two rank 1 tensors of the same dimension, and returns a rank 1 tensor with 1 dimension
   * \return a dot product transducer
   */
  transducer_t make_dot_product();


  // you don't need this transducer for assignment
  transducer_t make_readout_recognizer(const vector<symbol_t> &vocab);

  /**
   * multiply two tensors together
   * it takes two tensors and returns their product
   * if one of the tensor is rank 2, it will perform a matrix multiplication
   * \return a tensor multiplication transducer
   */
  transducer_t make_tensor_mul();

  /**
   * add multiple tensors together
   * it takes multiple tensors and returns their sum
   * \param num_inputs number of tensors to sum
   * \return a tensor add transducer
   */
  transducer_t make_tensor_add(unsigned num_inputs = 2);

  /**
   * take the negative of a tensor
   * it takes a tensor and returns its negative tensor
   * \return the negative tensor
   */
  transducer_t make_tensor_neg();

  /**
   * apply the sigmoid function
   * it takes a rank 1 tensor and returns a tensor of the same dimension, with each element applying sigmoid
   * \return
   */
  transducer_t make_sigmoid();

  /**
   * multiple two tensors element-wise
   * it takes 2 tensors and returns their element-wise multiplication
   * the 2 tensors must be of same dimension
   * \return the element-wise multiplication transducer
   */
  transducer_t make_tensor_cmult();

  template<typename... Args>
  transducer_t compose(transducer_t f0, transducer_t f1, Args... fs) {
    return compose(f0, compose(f1, fs...));
  }

  template<typename... Args>
  transducer_t group(transducer_t f0, transducer_t f1, Args... fs) {
    return group(f0, group(f1, fs...));
  }

  void send_get_request(const std::string& path);
}

#endif //SRC_MAKE_TRANSDUCER_HPP
