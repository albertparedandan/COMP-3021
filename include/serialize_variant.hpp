//
// Created by Dekai WU and YAN Yuchen on 20190401.
//

#ifndef XML_SERIALIZE_VARIANT_SERIALIZE_HPP
#define XML_SERIALIZE_VARIANT_SERIALIZE_HPP

#include <variant>
#include <cereal/cereal.hpp>
#include "xml_archive.hpp"

namespace cereal {

  template<int N, class Archive, typename Variant>
  typename std::enable_if<N == std::variant_size_v<Variant>, void>::type
  load_helper(std::size_t type_index, Archive &ar, Variant &x) {
    throw std::runtime_error("Error traversing variant during load");
  }

  template<int N, class Archive, typename Variant>
  typename std::enable_if<N < std::variant_size_v<Variant>, void>::type
  load_helper(std::size_t type_index, Archive &ar, Variant &x) {
    using T = std::variant_alternative_t<N, Variant>;
    if (N == type_index) {
      x = T();
      ar(std::get<T>(x));
    } else {
      load_helper<N + 1, Archive, Variant>(type_index, ar, x);
    }
  }

  template<int N, class Archive, typename Variant>
  void save_helper(std::size_t type_index, Archive &ar, const Variant &x) {
    using T = std::variant_alternative_t<N, Variant>;
    if (N == type_index) {
      ar(std::get<T>(x));
    } else {
      save_helper<N + 1, Archive, Variant>(type_index, ar, x);
    }
  }

  template<class Archive, typename... Ts>
  void load(Archive &ar, std::variant<Ts...> &x) {
    std::size_t type_index;
    ar(cereal::make_nvp("index", type_index));
    load_helper<0, Archive, std::variant<Ts...>>(type_index, ar, x);
  }

  template<typename... Ts>
  void load(hltc_xml_input_archive &ar, std::variant<Ts...> &x) {
    std::size_t type_index{};
    ar.attribute("index", type_index);
    load_helper<0, hltc_xml_input_archive, std::variant<Ts...>>(type_index, ar, x);
  }

  template<class Archive, typename... Ts>
  void save(Archive &ar, const std::variant<Ts...> &x) {
    ar(cereal::make_nvp("index", x.index()));
    auto visitor = [&](const auto& x) {
      ar(x);
    };
    std::visit(visitor, x);
  }

  template<typename... Ts>
  void save(hltc_xml_output_archive &ar, const std::variant<Ts...> &x) {
    ar.attribute("index", std::to_string(x.index()));
    auto visitor = [&](const auto& x) {
      ar(x);
    };
    std::visit(visitor, x);
  }

  template<typename Archive>
  void serialize(Archive &ar, std::monostate &x) {}
}

#endif //XML_SERIALIZE_VARIANT_SERIALIZE_HPP
