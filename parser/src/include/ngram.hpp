#pragma once

#include <array>
#include <iostream>

#include <boost/functional/hash.hpp>

namespace utils {

template <typename KEY_TYPE, int N> class NGram {
    static_assert(N > 0, "N must be greater than 0");

    friend struct std::hash<NGram<KEY_TYPE, N>>;

  private:
    std::array<KEY_TYPE, N> grams_;
    std::size_t hash_;

    void ComputeHash() {
        hash_ = 0;
        for (auto i = 0; i < N; ++i)
            boost::hash_combine(hash_, grams_[i]);
    }

  public:
    NGram() : grams_{}, hash_(0) {}

    // template < typename... Args >
    // NGram(Args&&... args)
    // : grams_{KEY_TYPE(std::forward< Args >(args))...} {
    //     static_assert(sizeof...(Args) == N,
    //                   "Invalid number of constructor arguments.");
    //     ComputeHash();
    // }

    NGram(const NGram &ngram) : grams_(ngram.grams_), hash_(ngram.hash_) {}

    NGram &operator=(const NGram &ngram) {
        grams_ = ngram.grams_;
        ComputeHash();
    }

    const KEY_TYPE &operator[](int index) const { return grams_[index]; }

    void SetByIndex(int index, const KEY_TYPE &gram) {
        grams_[index] = gram;
        ComputeHash();
    }

    template <typename... Args> void Set(Args &&... args) {
        static_assert(sizeof...(Args) == N, "Invalid number of Set arguments.");
        grams_ = {KEY_TYPE(std::forward<Args>(args))...};
        ComputeHash();
    }

    bool operator==(const NGram &ngram) const {
        return ngram.hash_ == hash_ && ngram.grams_ == grams_;
    }

    friend std::istream &operator>>(std::istream &is, NGram &ngram) {
        char c;
        is >> c; // '[';
        for (int i = 0; i < N; ++i)
            is >> ngram.grams_[i];
        is >> c; // ']'
        ngram.ComputeHash();
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const NGram &ngram) {
        os << "[ ";
        for (int i = 0; i < N; ++i)
            os << ngram.grams_[i] << " ";
        return os << "]";
    }
};

template <typename LABEL, typename HASH> class LabelSet {
    friend struct std::hash<utils::LabelSet<LABEL, HASH>>;

  private:
    HASH hash_;

  public:
    void Add(const LABEL &label) { hash_ |= (HASH)(1 << label); }

    void Remove(const LABEL &label) { hash_ &= ~(HASH)(1 << label); }

    bool Contains(const LABEL &label) const { return hash_ & (HASH)(1 << label); }

    const HASH &Hash() const { return hash_; }

    bool operator==(const LabelSet &label_set) { return label_set.hash_ == hash_; }
};

} // namespace utils

namespace std {

template <typename KEY_TYPE, int N> struct hash<utils::NGram<KEY_TYPE, N>> {
    using result_type = size_t;
    using argument_type = utils::NGram<KEY_TYPE, N>;
    result_type operator()(const argument_type &ngram) const { return ngram.hash_; }
};

template <typename LABEL, typename HASH> struct hash<utils::LabelSet<LABEL, HASH>> {
    using result_type = size_t;
    using argument_type = utils::LabelSet<LABEL, HASH>;
    result_type operator()(const argument_type &label_set) const { return label_set.hash_; }
};

} // namespace std

// Local Variables:
// mode: c++
// End:
