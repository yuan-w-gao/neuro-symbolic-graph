#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/basic.hpp"

#define PAD_LABEL "[pad]"
#define NULL_POSTAG '#'
#define NULL_SENSE "#"
#define NULL_LEMMA "#"
#define NULL_CARG "#"

namespace shrg {

class TokenSet {
    friend std::istream &operator>>(std::istream &is, TokenSet &token_set);
    friend std::ostream &operator<<(std::ostream &os, const TokenSet &token_set);

  protected:
    std::unordered_map<std::string, int> token2index_{{PAD_LABEL, 0}};
    std::vector<std::string> tokens_{PAD_LABEL};
    bool is_frozen_ = false;

  public:
    void Freeze() { is_frozen_ = true; }

    void Clear() {
        is_frozen_ = false;
        token2index_ = {{PAD_LABEL, 0}};
        tokens_ = {PAD_LABEL};
    }

    int Count() const { return tokens_.size(); }

    int Get(const std::string &token, int default_value = -1) const {
        auto it = token2index_.find(token);
        return it == token2index_.end() ? default_value : it->second;
    }

    int Index(const std::string &token) {
        auto it = token2index_.find(token);
        if (it == token2index_.end()) {
            //ASSERT_ERROR_RETURN(!is_frozen_, "TokenSet is frozen !!!", -1);
            int index = token2index_[token] = tokens_.size();
            tokens_.push_back(token);
            return index;
        }
        return it->second;
    }

    const std::string &operator[](int index) const {
        static std::string unk("[unk]");
        return index != -1 ? tokens_[index] : unk;
    }

    bool Save(const std::string &filename);
    bool Load(const std::string &filename);
};

} // namespace shrg

// Local Variables:
// mode: c++
//  End:
