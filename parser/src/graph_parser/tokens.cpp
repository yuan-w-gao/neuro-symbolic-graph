#include <fstream>

#include "tokens.hpp"

namespace shrg {

std::ostream &operator<<(std::ostream &os, const TokenSet &token_set) {
    ASSERT_ERROR_RETURN(token_set.is_frozen_, "TokenSet should be frozen before saving !!!", os);
    os << token_set.tokens_.size();
    for (auto &token : token_set.tokens_)
        os << token << '\n';
    return os;
}

std::istream &operator>>(std::istream &is, TokenSet &token_set) {
    ASSERT_ERROR_RETURN(!token_set.is_frozen_, "TokenSet is frozen !!!", is);
    int size;
    is >> size;
    token_set.tokens_.resize(size);
    for (int i = 0; i < size; ++i) {
        auto &token = token_set.tokens_[i];
        is >> token;
        token_set.token2index_[token] = i;
    }
    token_set.is_frozen_ = true;
    return is;
}

bool TokenSet::Save(const std::string &filename) {
    OPEN_OFSTREAM(os, filename, return false);

    LOG_INFO("Save Tokens ... ");
    os << *this;
    LOG_INFO("Save Tokens done");

    CLOSE_STREAM(os);
    return true;
}

bool TokenSet::Load(const std::string &filename) {
    OPEN_IFSTREAM(is, filename, return false);

    LOG_INFO("Load Tokens ... ");
    is >> *this;
    LOG_INFO("Tokens: " << Count());

    CLOSE_STREAM(is);
    return true;
}

} // namespace shrg
