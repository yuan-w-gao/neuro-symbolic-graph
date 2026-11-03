#include <boost/algorithm/string.hpp>
#include <cmath>
#include <iomanip>

#include "bleu.hpp"

#define LOG(x) ((x) == 0 ? INT64_MIN : std::log((x)))

#define SPLIT(result, text)                                                                        \
    {                                                                                              \
        (result).clear();                                                                          \
        boost::split((result), (text), [](char c) { return std::isspace(c); },                     \
                     boost::token_compress_on);                                                    \
    }

#define COLLECT(n, type)                                                                           \
    {                                                                                              \
        ngrams_map_.clear();                                                                       \
        for (int start = 0; start < type##_length_ - (n); ++start) {                               \
            ngram_.assign(1, (n) + '0');                                                           \
            for (int w = 0; w <= (n); ++w) {                                                       \
                ngram_.push_back(' ');                                                             \
                ngram_.append(type##_words_[start + w]);                                           \
            }                                                                                      \
            ++ngrams_map_[ngram_];                                                                 \
        }                                                                                          \
    }

namespace utils {

std::string MultiBLEU::ToString() {
    if (total_reference_length_ == 0)
        return "BLEU = 0, 0/0/0/0 (BP=0, ratio=0, hyp_len=0, ref_len=0)\n";

    double brevity_penalty = 1.0;
    double bleu_scores[4]{0, 0, 0, 0};
    double bleu = Value(brevity_penalty, bleu_scores);

    std::string buffer(200, ' ');
    int size = std::snprintf(&buffer[0], buffer.size(),
                             "BLEU = %.2f, %.1f/%.1f/%.1f/%.1f"
                             " (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)",
                             100 * bleu, 100 * bleu_scores[0], 100 * bleu_scores[1],
                             100 * bleu_scores[2], 100 * bleu_scores[3], brevity_penalty,
                             1.0 * total_sentence_length_ / total_reference_length_,
                             total_sentence_length_, total_reference_length_);
    buffer.erase(size, buffer.size() - size);
    return buffer;
}

double MultiBLEU::Value() {
    double brevity_penalty = 1.0;
    double bleu_scores[4]{0, 0, 0, 0};

    return Value(brevity_penalty, bleu_scores);
}

double MultiBLEU::Value(double &brevity_penalty, double bleu_scores[]) {
    for (int n = 0; n < 4; ++n)
        bleu_scores[n] = total_[n] == 0 ? 0 : 1.0 * correct_[n] / total_[n];

    if (total_sentence_length_ < total_reference_length_)
        brevity_penalty = std::exp(1 - 1.0 * total_reference_length_ / total_sentence_length_);

    double bleu = brevity_penalty * std::exp((LOG(bleu_scores[0]) + LOG(bleu_scores[1]) +
                                              LOG(bleu_scores[2]) + LOG(bleu_scores[3])) /
                                             4);
    return bleu;
}

void MultiBLEU::Collect(const std::string &reference) {
    ngram_ = boost::trim_copy(reference);
    SPLIT(reference_words_, ngram_);

    reference_length_ = reference_words_.size();

    int diff = std::abs(reference_length_ - sentence_length_);

    if (diff < closest_diff_) {
        closest_diff_ = diff;
        closest_length_ = reference_length_;
    } else if (diff == closest_diff_ && reference_length_ < closest_length_)
        closest_length_ = reference_length_;

    for (int n = 0; n < 4; ++n) {
        COLLECT(n, reference);

        for (auto &item : ngrams_map_) {
            auto it = reference_ngrams_map_.find(item.first);
            if (it == reference_ngrams_map_.end())
                reference_ngrams_map_[item.first] = item.second;
            else if (it->second < item.second)
                it->second = item.second;
        }
    }
}

void MultiBLEU::Compute() {
    total_sentence_length_ += sentence_length_;
    total_reference_length_ += closest_length_;
    for (int n = 0; n < 4; ++n) {
        COLLECT(n, sentence);

        for (auto &item : ngrams_map_) {
            int n = item.first[0] - '0';
            total_[n] += item.second;

            auto it = reference_ngrams_map_.find(item.first);
            if (it != reference_ngrams_map_.end())
                correct_[n] += std::min(it->second, item.second);
        }
    }
}

void MultiBLEU::Add(const std::string &sentence, const std::string &reference) {
    reference_ngrams_map_.clear();

    ngram_ = boost::trim_copy(sentence);
    SPLIT(sentence_words_, ngram_);

    closest_diff_ = 9999;
    closest_length_ = 9999;
    sentence_length_ = sentence_words_.size();

    Collect(reference);

    Compute();
}

void MultiBLEU::Add(const std::string &sentence, const std::vector<std::string> &references) {
    reference_ngrams_map_.clear();

    ngram_ = boost::trim_copy(sentence);
    SPLIT(sentence_words_, ngram_);

    closest_diff_ = 9999;
    closest_length_ = 9999;
    sentence_length_ = sentence_words_.size();

    for (auto &reference : references)
        Collect(reference);

    Compute();
}

} // namespace utils
