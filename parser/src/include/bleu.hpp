#pragma once

#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils {

class MultiBLEU {
  private:
    int total_sentence_length_;
    int total_reference_length_;

    int reference_length_;
    int sentence_length_;
    int closest_length_;
    int closest_diff_;

    std::array<int, 4> total_;
    std::array<int, 4> correct_;

    std::vector<std::string> sentence_words_;
    std::vector<std::string> reference_words_;

    std::unordered_map<std::string, int> reference_ngrams_map_;
    std::unordered_map<std::string, int> ngrams_map_;
    std::string ngram_;

  private:
    void Collect(const std::string &reference);
    void Compute();

    double Value(double &brevity_penalty, double bleu_scores[]);

  public:
    MultiBLEU() { Clear(); }

    void Add(const std::string &sentence, const std::vector<std::string> &references);

    void Add(const std::string &sentence, const std::string &reference);

    void Clear() {
        total_reference_length_ = 0;
        total_sentence_length_ = 0;
        total_ = {0, 0, 0, 0};
        correct_ = {0, 0, 0, 0};
    }

    std::string ToString();

    double Value();
};

} // namespace utils

// Local Variables:
// mode: c++
// End:
