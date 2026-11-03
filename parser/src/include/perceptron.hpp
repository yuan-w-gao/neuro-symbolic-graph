#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

namespace perceptron {

using DefaultTScore = std::int64_t;
using DefaultCScore = std::int32_t;

enum ScoreType {
    kNonAverage,
    kAverage,
};

template <typename CScore, typename TScore> class Score {
    template <typename, typename, typename> friend class PackedScoreMap;

  private:
    CScore current_score_;
    TScore total_score_;

    int last_update_;

    void UpdateCurrent(int value, int round) {
        if (round > last_update_)
            UpdateAverage(round);
        current_score_ += value;
        // In this round, current_core_ has increased by value
        total_score_ += value;
    }

    void UpdateAverage(int round) {
        if (round > last_update_) {
            total_score_ += static_cast<TScore>(current_score_ * (round - last_update_));
            last_update_ = round;
        }
    }

  public:
    Score(const CScore &c = 0, const TScore &t = 0)
        : current_score_(c), total_score_(t), last_update_(0){};

    bool Empty() const { return current_score_ == 0 && total_score_ == 0 && last_update_ == 0; }

    bool Zero() const { return current_score_ == 0 && total_score_ == 0; }

    bool operator==(const Score &s) const {
        return total_score_ == s.total_score_ && current_score_ == s.current_score_;
    }

    void Reset() {
        current_score_ = 0;
        total_score_ = 0;
        last_update_ = 0;
    }

    void AddScore(TScore &return_value, ScoreType type) {
        switch (type) {
        case kNonAverage:
            return_value += static_cast<TScore>(current_score_);
            break;
        case kAverage:
            return_value += total_score_;
            break;
        }
    }

    friend std::istream &operator>>(std::istream &is, Score &s) {
        char c;
        return is >> s.current_score_ >> c >> s.total_score_;
    }

    friend std::ostream &operator<<(std::ostream &os, const Score &s) {
        return os << s.current_score_ << " / " << s.total_score_;
    }
};

template <typename FEATURE_TYPE, typename UPDATE_TYPE = DefaultCScore,
          typename RETURN_TYPE = DefaultTScore>
class PackedScoreMap {
    using SCORE_TYPE = Score<UPDATE_TYPE, RETURN_TYPE>;

  private:
    std::unordered_map<FEATURE_TYPE, SCORE_TYPE> scores_map_;
    std::string name_;
    int count_;

  public:
    PackedScoreMap(const std::string &name) : name_(name), count_(0){};

    bool HasFeature(const FEATURE_TYPE &feature) const {
        return scores_map_.find(feature) != scores_map_.end();
    }

    void GetScore(RETURN_TYPE &return_value, const FEATURE_TYPE &feature, ScoreType type) {
        if (scores_map_.find(feature) != scores_map_.end())
            scores_map_[feature].AddScore(return_value, type);
    }

    void UpdateScore(const FEATURE_TYPE &feature, const UPDATE_TYPE &amount, int round) {
        scores_map_[feature].UpdateCurrent(amount, round);
    }

    void Clear() {
        for (auto &score : scores_map_)
            score.second.reset();
    }

    void ComputeAverage(int round) {
        count_ = 0;
        for (auto &score : scores_map_) {
            score.second.UpdateAverage(round);
            if (!score.second.Zero())
                ++count_;
        }
    }

    bool operator==(const PackedScoreMap &m) const { return scores_map_ == m.scores_map_; }

    friend std::istream &operator>>(std::istream &is, PackedScoreMap &psm) {
        SCORE_TYPE score;
        FEATURE_TYPE feature;
        char c;

        psm.scores_map_.clear();
        is >> psm.name_ >> psm.count_;
        for (int i = 0; i < psm.count_; ++i) {
            is >> feature >> c >> score;
            psm.scores_map_[feature] = score;
        }
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const PackedScoreMap &psm) {
        os << psm.name_ << " " << psm.count_ << '\n';
        for (auto &score : psm.scores_map_) {
            if (!score.second.Zero())
                os << score.first << " : " << score.second << '\n';
        }
        return os;
    }
};

} // namespace perceptron

// Local Variables:
// mode: c++
// End:
