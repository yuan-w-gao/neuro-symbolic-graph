#pragma once

#include <algorithm>
#include <utility>

namespace utils {

template <typename ValueType> class SimpleBeam {
  public:
    using iterator = ValueType **;
    using const_iterator = ValueType *const *;
    using size_type = std::size_t;

  private:
    const size_type capacity_;
    size_type size_;
    ValueType **pointers_;
    ValueType *data_;

  public:
    SimpleBeam() : capacity_(0), size_(0) {}

    SimpleBeam(int capacity)
        : capacity_(capacity), size_(0), pointers_(new ValueType *[capacity_]),
          data_(new ValueType[capacity_]) {
        for (std::size_t i = 0; i < capacity_; ++i)
            pointers_[i] = &data_[i];
    }

    SimpleBeam(const SimpleBeam &other)
        : capacity_(other.capacity_), size_(other.size_),
          pointers_(new ValueType *[other.capacity_]), data_(new ValueType[other.capacity_]) {
        for (int i = 0; i < capacity_; ++i) {
            data_[i] = other.data_[i];
            pointers_[i] = &data_[other.pointers_[i] - other.data_];
        }
    }

    SimpleBeam(SimpleBeam &&other)
        : capacity_(other.capacity_), size_(other.size_), pointers_(other.pointers_),
          data_(other.data_) {
        other.data_ = nullptr;
        other.pointers_ = nullptr;
    }

    ~SimpleBeam() {
        delete[] pointers_;
        delete[] data_;
    }

    size_type Size() const { return size_; }
    bool Full() const { return size_ == capacity_; }
    void Clear() { size_ = 0; }

    void Insert(const ValueType &v) {
        auto cmp = [](ValueType *x, ValueType *y) { return *x > *y; };
        if (size_ == capacity_) {
            if (v > *pointers_[0]) {
                std::pop_heap(pointers_, pointers_ + size_, cmp);
                --size_;
            } else
                return;
        }
        *pointers_[size_++] = v;
        std::push_heap(pointers_, pointers_ + size_, cmp);
    }

    ValueType &Best() {
        int best_index = 0;
        for (size_type i = 1; i < size_; ++i) {
            if (*pointers_[i] > *pointers_[best_index]) {
                best_index = i;
            }
        }
        return *pointers_[best_index];
    }

    iterator begin() { return &pointers_[0]; }
    iterator end() { return &pointers_[size_]; }
    const_iterator begin() const { return &pointers_[0]; }
    const_iterator end() const { return &pointers_[size_]; }

    bool CanInsert(const ValueType &v) { return size_ != capacity_ || v > *pointers_[0]; }

    void Sort() {
        std::sort(pointers_, pointers_ + size_, [](ValueType *x, ValueType *y) { return *x > *y; });
    }

    ValueType &ShrinkToOne() {
        size_ = 1;
        return *pointers_[0];
    }

    const ValueType &operator[](int index) { return *pointers_[index]; }
};

} // namespace utils

// Local Variables:
// mode: c++
// End:
