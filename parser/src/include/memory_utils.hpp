#pragma once

#include <cassert>
#include <memory>
#include <vector>
#include <cstdint>
// #include "basic.hpp"

namespace utils {

namespace detail {

template <typename T, bool = std::is_trivially_destructible<T>::value> struct DestroyAux {
    template <typename Iterator, typename Size>
    static inline void DestroyN(Iterator first, Size count) {
        for (; count > 0; (void)++first, --count)
            reinterpret_cast<T *>(&*first)->~T();
    }
    template <typename Iterator> static inline void Destroy(Iterator first) {
        reinterpret_cast<T *>(&*first)->~T();
    }
};

template <typename T> struct DestroyAux<T, true> {
    template <typename Iterator, typename Size>
    static inline void DestroyN(Iterator first, Size count) {}
    template <typename Iterator> static inline void Destroy(Iterator first) {}
};

} // namespace detail

// Below code is come from https://en.cppreference.com/w/cpp/types/aligned_storage
template <typename T, int POOL_WIDTH = 18> class MemoryPool {
    static const std::uint32_t POOL_SIZE = 1 << POOL_WIDTH;
    static const std::uint32_t OFFSET_MASK = POOL_SIZE - 1;

    using Block = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

  private:
    std::vector<Block *> pools_;
    std::uint32_t pool_offset_; // block offset in current pool
    std::uint32_t pool_index_;  // index of current pool
    // max block offset in data.back(). The content of data.back()[max_offset_] is uninitialized
    std::uint32_t max_offset_;

    bool MaybeAllocate() {
        if (pool_offset_ == POOL_SIZE) {
            pool_index_++;
            if (pool_index_ == pools_.size()) {
                pools_.push_back(new Block[POOL_SIZE]);
                max_offset_ = 0;
                // LOG_WARN("Pool#" << pools_.size() << " {" << TYPE_NAME(T) << "} ");
            }
            pool_offset_ = 0;
        }

        if (pool_index_ == pools_.size() - 1) {
            assert(pool_offset_ <= max_offset_); // out of range
            if (pool_offset_ == max_offset_) {
                ++max_offset_;
                return true;
            }
        }
        return false;
    }

    void Destroy() {
        for (Block *data_ptr : pools_) {
            detail::DestroyAux<T>::DestroyN(data_ptr,
                                            (data_ptr == pools_.back() ? max_offset_ : POOL_SIZE));
            delete[] data_ptr;
        }
    }

  public:
    MemoryPool() : pools_{new Block[POOL_SIZE]}, pool_offset_(0), pool_index_(0), max_offset_(0) {}

    MemoryPool(const MemoryPool &other) = delete;
    MemoryPool(MemoryPool &&other) = delete;

    ~MemoryPool() { Destroy(); }

    MemoryPool &operator=(const MemoryPool &other) = delete;
    MemoryPool &operator=(MemoryPool &&other) = delete;

    void Clear() {
        pool_index_ = 0;
        pool_offset_ = 0;
    }

    void Reset() {
        Destroy();
        pools_ = {new Block[POOL_SIZE]};
        pool_index_ = pool_offset_ = max_offset_ = 0;
    }

    std::size_t PoolSize() const { return pool_index_ + 1; }
    std::size_t Capacity() const { return pools_.size() << POOL_WIDTH; }
    std::size_t Size() const { return (pool_index_ << POOL_WIDTH) | pool_offset_; }

    T *operator[](std::size_t index) {
        return reinterpret_cast<T *>(
            &pools_[index >> POOL_WIDTH][index & ((1ULL << POOL_WIDTH) - 1)]);
    }

    void Pop() {
        // mark pool_offset_ as unused block
        if (pool_offset_ == 0) {
            --pool_index_;
            pool_offset_ = POOL_SIZE - 1;
        } else
            pool_offset_--;
    }

    template <typename... Args> T *Push(Args... args) {
        bool is_new_memory = MaybeAllocate();
        T *ptr = reinterpret_cast<T *>(&pools_[pool_index_][pool_offset_++]);
        if (!is_new_memory)
            detail::DestroyAux<T>::Destroy(ptr);
        return new (ptr) T(std::forward<Args>(args)...);
    }
};

} // namespace utils

// Local Variables:
// mode: c++
// End:
