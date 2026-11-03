#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <pybind11/pybind11.h>
#include <thread>

#include "../manager.hpp"

namespace shrg {

class Runner {
  public:
    Runner(Manager &manager, bool verbose);

    ~Runner();

    pybind11::list Run(const std::vector<int> &graph_indices);

  private:
    Manager &manager_;

    std::vector<std::thread> workers_;
    std::vector<int> graph_indices_;
    std::vector<std::promise<ParserError>> results_;

    // synchronization
    std::mutex main_mutex_;
    std::condition_variable condition_;
    bool stop_;
    bool verbose_;
};

} // namespace shrg
