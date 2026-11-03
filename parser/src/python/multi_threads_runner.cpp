#include <algorithm>
#include <numeric>

#include "multi_threads_runner.hpp"

namespace shrg {

namespace py = pybind11;

Runner::Runner(Manager &manager, bool verbose)
    : manager_(manager), stop_(false), verbose_(verbose) {
    auto num_contexts = manager.contexts.size();

    graph_indices_.resize(num_contexts, -1);
    for (uint context_index = 0; context_index < num_contexts; ++context_index) {
        workers_.emplace_back([this, context_index] {
            auto &context = manager_.contexts[context_index];
            int graph_index = -1;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(main_mutex_);
                    condition_.wait(lock, [this, context_index] {
                        return stop_ || graph_indices_[context_index] != -1;
                    });
                    graph_index = graph_indices_[context_index];
                    graph_indices_[context_index] = -1;
                    if (stop_ && graph_index == -1)
                        return;
                    // LOG_INFO("thread " << graph_index << " parsing...");
                }
                if (graph_index == -1)
                    continue;
                results_[context_index].set_value(context->Parse(graph_index));
            }
        });
        if (verbose_)
            LOG_INFO("thread " << workers_.back().get_id() << " started");
    }
}

Runner::~Runner() {
    {
        std::unique_lock<std::mutex> lock(main_mutex_);
        stop_ = true;
    }

    condition_.notify_all();
    for (std::thread &worker : workers_) {
        if (verbose_) {
            auto id = worker.get_id();
            worker.join();
            LOG_INFO("thread " << id << " exited.");
        } else
            worker.join();
    }
}

py::list Runner::Run(const std::vector<int> &indices) {
    auto num_tasks = indices.size();
    if (num_tasks > manager_.contexts.size())
        throw std::runtime_error("too many tasks");
    if (!results_.empty())
        throw std::runtime_error("results are not empty");

    std::vector<std::future<ParserError>> futures;
    {
        std::unique_lock<std::mutex> lock(main_mutex_);
        // don't allow enqueueing after stopping the pool
        if (stop_)
            throw std::runtime_error("enqueue on stopped Runner");

        for (uint i = 0; i < num_tasks; ++i) {
            std::promise<ParserError> p;

            futures.emplace_back(p.get_future());
            results_.emplace_back(std::move(p));

            graph_indices_[i] = indices[i];
        }
    }

    condition_.notify_all();

    py::list codes;
    for (auto &future : futures)
        codes.append(future.get());

    results_.clear();
    return codes;
}

} // namespace shrg
