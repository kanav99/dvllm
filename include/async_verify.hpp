#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include "verification.hpp"

// Job payload with deep-copied data
struct VerifierJob {
    int layer;
    int ID;
    std::vector<integer_t> qx;
    std::vector<integer_t> qy;
};

class AsyncVerifier {
private:
    Verifier* verifier_;
    std::queue<VerifierJob> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    bool stop_flag_ = false;

    void worker_loop() {
        while (true) {
            VerifierJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return !queue_.empty() || stop_flag_; });

                if (stop_flag_ && queue_.empty()) {
                    break;
                }

                job = std::move(queue_.front());
                queue_.pop();
            }

            // Run Eat asynchronously on the background worker thread
            verifier_->Eat(job.layer, job.ID, job.qx.data(), job.qx.size(), job.qy.data(), job.qy.size());
        }
    }

public:
    AsyncVerifier(Verifier* v) : verifier_(v) {
        worker_thread_ = std::thread(&AsyncVerifier::worker_loop, this);
    }

    ~AsyncVerifier() {
        finish_and_stop();
    }

    // Main thread enqueues job and immediately returns
    void push(int layer, int ID, const integer_t* qx, int n, const integer_t* qy, int d) {
        VerifierJob job;
        job.layer = layer;
        job.ID = ID;
        job.qx.assign(qx, qx + n); // Deep copy vector x
        job.qy.assign(qy, qy + d); // Deep copy vector y

        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(job));
        }
        cv_.notify_one();
    }

    // Blocks until all remaining enqueued jobs are processed before verification
    void finish_and_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_flag_) return;
            stop_flag_ = true;
        }
        cv_.notify_one();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
};
