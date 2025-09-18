#pragma once

#include <mutex>
#include <condition_variable>
#include <vector>
#include <optional>
#include <atomic>
#include <chrono>

/**
 * Overflow policy for circular buffer when full
 */
enum class OverflowPolicy {
    DROP_OLDEST,    // Drop oldest item to make room (default)
    DROP_NEWEST,    // Drop the new item being added
    BLOCK           // Block until space is available
};

/**
 * Thread-safe circular buffer implementation for frame queuing.
 * Optimized for single producer, single consumer scenario.
 */
template<typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity, OverflowPolicy policy = OverflowPolicy::DROP_OLDEST)
        : m_capacity(capacity)
        , m_overflowPolicy(policy)
        , m_buffer(capacity)
        , m_head(0)
        , m_tail(0)
        , m_size(0)
        , m_itemsDropped(0)
        , m_itemsAdded(0)
        , m_itemsRemoved(0)
        , m_shutdown(false) {
    }

    ~CircularBuffer() {
        Shutdown();
    }

    /**
     * Add an item to the buffer.
     * Behavior depends on overflow policy when buffer is full.
     *
     * @param item Item to add
     * @param timeoutMs Maximum time to wait if policy is BLOCK (0 = infinite)
     * @return true if item was added, false if dropped or timeout
     */
    bool Push(T item, uint32_t timeoutMs = 0) {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_shutdown) {
            return false;
        }

        m_itemsAdded++;

        // Handle full buffer based on policy
        if (m_size >= m_capacity) {
            switch (m_overflowPolicy) {
                case OverflowPolicy::DROP_OLDEST:
                    // Drop oldest item
                    m_head = (m_head + 1) % m_capacity;
                    m_size--;
                    m_itemsDropped++;
                    break;

                case OverflowPolicy::DROP_NEWEST:
                    // Drop the new item
                    m_itemsDropped++;
                    return false;

                case OverflowPolicy::BLOCK:
                    // Wait for space
                    if (timeoutMs > 0) {
                        auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(timeoutMs);
                        if (!m_notFull.wait_until(lock, deadline, [this] {
                            return m_size < m_capacity || m_shutdown;
                        })) {
                            m_itemsDropped++;
                            return false; // Timeout
                        }
                    } else {
                        m_notFull.wait(lock, [this] {
                            return m_size < m_capacity || m_shutdown;
                        });
                    }

                    if (m_shutdown) {
                        return false;
                    }
                    break;
            }
        }

        // Add item to buffer
        m_buffer[m_tail] = std::move(item);
        m_tail = (m_tail + 1) % m_capacity;
        m_size++;

        // Notify consumer
        m_notEmpty.notify_one();

        return true;
    }

    /**
     * Try to add an item without blocking.
     *
     * @param item Item to add
     * @return true if added, false if buffer full or shutdown
     */
    bool TryPush(T item) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_shutdown || m_size >= m_capacity) {
            if (m_overflowPolicy == OverflowPolicy::DROP_OLDEST && !m_shutdown) {
                // Drop oldest item
                m_head = (m_head + 1) % m_capacity;
                m_size--;
                m_itemsDropped++;
            } else if (m_overflowPolicy == OverflowPolicy::DROP_NEWEST) {
                m_itemsDropped++;
                return false;
            } else {
                return false;
            }
        }

        // Add item to buffer
        m_buffer[m_tail] = std::move(item);
        m_tail = (m_tail + 1) % m_capacity;
        m_size++;
        m_itemsAdded++;

        // Notify consumer
        m_notEmpty.notify_one();

        return true;
    }

    /**
     * Remove and return an item from the buffer.
     * Blocks until an item is available or timeout.
     *
     * @param timeoutMs Maximum time to wait (0 = infinite)
     * @return Optional containing item if available
     */
    std::optional<T> Pop(uint32_t timeoutMs = 0) {
        std::unique_lock<std::mutex> lock(m_mutex);

        // Wait for item or shutdown
        if (timeoutMs > 0) {
            auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
            if (!m_notEmpty.wait_until(lock, deadline, [this] {
                return m_size > 0 || m_shutdown;
            })) {
                return std::nullopt; // Timeout
            }
        } else {
            m_notEmpty.wait(lock, [this] {
                return m_size > 0 || m_shutdown;
            });
        }

        if (m_shutdown && m_size == 0) {
            return std::nullopt;
        }

        // Remove item from buffer
        T item = std::move(m_buffer[m_head]);
        m_head = (m_head + 1) % m_capacity;
        m_size--;
        m_itemsRemoved++;

        // Notify producer if using BLOCK policy
        if (m_overflowPolicy == OverflowPolicy::BLOCK) {
            m_notFull.notify_one();
        }

        return item;
    }

    /**
     * Try to remove and return an item without blocking.
     *
     * @return Optional containing item if available
     */
    std::optional<T> TryPop() {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_size == 0) {
            return std::nullopt;
        }

        // Remove item from buffer
        T item = std::move(m_buffer[m_head]);
        m_head = (m_head + 1) % m_capacity;
        m_size--;
        m_itemsRemoved++;

        // Notify producer if using BLOCK policy
        if (m_overflowPolicy == OverflowPolicy::BLOCK) {
            m_notFull.notify_one();
        }

        return item;
    }

    /**
     * Peek at the oldest item without removing it.
     *
     * @return Pointer to oldest item or nullptr if empty
     */
    const T* Peek() const {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_size == 0) {
            return nullptr;
        }

        return &m_buffer[m_head];
    }

    /**
     * Clear all items from the buffer.
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_head = 0;
        m_tail = 0;
        m_size = 0;

        // Clear the actual items (important for reference counted types)
        for (auto& item : m_buffer) {
            item = T{};
        }

        // Notify producer if using BLOCK policy
        if (m_overflowPolicy == OverflowPolicy::BLOCK) {
            m_notFull.notify_all();
        }
    }

    /**
     * Shutdown the buffer, waking all waiting threads.
     * After shutdown, all operations will fail.
     */
    void Shutdown() {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_shutdown = true;
        m_notEmpty.notify_all();
        m_notFull.notify_all();
    }

    /**
     * Get the current number of items in the buffer.
     */
    size_t Size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_size;
    }

    /**
     * Get the maximum capacity of the buffer.
     */
    size_t Capacity() const {
        return m_capacity;
    }

    /**
     * Check if the buffer is empty.
     */
    bool IsEmpty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_size == 0;
    }

    /**
     * Check if the buffer is full.
     */
    bool IsFull() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_size >= m_capacity;
    }

    /**
     * Check if the buffer has been shutdown.
     */
    bool IsShutdown() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_shutdown;
    }

    /**
     * Get buffer statistics.
     */
    struct Stats {
        uint64_t itemsAdded;
        uint64_t itemsRemoved;
        uint64_t itemsDropped;
        size_t currentSize;
        size_t capacity;
        OverflowPolicy policy;
    };

    Stats GetStats() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return Stats{
            m_itemsAdded.load(),
            m_itemsRemoved.load(),
            m_itemsDropped.load(),
            m_size,
            m_capacity,
            m_overflowPolicy
        };
    }

    /**
     * Reset statistics counters.
     */
    void ResetStats() {
        m_itemsAdded = 0;
        m_itemsRemoved = 0;
        m_itemsDropped = 0;
    }

    /**
     * Set the overflow policy.
     * Note: Changing policy while buffer is in use may have unexpected effects.
     */
    void SetOverflowPolicy(OverflowPolicy policy) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_overflowPolicy = policy;
    }

private:
    const size_t m_capacity;
    OverflowPolicy m_overflowPolicy;

    mutable std::mutex m_mutex;
    std::condition_variable m_notEmpty;
    std::condition_variable m_notFull;

    std::vector<T> m_buffer;
    size_t m_head;  // Index of oldest item
    size_t m_tail;  // Index where next item will be added
    size_t m_size;  // Current number of items

    std::atomic<uint64_t> m_itemsAdded;
    std::atomic<uint64_t> m_itemsRemoved;
    std::atomic<uint64_t> m_itemsDropped;

    bool m_shutdown;
};