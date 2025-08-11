\
/*
 * execution_engine.cpp
 * Minimal skeleton for a low-latency order handling module.
 * Extend with actual price-time matching and native timing.
 */
#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <mutex>

struct Order {
    int id;
    std::string side; // "buy" or "sell"
    double price;
    int quantity;
    bool operator<(const Order& other) const {
        // For priority queues; refine for real matching (price-time, etc.)
        if (side == "buy") return price < other.price;
        return price > other.price;
    }
};

class OrderBook {
public:
    void add(const Order& o) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (o.side == "buy") bids_.push_back(o);
        else asks_.push_back(o);
    }
    size_t nbids() const { return bids_.size(); }
    size_t nasks() const { return asks_.size(); }
private:
    std::vector<Order> bids_, asks_;
    mutable std::mutex mtx_;
};

extern "C" {
    // Example C API you can call via ctypes
    OrderBook* ob_new() { return new OrderBook(); }
    void ob_add(OrderBook* ob, int id, const char* side, double price, int qty) {
        Order o{ id, std::string(side), price, qty };
        ob->add(o);
    }
    size_t ob_nbids(OrderBook* ob) { return ob->nbids(); }
    size_t ob_nasks(OrderBook* ob) { return ob->nasks(); }
    void ob_free(OrderBook* ob) { delete ob; }
}
