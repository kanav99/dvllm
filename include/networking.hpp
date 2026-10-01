#pragma once

#include <arpa/inet.h> 
#include <errno.h> 
#include <netinet/in.h> 
#include <netinet/tcp.h>
#include <signal.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <sys/uio.h>
#include <sys/types.h> 
#include <unistd.h> 

#include <array>
#include <iostream>
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <thread>

const int DEFAULT_PORT = 5065;
const int NUM_PARALLEL_SOCKETS = 4; // Configurable parallel TCP streams

using u64 = unsigned long long;

std::map<int, uint64_t> data_sent;
std::map<int, uint64_t> data_received;
std::map<int, uint64_t> comm_time_in_ms;

// Global mapping for transparent socket multiplexing
std::map<int, std::vector<int>> logical_to_real_sockets;

// Parses "1.2.3.4" or "1.2.3.4:port" into an ip/port pair. Returns false on a
// malformed port so callers can report a usage error.
inline bool parse_endpoint(const std::string &endpoint, std::string &ip, int &port)
{
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos)
    {
        ip = endpoint;
        port = DEFAULT_PORT;
        return !ip.empty();
    }

    ip = endpoint.substr(0, colon);
    std::string port_str = endpoint.substr(colon + 1);
    if (ip.empty() || port_str.empty()) { return false; }
    if (port_str.find_first_not_of("0123456789") != std::string::npos) { return false; }

    long parsed = strtol(port_str.c_str(), NULL, 10);
    if (parsed < 1 || parsed > 65535) { return false; }
    port = (int)parsed;
    return true;
}

inline void tune_socket(int sockfd)
{
    int flag = 1;
    if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) != 0)
    {
        std::cerr << "[networking] Warning: failed to set TCP_NODELAY" << std::endl;
    }
}

// Scalar Struct Read: Route over the primary socket
template <class T>
void read_into(int connfd, T& buffer)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    u64 start = 0;
    int real_fd = logical_to_real_sockets.count(connfd) ? logical_to_real_sockets[connfd][0] : connfd;
    
    while (start < sizeof(buffer))
    {
        int n = read(real_fd, (char *)(&buffer) + start, sizeof(buffer) - start);
        if (n <= 0)
        {
            std::cerr << "[server] Error reading from socket" << std::endl;
            exit(1);
        }
        start += n;
    }
    data_received[connfd] += sizeof(buffer);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    comm_time_in_ms[connfd] += duration.count();
}

// Scalar Struct Write: Route over the primary socket
template <class T>
void write_into(int connfd, const T& buffer)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    u64 start = 0;
    int real_fd = logical_to_real_sockets.count(connfd) ? logical_to_real_sockets[connfd][0] : connfd;

    while (start < sizeof(buffer))
    {
        int n = write(real_fd, (const char*)(&buffer) + start, sizeof(buffer) - start);
        if (n <= 0)
        {
            std::cerr << "[server] Error writing to socket" << std::endl;
            exit(1);
        }
        start += n;
    }
    data_sent[connfd] += sizeof(buffer);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    comm_time_in_ms[connfd] += duration.count();
}

// Byte Array Read: Stripe payload concurrently across all mapped sockets
void read_into(int connfd, char *data, u64 size)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (logical_to_real_sockets.count(connfd) && NUM_PARALLEL_SOCKETS > 1) {
        auto& sockfds = logical_to_real_sockets[connfd];
        std::vector<std::thread> threads;
        u64 base_chunk = size / NUM_PARALLEL_SOCKETS;

        for (int i = 0; i < NUM_PARALLEL_SOCKETS; i++) {
            u64 offset = i * base_chunk;
            u64 length = (i == NUM_PARALLEL_SOCKETS - 1) ? (size - offset) : base_chunk;
            int fd = sockfds[i];

            threads.emplace_back([fd, data, offset, length]() {
                u64 start = 0;
                while (start < length) {
                    int n = read(fd, data + offset + start, length - start);
                    if (n <= 0) {
                        std::cerr << "[server] Error reading from socket in parallel" << std::endl;
                        exit(1);
                    }
                    start += n;
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    } else {
        u64 start = 0;
        int real_fd = logical_to_real_sockets.count(connfd) ? logical_to_real_sockets[connfd][0] : connfd;
        while (start < size)
        {
            int n = read(real_fd, data + start, size - start);
            if (n <= 0)
            {
                std::cerr << "[server] Error reading from socket" << std::endl;
                exit(1);
            }
            start += n;
        }
    }
    
    data_received[connfd] += size;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    comm_time_in_ms[connfd] += duration.count();
}

// Byte Array Write: Stripe payload concurrently across all mapped sockets
void write_into(int connfd, const char *data, u64 size)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (logical_to_real_sockets.count(connfd) && NUM_PARALLEL_SOCKETS > 1) {
        auto& sockfds = logical_to_real_sockets[connfd];
        std::vector<std::thread> threads;
        u64 base_chunk = size / NUM_PARALLEL_SOCKETS;

        for (int i = 0; i < NUM_PARALLEL_SOCKETS; i++) {
            u64 offset = i * base_chunk;
            u64 length = (i == NUM_PARALLEL_SOCKETS - 1) ? (size - offset) : base_chunk;
            int fd = sockfds[i];

            threads.emplace_back([fd, data, offset, length]() {
                u64 start = 0;
                while (start < length) {
                    int n = write(fd, data + offset + start, length - start);
                    if (n <= 0) {
                        std::cerr << "[server] Error writing to socket in parallel" << std::endl;
                        exit(1);
                    }
                    start += n;
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    } else {
        u64 start = 0;
        int real_fd = logical_to_real_sockets.count(connfd) ? logical_to_real_sockets[connfd][0] : connfd;
        while (start < size)
        {
            int n = write(real_fd, data + start, size - start);
            if (n <= 0)
            {
                std::cerr << "[server] Error writing to socket" << std::endl;
                exit(1);
            }
            start += n;
        }
    }

    data_sent[connfd] += size;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    comm_time_in_ms[connfd] += duration.count();
}

void writev_into(int connfd, struct iovec *iov, int iovcnt)
{
    int real_fd = logical_to_real_sockets.count(connfd) ? logical_to_real_sockets[connfd][0] : connfd;
    int idx = 0;
    while (idx < iovcnt)
    {
        ssize_t n = writev(real_fd, iov + idx, iovcnt - idx);
        if (n < 0)
        {
            std::cerr << "[server] Error writing to socket" << std::endl;
            exit(1);
        }

        while (idx < iovcnt && n >= (ssize_t)iov[idx].iov_len)
        {
            n -= iov[idx].iov_len;
            idx++;
        }

        if (idx < iovcnt && n > 0)
        {
            iov[idx].iov_base = (char *)iov[idx].iov_base + n;
            iov[idx].iov_len -= n;
        }
    }
}

template <class F>
void loop(const F callback, int port = DEFAULT_PORT) 
{
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
 
    struct sockaddr_in cliaddr, servaddr; 
    bzero(&servaddr, sizeof(servaddr)); 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port); 

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // binding server addr structure to listenfd 
    int bindstatus = bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)); 
    if (bindstatus != 0)
    {
        printf("port already used\n");
        exit(1);
    }
    
    // Scale connection backlog by number of sockets required per client
    listen(listenfd, 10 * NUM_PARALLEL_SOCKETS); 

    for (;;) { 
        std::vector<int> socks;
        for (int i = 0; i < NUM_PARALLEL_SOCKETS; i++) {
            socklen_t len = sizeof(cliaddr);
            int connfd = accept(listenfd, (struct sockaddr*)&cliaddr, &len); 
            tune_socket(connfd);
            socks.push_back(connfd);
        }
        
        int logical_id = socks[0];
        logical_to_real_sockets[logical_id] = socks;

        std::cerr << "[server] accepted parallel connection (" << NUM_PARALLEL_SOCKETS << " sockets) from " 
                  << inet_ntoa(cliaddr.sin_addr) << std::endl;
        
        callback(logical_id);
        
        std::cerr << "[server] finished connection from " << inet_ntoa(cliaddr.sin_addr) << std::endl;

        for (int fd : socks) {
            close(fd);
        }
        logical_to_real_sockets.erase(logical_id);
    } 
}

int connect(std::string ip = "127.0.0.1", int port = DEFAULT_PORT) 
{ 
    std::vector<int> socks;
    for (int i = 0; i < NUM_PARALLEL_SOCKETS; i++) {
        int sockfd; 
        struct sockaddr_in servaddr; 

        if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) { 
            printf("socket creation failed"); 
            exit(0); 
        }

        tune_socket(sockfd);

        memset(&servaddr, 0, sizeof(servaddr)); 

        // Filling server information 
        servaddr.sin_family = AF_INET; 
        servaddr.sin_port = htons(port); 
        servaddr.sin_addr.s_addr = inet_addr(ip.c_str()); 

        if (::connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) { 
            printf("\n Error : Connect Failed \n"); 
            exit(1);
        }
        socks.push_back(sockfd);
    }

    int logical_id = socks[0];
    logical_to_real_sockets[logical_id] = socks;
    return logical_id;
}