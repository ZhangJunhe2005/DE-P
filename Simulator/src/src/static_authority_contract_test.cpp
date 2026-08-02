#include "sensor_simulator.cuh"

#include <cstdint>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace raycast;

template <typename T>
void readExact(std::ifstream &stream, T *value, size_t count)
{
    stream.read(
        reinterpret_cast<char *>(value),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) throw std::runtime_error("truncated query file");
}

template <typename T>
void writeExact(std::ofstream &stream, const T *value, size_t count)
{
    stream.write(
        reinterpret_cast<const char *>(value),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) throw std::runtime_error("failed to write result file");
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: static_authority_contract_test "
                  << "ARTIFACT QUERIES RESULTS\n";
        return 2;
    }
    try {
        size_t free_before = 0, total_memory = 0, free_after = 0;
        cudaMemGetInfo(&free_before, &total_memory);
        const auto load_start = std::chrono::steady_clock::now();
        GridMap map(argv[1]);
        const auto load_stop = std::chrono::steady_clock::now();
        cudaMemGetInfo(&free_after, &total_memory);
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open query file");
        std::uint64_t count = 0;
        readExact(input, &count, 1);
        std::vector<StaticSphereQuery> queries(count);
        readExact(input, queries.data(), queries.size());
        const auto cpu_start = std::chrono::steady_clock::now();
        const auto cpu = map.queryStaticSphereCollisionBatchCpu(queries);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const auto gpu_start = std::chrono::steady_clock::now();
        const auto gpu = map.queryStaticSphereCollisionBatchGpu(queries);
        const auto gpu_stop = std::chrono::steady_clock::now();
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open result file");
        writeExact(output, &count, 1);
        writeExact(output, cpu.data(), cpu.size());
        writeExact(output, gpu.data(), gpu.size());
        output.flush();
        map.freeGridMap();
        const auto milliseconds = [](auto start, auto stop) {
            return std::chrono::duration<double, std::milli>(
                stop - start).count();
        };
        std::cout << "STATIC_AUTHORITY_CONTRACT_RESULT {"
                  << "\"count\":" << count
                  << ",\"map_hash\":\"" << map.authorityHash() << "\""
                  << ",\"load_and_upload_ms\":"
                  << milliseconds(load_start, load_stop)
                  << ",\"gpu_memory_bytes\":" << (free_before - free_after)
                  << ",\"cpu_batch_ms\":"
                  << milliseconds(cpu_start, cpu_stop)
                  << ",\"gpu_batch_ms\":"
                  << milliseconds(gpu_start, gpu_stop)
                  << "}\n";
    } catch (const std::exception &error) {
        std::cerr << "STATIC_AUTHORITY_CONTRACT_ERROR "
                  << error.what() << "\n";
        return 1;
    }
    return 0;
}
