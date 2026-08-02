#include "sensor_simulator.cuh"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>

namespace {
std::string fileSha256(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot hash " + path);
    SHA256_CTX context;
    SHA256_Init(&context);
    char buffer[1 << 16];
    while (stream.good()) {
        stream.read(buffer, sizeof(buffer));
        SHA256_Update(&context, buffer, stream.gcount());
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char value : digest)
        output << std::setw(2) << static_cast<int>(value);
    return output.str();
}
}

namespace raycast
{   
    GridMap::GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold = 1){
        const float epsilon = 0.001f;   // 避免数值误差导致 (1)建图空行 (2)边缘点被忽略
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        float length = max_pt(0) - min_pt(0) + 2 * epsilon;  // 保证各个边界最大值能被取到
        float width  = max_pt(1) - min_pt(1) + 2 * epsilon;
        float height = max_pt(2) - min_pt(2) + 2 * epsilon;
        Vector3f origin(min_pt(0), min_pt(1), min_pt(2));
        Vector3f map_size(length, width, height);
        origin_x_ = origin.x;
        origin_y_ = origin.y;
        origin_z_ = origin.z;

        Vector3i grid_size;
        grid_size.x = ceil(map_size.x / resolution);
        grid_size.y = ceil(map_size.y / resolution);
        grid_size.z = ceil(map_size.z / resolution);
        int grid_total_size = grid_size.x * grid_size.y * grid_size.z;

        resolution_   = resolution;
        grid_size_x_  = grid_size.x, 
        grid_size_y_  = grid_size.y, 
        grid_size_z_  = grid_size.z, 
        grid_size_yz_ = grid_size.y * grid_size.z;
        occupy_threshold_ = occupy_threshold;
        raycast_step_ = resolution;

        std::vector<int> h_map(grid_total_size, 0);
        // 点云全位于体素边界，有时候会有全空的行，加个很小的偏移
        for (size_t i = 0; i < cloud->points.size(); i++) {
            Vector3f point(cloud->points[i].x + epsilon, cloud->points[i].y + epsilon, cloud->points[i].z + epsilon);
            int idx = Vox2Idx(Pos2Vox(point));
            if (idx < grid_total_size) {
                h_map[idx]++;
            }
        }
        map_cpu_ = new int[grid_total_size];
        std::copy(h_map.begin(), h_map.end(), map_cpu_);
        std::vector<int> occupied_indices;
        for (int index = 0; index < grid_total_size; ++index)
            if (h_map[index] > occupy_threshold_)
                occupied_indices.push_back(index);
        occupied_count_ = static_cast<int>(occupied_indices.size());
        occupied_indices_cpu_ = new int[occupied_count_];
        std::copy(
            occupied_indices.begin(), occupied_indices.end(),
            occupied_indices_cpu_);
        cudaMalloc((void **)&map_cuda_, grid_total_size * sizeof(int));
        cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(
            (void **)&occupied_indices_cuda_,
            occupied_count_ * sizeof(int));
        cudaMemcpy(
            occupied_indices_cuda_, occupied_indices.data(),
            occupied_count_ * sizeof(int), cudaMemcpyHostToDevice);
    }

    GridMap::GridMap(const std::string &authority_artifact)
    {
        YAML::Node metadata = YAML::LoadFile(
            authority_artifact + "/occupancy_metadata.json");
        if (metadata["authority_version"].as<std::string>()
            != "static_geometry_authority_v1")
            throw std::runtime_error("static authority version mismatch");
        if (metadata["storage_order"].as<std::string>()
            != "x_major_y_middle_z_fast"
            || metadata["bit_packing"].as<std::string>() != "lsb_first"
            || metadata["static_collision_oob_policy"].as<std::string>()
               != "occupied_fail_closed")
            throw std::runtime_error("unsupported static authority schema");
        resolution_ = metadata["occupancy_resolution_m"].as<float>();
        if (std::abs(resolution_ - 0.1f) > 1e-7f)
            throw std::runtime_error("static authority resolution mismatch");
        origin_x_ = metadata["occupancy_origin"][0].as<float>();
        origin_y_ = metadata["occupancy_origin"][1].as<float>();
        origin_z_ = metadata["occupancy_origin"][2].as<float>();
        grid_size_x_ = metadata["occupancy_dimensions"][0].as<int>();
        grid_size_y_ = metadata["occupancy_dimensions"][1].as<int>();
        grid_size_z_ = metadata["occupancy_dimensions"][2].as<int>();
        grid_size_yz_ = grid_size_y_ * grid_size_z_;
        occupy_threshold_ = 0;
        raycast_step_ = resolution_;
        const std::string authority_hash =
            metadata["artifact_manifest_hash"].as<std::string>();
        if (authority_hash.size() != 64)
            throw std::runtime_error("invalid authority hash");
        std::copy(
            authority_hash.begin(), authority_hash.end(), authority_hash_);
        authority_hash_[64] = '\0';
        canonical_authority_ = true;
        const int total = grid_size_x_ * grid_size_y_ * grid_size_z_;
        std::ifstream stream(
            authority_artifact + "/occupancy.bin", std::ios::binary);
        if (!stream)
            throw std::runtime_error("cannot open canonical occupancy");
        std::vector<unsigned char> packed(
            (std::istreambuf_iterator<char>(stream)),
            std::istreambuf_iterator<char>());
        if (packed.size() != static_cast<size_t>((total + 7) / 8))
            throw std::runtime_error("canonical occupancy size mismatch");
        if (fileSha256(authority_artifact + "/occupancy.bin")
            != metadata["occupancy_hash"].as<std::string>())
            throw std::runtime_error("canonical occupancy hash mismatch");
        std::vector<int> host(total, 0);
        std::vector<int> occupied;
        for (int index = 0; index < total; ++index) {
            host[index] = (packed[index / 8] >> (index % 8)) & 1;
            if (host[index])
                occupied.push_back(index);
        }
        if (static_cast<int>(occupied.size())
            != metadata["occupied_voxel_count"].as<int>())
            throw std::runtime_error("occupied voxel count mismatch");
        map_cpu_ = new int[total];
        std::copy(host.begin(), host.end(), map_cpu_);
        occupied_count_ = static_cast<int>(occupied.size());
        occupied_indices_cpu_ = new int[occupied_count_];
        std::copy(
            occupied.begin(), occupied.end(), occupied_indices_cpu_);
        cudaMalloc((void **)&map_cuda_, total * sizeof(int));
        cudaMemcpy(
            map_cuda_, host.data(), total * sizeof(int),
            cudaMemcpyHostToDevice);
        cudaMalloc(
            (void **)&occupied_indices_cuda_,
            occupied_count_ * sizeof(int));
        cudaMemcpy(
            occupied_indices_cuda_, occupied.data(),
            occupied_count_ * sizeof(int), cudaMemcpyHostToDevice);
    }

    void GridMap::freeGridMap()
    {
        if (map_cuda_) cudaFree(map_cuda_);
        if (occupied_indices_cuda_) cudaFree(occupied_indices_cuda_);
        delete[] map_cpu_;
        delete[] occupied_indices_cpu_;
        map_cuda_ = nullptr;
        occupied_indices_cuda_ = nullptr;
        map_cpu_ = nullptr;
        occupied_indices_cpu_ = nullptr;
    }

    __host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f &pos)
    {
        Vector3i vox;
        vox.x = floor((pos.x - origin_x_) / resolution_);
        vox.y = floor((pos.y - origin_y_) / resolution_);
        vox.z = floor((pos.z - origin_z_) / resolution_);
        return vox;
    }

    __host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i &vox)
    {
        Vector3f pos;
        pos.x = (vox.x + 0.5f) * resolution_ + origin_x_;
        pos.y = (vox.y + 0.5f) * resolution_ + origin_y_;
        pos.z = (vox.z + 0.5f) * resolution_ + origin_z_;
        return pos;
    }

    __host__ __device__ int GridMap::Vox2Idx(const Vector3i &vox)
    {
        return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
    }

    __host__ __device__ Vector3i GridMap::Idx2Vox(int idx)
    {
        return Vector3i(idx / grid_size_yz_, (idx % grid_size_yz_) / grid_size_z_, idx % grid_size_z_);
    }

    __device__ int GridMap::symmetricIndex(int index, int length)
    {
        index = index % (2 * length - 2);
        if (index < 0)
        {
            index += (2 * length - 2);
        }

        if (index >= length)
        {
            index = 2 * length - 2 - index;
        }
        return index;
    }

    // -1: z越界; 0: 空闲; 1: 占据
    __device__  int GridMap::mapQuery(const Vector3f &pos){
        Vector3i vox = Pos2Vox(pos);
        vox.x = symmetricIndex(vox.x, grid_size_x_);
        vox.y = symmetricIndex(vox.y, grid_size_y_);

        if (vox.z >= grid_size_z_)
            return 0;
        if (vox.z <= 0)
            return 1;

        int idx = Vox2Idx(vox);
        if (map_cuda_[idx] > occupy_threshold_)
            return 1;
        return 0;        
    }

    StaticCollisionResult GridMap::queryStaticSphereCollision(
        const StaticSphereQuery &query) const
    {
        StaticCollisionResult result{};
        result.voxel_x = result.voxel_y = result.voxel_z = -1;
        result.minimum_gap_m = std::numeric_limits<float>::max();
        double selected_gap = std::numeric_limits<float>::max();
        const double maximum_x =
            origin_x_ + grid_size_x_ * resolution_;
        const double maximum_y =
            origin_y_ + grid_size_y_ * resolution_;
        const double maximum_z =
            origin_z_ + grid_size_z_ * resolution_;
        result.out_of_bounds =
            query.x - query.radius
                < origin_x_ - STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.y - query.radius
                < origin_y_ - STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.z - query.radius
                < origin_z_ - STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.x + query.radius
                > maximum_x + STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.y + query.radius
                > maximum_y + STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.z + query.radius
                > maximum_z + STATIC_AUTHORITY_CONTACT_TOLERANCE_M;
        result.collision = result.out_of_bounds;
        if (query.radius < 0.0 || !std::isfinite(query.radius)
            || !std::isfinite(query.x) || !std::isfinite(query.y)
            || !std::isfinite(query.z)) {
            result.collision = 1;
            result.out_of_bounds = 1;
            return result;
        }
        for (int offset = 0; offset < occupied_count_; ++offset) {
            const int index = occupied_indices_cpu_[offset];
            const int vx = index / grid_size_yz_;
            const int vy = (index % grid_size_yz_) / grid_size_z_;
            const int vz = index % grid_size_z_;
            const double minimum_x = origin_x_ + vx * resolution_;
            const double minimum_y = origin_y_ + vy * resolution_;
            const double minimum_z = origin_z_ + vz * resolution_;
            const double maximum_voxel_x = minimum_x + resolution_;
            const double maximum_voxel_y = minimum_y + resolution_;
            const double maximum_voxel_z = minimum_z + resolution_;
            const double dx = query.x < minimum_x ? minimum_x - query.x
                : (query.x > maximum_voxel_x
                   ? query.x - maximum_voxel_x : 0.0);
            const double dy = query.y < minimum_y ? minimum_y - query.y
                : (query.y > maximum_voxel_y
                   ? query.y - maximum_voxel_y : 0.0);
            const double dz = query.z < minimum_z ? minimum_z - query.z
                : (query.z > maximum_voxel_z
                   ? query.z - maximum_voxel_z : 0.0);
            const double gap = std::sqrt(dx*dx + dy*dy + dz*dz)
                - query.radius;
            if (gap < result.minimum_gap_m)
                result.minimum_gap_m = gap;
            if (gap < selected_gap
                    - STATIC_AUTHORITY_CONTACT_TOLERANCE_M) {
                selected_gap = gap;
                result.voxel_x = vx;
                result.voxel_y = vy;
                result.voxel_z = vz;
                result.bounds_min_x = minimum_x;
                result.bounds_min_y = minimum_y;
                result.bounds_min_z = minimum_z;
                result.bounds_max_x = maximum_voxel_x;
                result.bounds_max_y = maximum_voxel_y;
                result.bounds_max_z = maximum_voxel_z;
            }
        }
        result.queried_voxel_count = occupied_count_;
        if (result.minimum_gap_m <= STATIC_AUTHORITY_CONTACT_TOLERANCE_M)
            result.collision = 1;
        if (!result.collision)
            result.voxel_x = result.voxel_y = result.voxel_z = -1;
        return result;
    }

    __device__ StaticCollisionResult
    GridMap::queryStaticSphereCollisionDevice(
        const StaticSphereQuery &query) const
    {
        StaticCollisionResult result{};
        result.voxel_x = result.voxel_y = result.voxel_z = -1;
        // Exact binary32 maximum promoted to double. This is the versioned
        // empty-space sentinel shared with CPU/offline serialization.
        result.minimum_gap_m = 0x1.fffffep+127;
        double selected_gap = 0x1.fffffep+127;
        const double maximum_x =
            origin_x_ + grid_size_x_ * resolution_;
        const double maximum_y =
            origin_y_ + grid_size_y_ * resolution_;
        const double maximum_z =
            origin_z_ + grid_size_z_ * resolution_;
        result.out_of_bounds =
            query.x - query.radius
                < origin_x_ - STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.y - query.radius
                < origin_y_ - STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.z - query.radius
                < origin_z_ - STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.x + query.radius
                > maximum_x + STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.y + query.radius
                > maximum_y + STATIC_AUTHORITY_CONTACT_TOLERANCE_M
            || query.z + query.radius
                > maximum_z + STATIC_AUTHORITY_CONTACT_TOLERANCE_M;
        result.collision = result.out_of_bounds;
        for (int offset = 0; offset < occupied_count_; ++offset) {
            const int index = occupied_indices_cuda_[offset];
            const int vx = index / grid_size_yz_;
            const int vy = (index % grid_size_yz_) / grid_size_z_;
            const int vz = index % grid_size_z_;
            const double minimum_x = origin_x_ + vx * resolution_;
            const double minimum_y = origin_y_ + vy * resolution_;
            const double minimum_z = origin_z_ + vz * resolution_;
            const double maximum_voxel_x = minimum_x + resolution_;
            const double maximum_voxel_y = minimum_y + resolution_;
            const double maximum_voxel_z = minimum_z + resolution_;
            const double dx = query.x < minimum_x ? minimum_x - query.x
                : (query.x > maximum_voxel_x
                   ? query.x - maximum_voxel_x : 0.0);
            const double dy = query.y < minimum_y ? minimum_y - query.y
                : (query.y > maximum_voxel_y
                   ? query.y - maximum_voxel_y : 0.0);
            const double dz = query.z < minimum_z ? minimum_z - query.z
                : (query.z > maximum_voxel_z
                   ? query.z - maximum_voxel_z : 0.0);
            const double gap = sqrt(dx*dx + dy*dy + dz*dz)
                - query.radius;
            if (gap < result.minimum_gap_m)
                result.minimum_gap_m = gap;
            if (gap < selected_gap
                    - STATIC_AUTHORITY_CONTACT_TOLERANCE_M) {
                selected_gap = gap;
                result.voxel_x = vx;
                result.voxel_y = vy;
                result.voxel_z = vz;
                result.bounds_min_x = minimum_x;
                result.bounds_min_y = minimum_y;
                result.bounds_min_z = minimum_z;
                result.bounds_max_x = maximum_voxel_x;
                result.bounds_max_y = maximum_voxel_y;
                result.bounds_max_z = maximum_voxel_z;
            }
        }
        result.queried_voxel_count = occupied_count_;
        if (result.minimum_gap_m <= STATIC_AUTHORITY_CONTACT_TOLERANCE_M)
            result.collision = 1;
        if (!result.collision)
            result.voxel_x = result.voxel_y = result.voxel_z = -1;
        return result;
    }

    __global__ void staticSphereCollisionKernel(
        StaticCollisionResult *results, GridMap grid_map,
        const StaticSphereQuery *queries, int count)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count)
            results[index] =
                grid_map.queryStaticSphereCollisionDevice(queries[index]);
    }

    std::vector<StaticCollisionResult>
    GridMap::queryStaticSphereCollisionBatchCpu(
        const std::vector<StaticSphereQuery> &queries) const
    {
        std::vector<StaticCollisionResult> results;
        results.reserve(queries.size());
        for (const auto &query : queries)
            results.push_back(queryStaticSphereCollision(query));
        return results;
    }

    std::vector<StaticCollisionResult>
    GridMap::queryStaticSphereCollisionBatchGpu(
        const std::vector<StaticSphereQuery> &queries) const
    {
        if (queries.empty()) return {};
        StaticSphereQuery *device_queries = nullptr;
        StaticCollisionResult *device_results = nullptr;
        cudaMalloc(
            (void **)&device_queries,
            queries.size() * sizeof(StaticSphereQuery));
        cudaMalloc(
            (void **)&device_results,
            queries.size() * sizeof(StaticCollisionResult));
        cudaMemcpy(
            device_queries, queries.data(),
            queries.size() * sizeof(StaticSphereQuery),
            cudaMemcpyHostToDevice);
        const int threads = 128;
        const int blocks =
            (static_cast<int>(queries.size()) + threads - 1) / threads;
        staticSphereCollisionKernel<<<blocks, threads>>>(
            device_results, *this, device_queries,
            static_cast<int>(queries.size()));
        cudaDeviceSynchronize();
        std::vector<StaticCollisionResult> results(queries.size());
        cudaMemcpy(
            results.data(), device_results,
            results.size() * sizeof(StaticCollisionResult),
            cudaMemcpyDeviceToHost);
        cudaFree(device_queries);
        cudaFree(device_results);
        return results;
    }

    __global__ void cameraRaycastKernel(float* depth_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);

        if (u < camera_param.image_width && v < camera_param.image_height)
        {
            // 计算射线方向
            float y = -(u - camera_param.cx) / camera_param.fx;
            float z = -(v - camera_param.cy) / camera_param.fy;
            float x = 1.0f;

            // 归一化射线方向
            float length = sqrtf(x * x + y * y + z * z);
            x /= length;
            y /= length;
            z /= length;

            // 计算每个轴的增量比例 (x方向固定步长避免近距离处畸变; 0.5是瞎设的防止过于稀疏导致错误)
            float dx = 0.5 * grid_map.raycast_step_;
            float dy = (y / x) * dx;
            float dz = (z / x) * dx;

            // 递增射线方向上的每个轴
            int scale = 0;
            float depth = 0.0f;

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                if (occupied == 1)
                {
                    // depth = point_x;  // 直接这样赋值会有一点误差
                    // 栅格化避免平面变曲面 (有些冗余，但在机体系栅格化会有类似摩尔纹的东西)
                    Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                    Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                    float3 occ_point_w_ = make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                    float3 occ_point_c_ = T_wc.inv() * occ_point_w_;
                    depth = occ_point_c_.x;
                    break;
                }

                if (point_x >= camera_param.max_depth_dist){
                    depth = camera_param.max_depth_dist;
                    break;
                }
            }

            // Occupancy is sampled along the ray, but the reported depth is
            // snapped to the occupied voxel centre.  That centre may lie a
            // few centimetres beyond max_depth_dist even when point_x did
            // not.  Enforce the public 32FC1 CameraInfo range on the final
            // value, regardless of which loop exit produced it.
            depth = fminf(camera_param.max_depth_dist, fmaxf(0.0f, depth));
            // 将深度值存储到输出数组中
            if (camera_param.normalize_depth)
                depth = depth / camera_param.max_depth_dist;
            depth_values[v * camera_param.image_width + u] = depth;
        }
    }

    void renderDepthImage(GridMap* grid_map, CameraParams* camera_param, cudaMat::SE3<float>& T_wc, cv::Mat& depth_image)
    {   
        float* depth_values;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));

        // 在GPU上启动核函数
        cameraRaycastKernel<<<camera_param->image_height, camera_param->image_width>>>(depth_values, *grid_map, *camera_param, T_wc);
        
        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);

        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(depth_values);
        return;
    }

    __global__ void lidarRaycastKernel(Vector3f* point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc)
    {
        int h = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);
        if (h < lidar_param.horizontal_num && v < lidar_param.vertical_lines)
        {   
            float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
            float vertical_angle = lidar_param.vertical_angle_start + v * vertical_resolution;
            float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
            float cos_vert = std::cos(vertical_angle * M_PI / 180.0);
            float horizontal_angle = h * lidar_param.horizontal_resolution;
            float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
            float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);
            // 计算射线方向
            Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);

            // 计算每个轴的增量比例
            float dx = ray_direction.x * grid_map.raycast_step_;
            float dy = ray_direction.y * grid_map.raycast_step_;
            float dz = ray_direction.z * grid_map.raycast_step_;

            // 递增射线方向上的每个轴
            int scale = 0;
            Vector3f point_value(0, 0, 0);

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                float ray_length = sqrtf(point_x * point_x + point_y * point_y + point_z * point_z);

                if (occupied == 1)
                {
                    point_value = Vector3f(point_x, point_y, point_z);
                    Vector3i vox_body = grid_map.Pos2Vox(point_value);  // 栅格化避免平面变曲面
                    point_value = grid_map.Vox2Pos(vox_body);
                    break;
                }

                if (ray_length > lidar_param.max_lidar_dist){
                    break;
                }
            }

            // 将点云值存储到输出数组中，(0, 0, 0)为无效值
            point_values[v * lidar_param.horizontal_num + h] = point_value;
        }
    }

    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wc, pcl::PointCloud<pcl::PointXYZ>& lidar_points){
        Vector3f* point_values;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&point_values, num_elements * sizeof(Vector3f));

        // 在GPU上启动核函数
        lidarRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(point_values, *grid_map, *lidar_param, T_wc);
        
        cudaDeviceSynchronize();

        std::vector<Vector3f> cpu_points(num_elements);
        cudaMemcpy(cpu_points.data(), point_values, num_elements * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        
        lidar_points.points.clear();
        lidar_points.points.reserve(num_elements);
        
        for (const auto& point : cpu_points) {
            if (point.x != 0 || point.y != 0 || point.z != 0) {
                lidar_points.points.emplace_back(point.x, point.y, point.z);
            }
        }
        cudaFree(point_values);
        return;
    }


    
}
