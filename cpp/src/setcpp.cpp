#include "setcpp.h"
#include <algorithm>
#include <numeric>

py::array_t<double> euclidean_cluster(py::array_t<double> data)
{
    // Read in the cloud data
    //pcl::PCDReader reader;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    //reader.read ("table_scene_lms400.pcd", *cloud);
    auto data_r = data.mutable_unchecked<2>();
    int no_points;
    no_points = data_r.shape(0);
    cloud->width    = no_points;
    cloud->height   = 1;
    cloud->is_dense = true;
    cloud->points.resize ((cloud->width) * (cloud->height));

    for(int i=0;i<no_points;i++)
    {
        cloud->points[i].x = data_r(i,0);
        cloud->points[i].y = data_r(i,1);
        cloud->points[i].z = data_r(i,2);
    }

    //std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*
    //std::cout << "PointCloud before filtering has: " << no_points << " data points." << std::endl; //*

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.0001f, 0.0001f, 0.0001f);
    vg.filter (*cloud_filtered);
    //std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (5);
    seg.setDistanceThreshold (0.00001);

    int k=0;
    int cloud_sp = 0;
    int i=0, nr_points = (int) cloud_filtered->points.size ();
    if(1)
    {
        while (cloud_filtered->points.size () > 0.5 * nr_points)  //0.3
        {
            // Segment the largest planar component from the remaining cloud
            seg.setInputCloud (cloud_filtered);
            seg.segment (*inliers, *coefficients);
            if (inliers->indices.size () == 0 )
            {    
                
                //std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                break;
		
            }

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud (cloud_filtered);
            extract.setIndices (inliers);
            extract.setNegative (false);

            // Get the points associated with the planar surface
            extract.filter (*cloud_plane);
            //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

            // Remove the planar inliers, extract the rest
            extract.setNegative (true); //true
            extract.filter (*cloud_f);
            *cloud_filtered = *cloud_f;
            if (cloud_filtered->points.size () == 0 )
            {
                cloud_sp = 1;
                break;
            }
        }
    }

    else if(cloud_sp == 1)
    {
        auto ret_arr = py::array_t<double>(1);	
        auto n_ret = ret_arr.mutable_unchecked<>();
        n_ret(0) = -1;
	return ret_arr;
    }
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.2); // 1cm
    ec.setMinClusterSize (1);
    ec.setMaxClusterSize (30000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
    int j = 0;
    int size_cl = cluster_indices.size();
    //cout<<size_cl;
    //cout<<"here";
    //py::array_t<double>* ret_arr = new py::array_t<double>[100];
    //py::array_t<double>* ret_r = ret_arr.mutable_unchecked<>();
    auto ret_arr = py::array_t<double>((size_cl+1)*500*3);	
    auto n_ret = ret_arr.mutable_unchecked<>();
    std::vector<pcl::PointIndices>::const_iterator it;
    k=0;

    for (it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        //std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        //std::stringstream ss;
        //ss << "cloud_cluster_" << j << ".pcd";
        //writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*

        for (int i=0;i<cloud_cluster->points.size()&&i<500;i++)
        {
            n_ret(k+i*3+0) = cloud_cluster->points[i].x;
            n_ret(k+i*3+1) = cloud_cluster->points[i].y;
            n_ret(k+i*3+2) = cloud_cluster->points[i].z;
        }
        for (int i=cloud_cluster->points.size();i<500;i++)
        {
            n_ret(k+i*3+0) = 0;
            n_ret(k+i*3+1) = 0;
            n_ret(k+i*3+2) = 0;
        }
        //cout<<n_ret(k+i*3+0)<<" "<<n_ret(k+i*3+1)<<" "<<n_ret(k+i*3+2);
        //cout<<"\n"<<j<<"\t";

        j++;
        k = j*500*3;
    }
    return ret_arr;
    //return data;
}

std::vector<float> enforceSmoothness(std::vector<float> data, float smoothness) {
    // Since this is pass by value, the data is already copied. Hence, smoothing can be done in-place.
    for(int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            if (data[i] > data[j])
                data[i] = min(data[i], data[j] + smoothness * abs(float(i - j)));
        }
    }
    return data;
}

// =====================================================================================================================
// region Smoothness DP Abstract class
// =====================================================================================================================

template <typename T>
SmoothnessDP<T>::SmoothnessDP(int C_, float min_range, float max_range, float smoothness, T INF_): C(C_), INF(INF_) {
    float interval = smoothness / DIV_FACTOR;

    R = int((max_range - min_range) / interval) + 1;
    for (int i = 0; i < R; i++)
        ranges[i] = min_range + float(i) * interval;

    std::cout << "SMOOTHNESS-DP INIT: the range interval (" << min_range << ", " << max_range << ") has been divided into "
              << R-1 << " segments of length " << interval << " each." << std::endl;
}

template <typename T>
std::vector<float> SmoothnessDP<T>::dp() {
    // dynamic programming loop: from the last ray to the first
    for (int c = C-1; c >= 0; c--) {
        for (int r = 0; r < R; r++) {
            float range = ranges[r];
            T curr_cost = cost(c, r);

            // last ray
            if (c == C-1) {
                dp_cost[c][r] = curr_cost;
                dp_next_idx[c][r] = -1;
            }
            // any ray except the last ray
            else {
                T next_cost = INF;
                int next_idx = -1;

                int min_next_idx = max(r - DIV_FACTOR, 0);
                int max_next_idx = min(r + DIV_FACTOR, R-1);

                // find the lowest cost neighbor on the next ray
                for (int idx = min_next_idx; idx <= max_next_idx; idx++) {
                    if (dp_cost[c+1][idx] < next_cost) {
                        next_cost = dp_cost[c+1][idx];
                        next_idx = idx;
                    }
                }

                dp_cost[c][r] = curr_cost + next_cost;
                dp_next_idx[c][r] = next_idx;
            }
        }
    }

    T min_cost = INF;
    int curr_idx = -1;

    // find best first idx
    for (int idx = 0; idx < R; idx++) {
        if (dp_cost[0][idx] < min_cost) {
            min_cost = dp_cost[0][idx];
            curr_idx = idx;
        }
    }

    // forward pass to get smoothed ranges
    std::vector<float> outputRanges (C);
    for (int c = 0; c < C; c++) {
        float range = ranges[curr_idx];
        outputRanges[c] = range;

        // update curr_idx
        curr_idx = dp_next_idx[c][curr_idx];
    }

    return outputRanges;
}

template<typename T>
std::vector<float> SmoothnessDP<T>::randomCurtain(const std::string& r_sampling) {
    float MAX_RANGE = ranges[R-1];
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // create target ranges
    std::vector<float> target_ranges(C);
    for (int c = 0; c < C; c++) {
        float u = distribution(gen_);  // uniform random sample in [0, 1]

        float tRange;
        if (r_sampling == "uniform") {
            tRange = u * MAX_RANGE;
        }
        else if (r_sampling == "linear") {
            tRange = std::sqrt(u) * MAX_RANGE;
        }
        else {
            throw std::invalid_argument("r_sampling has to be one of <uniform> or <linear>");
        }

        target_ranges[c] = tRange;
    }

    // best r for current camera ray
    int curr_r = 0;

    // iterate over all rays
    std::vector<float> outputRanges (C);
    for (int c = 0; c < C; c++) {
        const float& target_range = target_ranges[c];

        // for the first camera ray, pick amongst all ranges, else pick only amongst neighbors
        int min_r = (c == 0 ? 0   : max(curr_r - DIV_FACTOR, 0));
        int max_r = (c == 0 ? R-1 : min(curr_r + DIV_FACTOR, R-1));

        // update curr_r for the current ray
        float best_dist = INF_FLOAT;
        for (int r = min_r; r <= max_r; r++) {
            const float& range = ranges[r];
            float dist = abs(range - target_range);
            if (dist < best_dist) {
                curr_r = r;
                best_dist = dist;
            }
        }

        // assign corresponding range to output
        outputRanges[c] = ranges[curr_r];
    }

    return outputRanges;
}

//  endregion
// =====================================================================================================================
// region SmoothnessDPL1Cost
// =====================================================================================================================

SmoothnessDPL1Cost::SmoothnessDPL1Cost(int C_, float min_range, float max_range, float smoothness):
    SmoothnessDP<float>(C_, min_range, max_range, smoothness, INF_FLOAT) {};

float SmoothnessDPL1Cost::cost(int c, int r) {
    const float range = ranges[r];
    const float inputRange = (*inputRanges)[c];
    return abs(range - inputRange);
}

std::vector<float> SmoothnessDPL1Cost::smoothedRanges(std::vector<float> inputRanges_) {
    inputRanges = &inputRanges_;  // set inputRanges
    return dp();  // run dp
}

//  endregion
// =====================================================================================================================
// region SmoothnessDPPairGridCost
// =====================================================================================================================

SmoothnessDPPairGridCost::SmoothnessDPPairGridCost(int C_, float min_range, float max_range, float smoothness):
    SmoothnessDP<Pair>(C_, min_range, max_range, smoothness, Pair(INF_FLOAT, INF_FLOAT)) {};

std::vector<float> SmoothnessDPPairGridCost::getRanges() {
    return std::vector<float> (ranges, ranges + R);
}

Pair operator+ (const Pair& p1, const Pair& p2) {
    return { p1.first + p2.first, p1.second + p2.second };
}

Pair SmoothnessDPPairGridCost::cost(int c, int r) {
    // inputPairGrid from python is an array of type (np.ndarray, dtype=np.float32, shape=(R, C, 2))
    return (*inputPairGrid)[r][c];
}

std::vector<float> SmoothnessDPPairGridCost::smoothedRanges(PairGrid inputPairGrid_) {
    if ((inputPairGrid_.size() != R) || (inputPairGrid_[0].size() != C))
        throw std::runtime_error("SmoothnessDPPairGridCost::smoothedRanges(): Input pair grid is not of size (R, C, 2) ");

    inputPairGrid = &inputPairGrid_;  // set inputPairGrid
    return dp();  // run dp
}

//  endregion
// =====================================================================================================================
// region SmoothnessGreedy
// =====================================================================================================================

SmoothnessGreedy::SmoothnessGreedy(int C_, float minRange_, float maxRange_, float smoothness_) :
    C(C_), minRange(minRange_), maxRange(maxRange_), smoothness(smoothness_) {}

    std::vector<float> SmoothnessGreedy::smoothedRanges(std::vector<float> ranges, std::vector<float> weights) {
    // Since arguments is pass by value, the data is already copied. Hence, smoothing can be done in-place.

    // sort cost in descending order
    // initialize original index locations
    std::vector<int> idxs(weights.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    // sort indexes based in descending order based on cost
    std::sort(idxs.begin(), idxs.end(), [&weights](int i1, int i2) {return weights[i1] > weights[i2];});

    std::set<int> seenIdxs;

    for (int idx: idxs) {
        float range = ranges[idx];

        auto iter = seenIdxs.upper_bound(idx);  // points to element that is strictly greater than idx

        // finding the next idx
        if (iter != seenIdxs.end()) {  // there is a greater idx
            int nextIdx = *iter;
            float deltaRange = smoothness * std::abs(nextIdx - idx);

            float nextRange = ranges[nextIdx];
            range = clamp(range, nextRange - deltaRange, nextRange + deltaRange);
        }

        // finding the prev idx
        if (iter != seenIdxs.begin()) {  //  there is a lesser idx
            int prevIdx = *(--iter);
            float deltaRange = smoothness * std::abs(idx - prevIdx);

            float prevRange = ranges[prevIdx];
            range = clamp(range, prevRange - deltaRange, prevRange + deltaRange);
        }

        ranges[idx] = range;
        seenIdxs.insert(idx);
    }

    return ranges;
}

// endregion
// =====================================================================================================================

