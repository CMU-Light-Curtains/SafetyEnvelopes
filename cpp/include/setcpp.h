#ifndef SETCPP_H
#define SETCPP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <random>
#include "iostream"
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#define DIV_FACTOR 2  // will set the interval to be smoothness/DIV_FACTOR
#define MAX_R 1000
#define MAX_C 1024
#define INF_FLOAT std::numeric_limits<float>::infinity()

typedef std::pair<float, float> Pair;
typedef std::vector<std::vector<Pair>> PairGrid;

namespace py = pybind11;
using namespace std;

template <typename T>
inline T clamp(T x, T lo, T hi) { return std::max(std::min(x, hi), lo); }

py::array_t<double> euclidean_cluster(py::array_t<double> data);

// Enforces heuristic smoothness: the difference in ranges between consecutive camera rays shouldn't exceed a threshold.
// It does so by strictly decreasing the ranges on each camera rays until all smoothness constraints are satisfied.
// It uses a nested for loop and takes O(N^2) time.
// However, I suspect that the same result can be obtained as a special case of SmoothnessGreedy.smoothedRanges() where
// the weights are set equal to the negative of ranges (i.e. the lowest ranges have the highest priority). This would
// take only O(N logN) time.
// TODO: implement enforceSmoothness as a special case of SmoothnessGreedy.smoothedRanges() with weights = -ranges.
std::vector<float> enforceSmoothness(std::vector<float> data, float smoothness);

// =====================================================================================================================
// region Smoothness DP Abstract class
// =====================================================================================================================

template <typename T>  // T represents the type of the cost
class SmoothnessDP {
public:
    SmoothnessDP(int C_, float min_range, float max_range, float smoothness, T INF_);
    std::vector<float> randomCurtain(const std::string& r_sampling);

protected:
    float ranges[MAX_R];
    T dp_cost[MAX_C][MAX_R]{};
    int dp_next_idx[MAX_C][MAX_R]{};

    int R;  // number of ranges
    int C;  // number of camera rays

    const T INF;  // defines infinity for the type of cost
    virtual T cost(int c, int r) = 0;
    std::vector<float> dp();

    std::mt19937 gen_;  // random number generator
};

// explicit template class instantiation
template class SmoothnessDP<float>;
template class SmoothnessDP<Pair>;

// endregion
// =====================================================================================================================
// region SmoothnessDPL1Cost
// =====================================================================================================================

class SmoothnessDPL1Cost: public SmoothnessDP<float> {
public:
    SmoothnessDPL1Cost(int C_, float min_range, float max_range, float smoothness);
    std::vector<float> smoothedRanges(std::vector<float> inputRanges);

private:
    std::vector<float>* inputRanges;

    float cost(int c, int r) override;
};

// endregion
// =====================================================================================================================
// region SmoothnessDPPairGridCost
// =====================================================================================================================

// some functionality for pair
Pair operator+ (const Pair&, const Pair&);

class SmoothnessDPPairGridCost: public SmoothnessDP<Pair> {
public:
    SmoothnessDPPairGridCost(int C_, float min_range, float max_range, float smoothness);
    std::vector<float> getRanges();
    std::vector<float> smoothedRanges(PairGrid inputPairGrid);

private:
    PairGrid* inputPairGrid;

    Pair cost(int c, int r) override;
};

// endregion
// =====================================================================================================================
// region SmoothnessGreedy
// =====================================================================================================================

class SmoothnessGreedy {
public:
    SmoothnessGreedy(int C_, float minRange_, float maxRange_, float smoothness_);
    std::vector<float> smoothedRanges(std::vector<float> ranges, std::vector<float> weights);

private:
    size_t C;
    float minRange, maxRange, smoothness;
};

// endregion
// =====================================================================================================================

#endif
