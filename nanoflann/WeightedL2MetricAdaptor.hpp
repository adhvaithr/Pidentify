#include <vector>
#include <cmath>
#include "nanoflann.hpp"

template <
	class T, class DataSource, typename _DistanceType = T,
	typename IndexType = uint32_t>
struct WeightedL2MetricSimpleAdaptor {
	using ElementType = T;
	using DistanceType = _DistanceType;

	const DataSource& data_source;
	const std::vector<double>& weights;

	WeightedL2MetricSimpleAdaptor(const DataSource& _data_source, const std::vector<double>& weights)
		: data_source(_data_source), weights(weights) {};

	inline DistanceType evalMetric(const T* a, const IndexType b_idx, size_t size) const {
		DistanceType result = DistanceType();
		for (size_t i = 0; i < size; ++i) {
			const DistanceType diff = a[i] - data_source.kdtree_get_pt(b_idx, i);
			result += std::pow(diff, 2) * weights[i];
		}

		return result;
	}

	template <typename U, typename V>
	inline DistanceType accum_dist(const U a, const V b, const size_t idx) const {
		return std::pow((a - b), 2) * weights[idx];
	}
};

struct metric_Weighted_L2_Simple : public nanoflann::Metric {
	template <class T, class DataSource, typename IndexType = uint32_t>
	struct traits {
		using distance_t = WeightedL2MetricSimpleAdaptor<T, DataSource, T, IndexType>;
	};
};

template <
	class T, class DataSource, typename _DistanceType = T,
	typename IndexType = uint32_t>
struct WeightedL2MetricAdaptor {
	using ElementType = T;
	using DistanceType = _DistanceType;

	const DataSource& data_source;
	const std::vector<double>& weights;

	WeightedL2MetricAdaptor(const DataSource& _data_source, const std::vector<double>& weights)
		: data_source(_data_source), weights(weights) {};

	DistanceType evalMetric(const T* a, const IndexType b_idx, size_t size,
		DistanceType worst_dist = -1) const {
		DistanceType result = DistanceType();
		const T* last = a + size;
		const T* lastgroup = last - 3;
		const T* first = a;
		size_t d = 0;

		/* Process 4 items with each loop for efficiency. */
		while (a < lastgroup) {
			const DistanceType diff0 = std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++));
			const DistanceType diff1 = std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++));
			const DistanceType diff2 = std::abs(a[2] - data_source.kdtree_get_pt(b_idx, d++));
			const DistanceType diff3 = std::abs(a[3] - data_source.kdtree_get_pt(b_idx, d++));
			result += std::pow(diff0, 2) * weights[&a[0] - first] + std::pow(diff1, 2) * weights[&a[1] - first] +
				std::pow(diff2, 2) * weights[&a[2] - first] + std::pow(diff3, 2) * weights[&a[3] - first];
			a += 4;
			if ((worst_dist > 0) && (result > worst_dist)) { return result; }
		}

		/* Process last 0-3 components. */
		while (a < last) {
			const DistanceType diff0 = *a - data_source.kdtree_get_pt(b_idx, d++);
			result += std::pow(diff0, 2) * weights[a++ - first];
		}

		return result;
	}

	template <typename U, typename V>
	inline DistanceType accum_dist(const U a, const V b, const size_t idx) const {
		return std::pow((a - b), 2) * weights[idx];
	}
};

struct metric_Weighted_L2 : public nanoflann::Metric {
	template <class T, class DataSource, typename IndexType = uint32_t>
	struct traits {
		using distance_t = WeightedL2MetricAdaptor<T, DataSource, T, IndexType>;
	};
};
