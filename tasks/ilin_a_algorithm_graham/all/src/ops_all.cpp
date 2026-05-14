#include "ilin_a_algorithm_graham/all/include/ops_all.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <mpi.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include "ilin_a_algorithm_graham/common/include/common.hpp"

namespace ilin_a_algorithm_graham {

namespace {
double Orient(const Point& p, const Point& q, const Point& r) {
  return ((q.x - p.x) * (r.y - p.y)) - ((q.y - p.y) * (r.x - p.x));
}

double DistanceSq(const Point& p, const Point& q) {
  double dx = p.x - q.x;
  double dy = p.y - q.y;
  return (dx * dx) + (dy * dy);
}

Point FindLowestLeftmost(const std::vector<Point>& points) {
  Point p0 = points[0];
  for (size_t i = 1; i < points.size(); ++i) {
    if (points[i].y < p0.y || (points[i].y == p0.y && points[i].x < p0.x)) {
      p0 = points[i];
    }
  }
  return p0;
}

Point FindLowestLeftmostParallel(const std::vector<Point>& points) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, points.size()), points[0],
      [&](const tbb::blocked_range<size_t>& r, Point init) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          if (points[i].y < init.y || (points[i].y == init.y && points[i].x < init.x)) {
            init = points[i];
          }
        }
        return init;
      },
      [](const Point& a, const Point& b) {
        if (a.y < b.y || (a.y == b.y && a.x < b.x)) {
          return a;
        }
        return b;
      });
}

class PointComparator {
 public:
  explicit PointComparator(const Point& p0) : p0_(p0) {}
  
  bool operator()(const Point& a, const Point& b) const {
    double o = Orient(p0_, a, b);
    if (o != 0.0) {
      return o > 0;
    }
    return DistanceSq(p0_, a) < DistanceSq(p0_, b);
  }

 private:
  Point p0_;
};

void GrahamScan(const std::vector<Point>& sorted, const Point& p0, std::vector<Point>& hull) {
  hull.reserve(sorted.size() + 1);
  hull.push_back(p0);
  hull.push_back(sorted[0]);

  for (size_t i = 1; i < sorted.size(); ++i) {
    while (hull.size() >= 2) {
      Point p = hull[hull.size() - 2];
      Point q = hull[hull.size() - 1];
      if (Orient(p, q, sorted[i]) <= 0.0) {
        hull.pop_back();
      } else {
        break;
      }
    }
    hull.push_back(sorted[i]);
  }
}
}  // namespace

IlinAGrahamALL::IlinAGrahamALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool IlinAGrahamALL::ValidationImpl() {
  return !GetInput().points.empty();
}

bool IlinAGrahamALL::PreProcessingImpl() {
  points_ = GetInput().points;
  hull_.clear();
  return true;
}

bool IlinAGrahamALL::RunImpl() {
  int rank = -1;
  int size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (points_.size() < 3) {
    hull_ = points_;
    return true;
  }

  Point p0 = FindLowestLeftmostParallel(points_);
  
  Point global_p0 = p0;
  MPI_Allreduce(&p0, &global_p0, sizeof(Point) / sizeof(double), MPI_DOUBLE, MPI_MINLOC, MPI_COMM_WORLD);

  std::vector<Point> sorted;
  sorted.reserve(points_.size());
  for (const Point& p : points_) {
    if (p.x != global_p0.x || p.y != global_p0.y) {
      sorted.push_back(p);
    }
  }
  
  tbb::parallel_sort(sorted.begin(), sorted.end(), PointComparator(global_p0));

  int local_count = static_cast<int>(sorted.size());
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  int total_count = 0;
  for (int i = 0; i < size; ++i) {
    displs[i] = total_count;
    total_count += counts[i];
  }
  
  std::vector<Point> all_points(total_count);
  MPI_Allgatherv(sorted.data(), local_count, MPI_BYTE,
                 all_points.data(), counts.data(), displs.data(), MPI_BYTE,
                 MPI_COMM_WORLD);

  std::vector<Point> global_sorted;
  global_sorted.reserve(total_count);
  
  PointComparator cmp(global_p0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < counts[i]; ++j) {
      global_sorted.push_back(all_points[displs[i] + j]);
    }
  }
  
  tbb::parallel_sort(global_sorted.begin(), global_sorted.end(), cmp);

  std::vector<Point> hull;
  GrahamScan(global_sorted, global_p0, hull);

  if (rank == 0) {
    hull_ = std::move(hull);
  }
  
  MPI_Bcast(&hull_, sizeof(hull_), MPI_BYTE, 0, MPI_COMM_WORLD);
  
  return true;
}

bool IlinAGrahamALL::PostProcessingImpl() {
  if (hull_.empty()) {
    return false;
  }
  GetOutput() = hull_;
  return true;
}

}  // namespace ilin_a_algorithm_graham