#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;

const double kInf = -1.;

const int kN = 121;        // 11 x 11 periodic grid
const int kRows = 11;
const int kCols = 11;
const int kInitialVertex = 0;
const int kFixations = 10'000'000;
const std::vector<double> kRs = {1.0, 2.0, 5.0, 10.0, kInf};

using Graph = std::vector<std::vector<int>>;

Graph MakePeriodicGrid(int rows, int cols) {
  Graph graph(rows * cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int node = i * cols + j;
      int up    = ((i - 1 + rows) % rows) * cols + j;
      int down  = ((i + 1) % rows) * cols + j;
      int left  = i * cols + ((j - 1 + cols) % cols);
      int right = i * cols + ((j + 1) % cols);
      graph[node] = {up, down, left, right};
    }
  }
  return graph;
}

// Returns the last vertex converted on fixation, or nullopt on extinction.
std::optional<int> Trial(const Graph& graph, int v0, double r, std::mt19937& rng) {
  const int n = graph.size();
  std::vector<bool> is_mutant(n, false);
  is_mutant[v0] = true;

  std::vector<int> mutants = {v0};
  std::vector<int> residents;
  residents.reserve(n - 1);
  for (int i = 0; i < n; i++)
    if (i != v0) residents.push_back(i);

  std::vector<int> mutant_pos(n, -1);
  std::vector<int> resident_pos(n, -1);
  mutant_pos[v0] = 0;
  for (int i = 0; i < static_cast<int>(residents.size()); i++)
    resident_pos[residents[i]] = i;

  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  int last_dier = -1;

  while (!mutants.empty() && static_cast<int>(mutants.size()) < n) {
    std::vector<int>& birthing_pop = ([&]() -> std::vector<int>& {
      if (r == kInf) {
        return mutants;
      } else {
        const double km = mutants.size();
        const double p_mutant = r * km / (n + (r - 1.0) * km);
        return uniform(rng) < p_mutant ? mutants : residents;
      }
    })();

    int birther = birthing_pop[std::uniform_int_distribution<int>(0, birthing_pop.size() - 1)(rng)];
    const auto& nbrs = graph[birther];
    int dier = nbrs[std::uniform_int_distribution<int>(0, static_cast<int>(nbrs.size()) - 1)(rng)];

    if (is_mutant[birther] && !is_mutant[dier]) {
      is_mutant[dier] = true;
      mutant_pos[dier] = static_cast<int>(mutants.size());
      mutants.push_back(dier);
      int ri = resident_pos[dier];
      int last_res = residents.back();
      residents[ri] = last_res;
      resident_pos[last_res] = ri;
      residents.pop_back();
      resident_pos[dier] = -1;
      last_dier = dier;
    } else if (!is_mutant[birther] && is_mutant[dier]) {
      is_mutant[dier] = false;
      resident_pos[dier] = static_cast<int>(residents.size());
      residents.push_back(dier);
      int mi = mutant_pos[dier];
      int last_mut = mutants.back();
      mutants[mi] = last_mut;
      mutant_pos[last_mut] = mi;
      mutants.pop_back();
      mutant_pos[dier] = -1;
    }
  }

  if (mutants.empty()) return std::nullopt;
  return last_dier;
}

std::string GetTimestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&t);
  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return ss.str();
}

int main() {
  std::string timestamp = GetTimestamp();
  fs::path out_dir = fs::path("data") / "figure_5_periodic_grid" / timestamp;
  fs::create_directories(out_dir);

  Graph graph = MakePeriodicGrid(kRows, kCols);

  // Store counts for all r values so we can write a combined file.
  std::vector<std::vector<long long>> all_counts(kRs.size(), std::vector<long long>(kN, 0));

  for (int ri = 0; ri < static_cast<int>(kRs.size()); ri++) {
    double r = kRs[ri];
    std::string r_label = (r == kInf) ? "inf" : std::to_string(r);
    std::cout << "r=" << r_label << " ..." << std::flush;

    auto t0 = std::chrono::steady_clock::now();

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<long long>> local_counts(max_threads, std::vector<long long>(kN, 0));

    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      // seed it but also index it..
      // use knuth's method
      std::mt19937 rng(42u + static_cast<unsigned>(tid) * 2654435761u +
                       static_cast<unsigned>(ri) * 999983u);

      #pragma omp for schedule(dynamic, 1024)
      for (int i = 0; i < kFixations; i++) {
        while (true) {
          auto result = Trial(graph, kInitialVertex, r, rng);
          if (result) {
            local_counts[tid][*result]++;
            break;
          }
        }
      }
    }

    for (int t = 0; t < max_threads; t++)
      for (int v = 0; v < kN; v++)
        all_counts[ri][v] += local_counts[t][v];

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done in " << std::fixed << std::setprecision(1) << secs << "s" << std::endl;
  }

  // Write combined CSV.
  {
    std::ofstream f(out_dir / "simulations.csv");
    f << "r,last_vertex,count\n";
    for (int ri = 0; ri < static_cast<int>(kRs.size()); ri++) {
      double r_out = (kRs[ri] == kInf) ? -1.0 : kRs[ri];
      for (int v = 0; v < kN; v++) {
        if (all_counts[ri][v] > 0)
          f << r_out << "," << v << "," << all_counts[ri][v] << "\n";
      }
    }
  }

  std::cout << "periodic_grid done -> " << out_dir << std::endl;
  return 0;
}
