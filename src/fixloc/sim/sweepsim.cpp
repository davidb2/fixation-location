// sweepsim: the Birth-death process on any directed graph, plain Monte Carlo in time steps.
//
// Each time step one individual reproduces, chosen with probability proportional to fitness
// (mutant r, wild type 1). Its offspring replaces a uniformly chosen out-neighbor.
// A vertex with no out-neighbors changes nothing.
//
// For every initial vertex u = 0..N-1 the program repeats runs from a single mutant at u
// until K of them end in fixation, or until max_runs runs.
// Each run ending in fixation is written as one record:
//   int32 initial vertex, int32 fixation location, int64 fixation time in time steps,
//   then for each checkpoint time t_k: ceil(N/8) bytes, bit w = 1 if vertex w is a mutant at t_k
//   (all ones if fixation occurred before t_k).
// A text file <out>.meta lists, per initial vertex, the number of runs and of fixations, which
// give the fixation probability. Runs ending in extinction are counted but not recorded.
//
// usage: sweepsim edges.txt r K seed out.bin [max_runs_per_origin] [t1 t2 ...]
// edges.txt: first line N, then one directed edge "a b" per line (undirected = both ways).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Rng {  // xoshiro256**
  uint64_t s[4];
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  explicit Rng(uint64_t seed) { for (auto& v : s) v = splitmix(seed); }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  uint32_t below(uint32_t n) { return (uint32_t)(((__uint128_t)next() * n) >> 64); }
  double unit() { return (next() >> 11) * 0x1.0p-53; }
};

struct Pop {  // two index lists (mutants, wild types) with O(1) moves
  std::vector<int> mut, wild, pos;
  std::vector<uint8_t> type;
  void reset(int N, int origin) {
    mut.clear(); wild.clear(); pos.assign(N, 0); type.assign(N, 0);
    for (int i = 0; i < N; ++i) { if (i == origin) continue; pos[i] = wild.size(); wild.push_back(i); }
    pos[origin] = 0; mut.push_back(origin); type[origin] = 1;
  }
  void flip(int y) {
    auto& from = type[y] ? mut : wild;
    auto& to = type[y] ? wild : mut;
    int last = from.back(); from[pos[y]] = last; pos[last] = pos[y]; from.pop_back();
    pos[y] = to.size(); to.push_back(y); type[y] ^= 1;
  }
};

int main(int argc, char** argv) {
  if (argc < 6) { std::cerr << "usage: sweepsim edges.txt r K seed out.bin [max_runs] [t1 t2 ...]\n"; return 1; }
  std::ifstream in(argv[1]);
  int N; in >> N;
  std::vector<std::vector<int>> out(N);
  int a, b;
  while (in >> a >> b) if (a != b) out[a].push_back(b);
  const double r = atof(argv[2]);
  const long K = atol(argv[3]);
  const uint64_t seed = strtoull(argv[4], nullptr, 10);
  const std::string path = argv[5];
  const long max_runs = argc > 6 ? atol(argv[6]) : K * 100000L;
  std::vector<int64_t> checks;
  for (int i = 7; i < argc; ++i) checks.push_back(atoll(argv[i]));
  const int nb = (N + 7) / 8, C = (int)checks.size();
  const size_t rec = 4 + 4 + 8 + (size_t)C * nb;
  const int64_t step_cap = 4000000000LL;

  std::vector<std::vector<uint8_t>> buf(N);
  std::vector<long> runs(N, 0), fixes(N, 0), capped(N, 0);

#pragma omp parallel for schedule(dynamic, 1)
  for (int u = 0; u < N; ++u) {
    Rng rng(seed * 1000003ULL + (uint64_t)u * 7919ULL + 17);
    Pop P;
    std::vector<uint8_t> snap((size_t)C * nb);
    auto& B = buf[u];
    B.reserve(rec * K);
    while (fixes[u] < K && runs[u] < max_runs) {
      runs[u]++;
      P.reset(N, u);
      int64_t t = 0; int k = 0, last = -1;
      std::fill(snap.begin(), snap.end(), 0);
      while (!P.mut.empty() && !P.wild.empty()) {
        if (k < C && t == checks[k]) {   // state at the start of step t (t steps done)
          for (int w : P.mut) snap[(size_t)k * nb + (w >> 3)] |= (uint8_t)(1u << (w & 7));
          ++k;
        }
        const double m = (double)P.mut.size(), wd = (double)P.wild.size();
        const bool pick_mut = rng.unit() * (r * m + wd) < r * m;
        const auto& L = pick_mut ? P.mut : P.wild;
        const int x = L[rng.below((uint32_t)L.size())];
        const auto& nb_x = out[x];
        ++t;
        if (nb_x.empty()) continue;
        const int y = nb_x[rng.below((uint32_t)nb_x.size())];
        if (P.type[y] != P.type[x]) {
          P.flip(y);
          if (P.wild.empty()) last = y;
        }
        if (t >= step_cap) { capped[u]++; break; }
      }
      if (!P.wild.empty()) continue;      // extinction or step cap
      for (; k < C; ++k) std::memset(&snap[(size_t)k * nb], 0xFF, nb);
      fixes[u]++;
      const size_t off = B.size();
      B.resize(off + rec);
      int32_t uu = u, vv = last; int64_t tt = t;
      std::memcpy(&B[off], &uu, 4); std::memcpy(&B[off + 4], &vv, 4); std::memcpy(&B[off + 8], &tt, 8);
      if (C) std::memcpy(&B[off + 16], snap.data(), snap.size());
    }
  }
  FILE* f = fopen(path.c_str(), "wb");
  for (int u = 0; u < N; ++u) fwrite(buf[u].data(), 1, buf[u].size(), f);
  fclose(f);
  std::ofstream meta(path + ".meta");
  meta << "N " << N << " r " << r << " K " << K << " seed " << seed << " checkpoints";
  for (auto c : checks) meta << " " << c;
  meta << "\norigin runs fixations capped\n";
  for (int u = 0; u < N; ++u) meta << u << " " << runs[u] << " " << fixes[u] << " " << capped[u] << "\n";
  return 0;
}
