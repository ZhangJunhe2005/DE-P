#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  if (argc != 8) return 2;
  const int seed = std::atoi(argv[1]);
  const int count = std::atoi(argv[2]);
  const double x_half = std::atof(argv[3]);
  const double y_half = std::atof(argv[4]);
  const double z_high = std::atof(argv[5]);
  const double width_low = std::atof(argv[6]);
  const double width_high = std::atof(argv[7]);
  std::default_random_engine engine(seed);
  std::uniform_real_distribution<double> x(-x_half, x_half);
  std::uniform_real_distribution<double> y(-y_half, y_half);
  std::uniform_real_distribution<double> width(width_low, width_high);
  std::uniform_real_distribution<double> height(0.0, z_high);
  std::cout << std::setprecision(17);
  for (int index = 0; index < count; ++index) {
    std::cout << index << " " << x(engine) << " " << y(engine) << " "
              << width(engine) << " " << height(engine) << "\n";
  }
  return 0;
}
