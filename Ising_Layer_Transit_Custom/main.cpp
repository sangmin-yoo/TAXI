#include <bits/stdc++.h>
using namespace std;

// Structure to represent a point with coordinates and info value
struct Point {
    double x, y;
    int info;
};

// Function to calculate Euclidean distance between two points
static inline double dist_points(const Point &A, const Point &B) {
    double dx = A.x - B.x;
    double dy = A.y - B.y;
    return sqrt(dx * dx + dy * dy);
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Check for correct usage: now expect two arguments
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <points_input_file> <matrix_input_file>\n";
        return 1;
    }

    string points_input_file = argv[1];
    string matrix_input_file = argv[2];

    // Derive output file name by replacing .in with .out if present in the first input file
    // else append .out
    string output_file;
    {
        size_t pos = points_input_file.rfind(".in");
        if (pos != string::npos && pos + 3 == points_input_file.size()) {
            output_file = points_input_file.substr(0, pos) + ".out";
        } else {
            output_file = points_input_file + ".out";
        }
    }

    // Open points input file
    ifstream points_infile(points_input_file);
    if (!points_infile.is_open()) {
        cerr << "Error: Unable to open points file " << points_input_file << "\n";
        return 1;
    }

    // Read number of points
    int N;
    points_infile >> N;
    if (N % 2 != 0) {
        cerr << "Error: N must be even.\n";
        return 1;
    }

    // Read points data
    vector<string> point_lines(N + 1); // 1-based indexing
    vector<Point> points(N + 1); // 1-based indexing
    {
        string line;
        getline(points_infile, line); // Consume the remaining newline after N

        for (int i = 1; i <= N; i++) {
            if (!getline(points_infile, line)) {
                cerr << "Error: Not enough point data in the points file.\n";
                return 1;
            }
            point_lines[i] = line;
            // Parse the line into x, y, info
            double x, y;
            int info;
            stringstream ss(line);
            ss >> x >> y >> info;
            if (ss.fail()) {
                cerr << "Error: Invalid point format at line " << (i + 1) << ".\n";
                return 1;
            }
            points[i] = Point{x, y, info};
        }
    }
    points_infile.close();

    // Now read the matrix from the second file argument
    ifstream matrix_infile(matrix_input_file);
    if (!matrix_infile.is_open()) {
        cerr << "Error: Unable to open matrix file " << matrix_input_file << "\n";
        return 1;
    }

    int rows, cols;
    if (!(matrix_infile >> rows >> cols)) {
        cerr << "Error: Unable to read matrix dimensions.\n";
        return 1;
    }

    if (rows <= 0 || cols <= 0) {
        cerr << "Error: Invalid matrix dimensions.\n";
        return 1;
    }

    vector<vector<double>> matrix(rows, vector<double>(cols));

    // Read the matrix rows
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (!(matrix_infile >> matrix[r][c])) {
                cerr << "Error: Not enough data to fill the matrix.\n";
                return 1;
            }
        }
    }

    matrix_infile.close();

    // Form pairs: (1,2), (3,4), ..., (N-1,N)
    int pair_count = N / 2;
    vector<pair<int, int>> pairs(pair_count);
    for (int i = 0; i < pair_count; i++) {
        pairs[i] = {2 * i + 1, 2 * i + 2};
    }

    // Construct distance matrix
    // D[i][j] = distance between point i and point j
    // INF for incompatible info values
    const double INF = 1e9;
    vector<vector<double>> D(N + 1, vector<double>(N + 1, INF));

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (i == j) {
                D[i][j] = INF; // No self-loop
                continue;
            }
            // Check if i and j form a pair
            if ((i % 2 == 1 && j == i + 1) || (j % 2 == 1 && i == j + 1)) {
                D[i][j] = 0.0; // Internal pair distance
            }
            else {
                if (points[i].info == points[j].info) {
                    if (matrix[i-1][j-1] == -1)
                        cerr<<i<<" "<<j<<" Incompatible distance value"<<endl;
                    //D[i][j] = dist_points(points[i], points[j]); // Euclidean distance
                    D[i][j] = matrix[i-1][j-1]; // Custom distance
                }
                else {
                    D[i][j] = INF; // Incompatible info values
                }
            }
        }
    }

    // Initialize route
    // route[0] = p1, route[1] = p2 (first pair, fixed)
    // route[N-2] = pN-1, route[N-1] = pN (last pair, fixed)
    // Middle pairs will be shuffled and oriented
    vector<int> route(N, -1);
    // Vector to track orientations (false: (f, s), true: (s, f))
    vector<bool> orientation(pair_count, false);

    // Fixed first pair
    route[0] = pairs[0].first;
    route[1] = pairs[0].second;

    // Fixed last pair
    route[N - 2] = pairs[pair_count - 1].first;
    route[N - 1] = pairs[pair_count - 1].second;

    // Create a list of middle pairs indices (1 to pair_count - 2)
    vector<int> middle_pairs;
    for (int i = 1; i < pair_count - 1; i++) {
        middle_pairs.push_back(i);
    }

    // Shuffle middle pairs
    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
    shuffle(middle_pairs.begin(), middle_pairs.end(), rng);

    // Assign shuffled middle pairs to the route with random orientations
    int current_pos = 2; // Start filling from index 2
    uniform_int_distribution<int> uid(0, 1);
    for (int idx : middle_pairs) {
        bool orient = (uid(rng) == 1);
        orientation[idx] = orient;

        if (!orient) {
            route[current_pos++] = pairs[idx].first;
            route[current_pos++] = pairs[idx].second;
        }
        else {
            route[current_pos++] = pairs[idx].second;
            route[current_pos++] = pairs[idx].first;
        }
    }

    // Function to compute total route cost
    auto compute_cost = [&](const vector<int> &rt) -> double {
        double sum = 0.0;
        for (int i = 0; i + 1 < (int)rt.size(); i++) {
            double c = D[rt[i]][rt[i + 1]];
            if (c >= INF) {
                return INF; // Invalid route due to incompatible info
            }
            sum += c;
        }
        return sum;
    };

    double best_cost = compute_cost(route);
    vector<int> best_route = route;

    // Simulated Annealing Parameters
    double initial_acceptance = 0.30; // 30% initial acceptance probability for worse solutions
    double final_acceptance = 0.01;   // 1% final acceptance probability for worse solutions
    int max_iter = 1000000;            // Maximum number of iterations
    double epsilon = 1e-5;             // Small value to ensure P approaches final_acceptance

    // Calculate decay constant k based on max_iter and desired final acceptance
    // P = final_acceptance + (initial_acceptance - final_acceptance) * exp(-k * iter)
    // We want P(max_iter) ≈ final_acceptance + (initial_acceptance - final_acceptance) * exp(-k * max_iter) = final_acceptance + (initial - final) * epsilon / (initial - final)
    // Thus, exp(-k * max_iter) = epsilon
    double k = log(1.0 / epsilon) / max_iter;

    // Define free indices (middle pairs only)
    // Each middle pair occupies two positions in the route
    // Identify the starting and ending indices for middle pairs
    int start_free = 2;
    int end_free = N - 3; // Last two positions are fixed
    if (end_free < start_free) {
        // No free pairs to swap
        // Output the fixed route with distance
        ofstream outfile(output_file);
        if (!outfile.is_open()) {
            cerr << "Error: Unable to open output file " << output_file << "\n";
            return 1;
        }
        // Write each point in the fixed route
        for (int i = 1; i <= N; i++) {
            outfile << point_lines[route[i - 1]] << "\n";
        }
        // Compute and write the distance
        double total_distance = compute_cost(route);
        // Ensure consistent decimal precision
        outfile << fixed << setprecision(6) << "dist " << total_distance << "\n";
        outfile.close();
        return 0;
    }

    uniform_int_distribution<int> free_dist(start_free, end_free);
    uniform_real_distribution<double> urd(0.0, 1.0);

    for (int iter = 0; iter < max_iter; iter++) {
        // Calculate the current acceptance probability based on iteration using exponential decay
        double P = final_acceptance + (initial_acceptance - final_acceptance) * exp(-k * iter);

        // Select two random free pair starting indices (each pair occupies two positions)
        int a = free_dist(rng);
        int b = free_dist(rng);
        if (a == b) continue; // Ensure distinct pairs

        // Ensure that 'a' and 'b' are even indices (start of a pair)
        if (a % 2 != 0) a--;
        if (b % 2 != 0) b--;

        // Create a copy of the current route
        vector<int> new_route = route;

        // Swap the two pairs
        swap(new_route[a], new_route[b]);
        swap(new_route[a + 1], new_route[b + 1]);

        // Optionally, flip the orientations of the swapped pairs
        bool flip_a = (urd(rng) < 0.5);
        bool flip_b = (urd(rng) < 0.5);
        if (flip_a) swap(new_route[a], new_route[a + 1]);
        if (flip_b) swap(new_route[b], new_route[b + 1]);

        // Compute the new cost
        double new_cost = compute_cost(new_route);

        // Calculate the change in cost
        double delta = new_cost - best_cost;

        // Acceptance criteria:
        // - Always accept if the move improves the solution
        // - Accept worse moves with probability P
        bool accept;
        if (delta < 0) {
            accept = true;
        }
        else {
            double rand_val = urd(rng);
            accept = (rand_val < P);
        }

        if (accept) {
            // Accept the new route
            route = new_route;
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_route = route;
            }
        }
        // Else reject the move and keep the current route
    }

    // Verify if the best route is valid
    if (best_cost >= INF) {
        cerr << "Error: No feasible route found.\n";
        return 1;
    }

    // Output the best route with exact input formatting and append the distance
    ofstream outfile(output_file);
    if (!outfile.is_open()) {
        cerr << "Error: Unable to open output file " << output_file << "\n";
        return 1;
    }

    // Write each point in the best route, identical to input formatting
    for (int i = 0; i < N; i++) {
        outfile << point_lines[best_route[i]] << "\n";
    }

    // Compute the total distance
    double total_distance = compute_cost(best_route);

    // Append the distance in the format: dist <distance>
    outfile << "dist " << total_distance << "\n";

    outfile.close();
    return 0;
}