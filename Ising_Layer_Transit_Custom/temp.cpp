#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    int info;
};

static inline double dist_points(const Point &A, const Point &B) {
    double dx = A.x - B.x;
    double dy = A.y - B.y;
    return sqrt(dx*dx + dy*dy);
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    string input_file = argv[1];
    // Derive output file name by replacing .in with .out if present, else just append .out
    string output_file;
    {
        size_t pos = input_file.rfind(".in");
        if (pos != string::npos && pos + 3 == input_file.size()) {
            output_file = input_file.substr(0, pos) + ".out";
        } else {
            output_file = input_file + ".out";
        }
    }

    ifstream infile(input_file);
    if (!infile.is_open()) {
        cerr << "Error: Unable to open file " << input_file << "\n";
        return 1;
    }

    int N; infile >> N;
    if (N % 2 != 0) {
        cerr << "N must be even.\n";
        return 1;
    }

    vector<Point> points(N+1);
    for (int i = 1; i <= N; i++) {
        infile >> points[i].x >> points[i].y >> points[i].info;
    }
    infile.close();

    int pair_count = N/2;
    // Initial pairs (always (p(2i+1), p(2i+2)))
    // We'll store them in a way that we can reorder them at insertion time if needed.
    vector<pair<int,int>> pairs(pair_count);
    for (int i = 0; i < pair_count; i++) {
        pairs[i] = {2*i+1, 2*i+2};
    }

    vector<int> info_first(pair_count), info_second(pair_count);
    for (int i = 0; i < pair_count; i++) {
        info_first[i] = points[pairs[i].first].info;
        info_second[i] = points[pairs[i].second].info;
    }


    // Map: info_val -> vector of pairs that start with that info (in their defined order)
    unordered_map<int, vector<int>> pairs_by_info;
    for (int i = 0; i < pair_count; i++) {
        // If we were to connect from some point X to this pair's first point, we need info_first[i].
        pairs_by_info[info_first[i]].push_back(i);
    }


    
    // Function to insert a pair into the route considering orientation at insertion:
    // Given the route ends at 'endpoint', we want to insert pair idx:
    // If points[endpoint].info == points[pairs[idx].first].info, we can insert as (first, second).
    // If not, but points[endpoint].info == points[pairs[idx].second].info, we flip the pair to (second, first).
    // If neither matches, we can't insert this pair here.
    auto try_insert_pair = [&](int endpoint, int idx) -> pair<int,int> {
        int f = pairs[idx].first;
        int s = pairs[idx].second;
        int end_info = points[endpoint].info;
        // We want endpoint -> something. We must match info:
        // If end_info == points[f].info, insert as (f,s).
        // If end_info == points[s].info, insert as (s,f).
        // Else, not feasible.
        if (points[f].info == end_info) {
            return make_pair(f, s);
        } else if (points[s].info == end_info) {
            return make_pair(s, f);
        } else {
            return make_pair(-1,-1); // not feasible
        }
    };

    // Build initial route:
    // route: we will store pairs in the order they appear, but after deciding orientation at insertion,
    // we store them in that oriented form.
    // We need to store the route as pairs of points now, since orientation may differ from original definition.
    vector<pair<int,int>> route(pair_count, {-1,-1});
    vector<bool> used(pair_count, false);

    // First pair must start with p1 and second must be p2:
    // The first pair is (p1,p2) by definition, no need to flip.
    route[0] = {1,2};
    used[0] = true;

    // Last pair must end with pN:
    // The last pair is (pN-1, pN). It already ends with pN, no flip needed.
    used[pair_count-1] = true;
    // We'll fill route[1 .. pair_count-2]
    

    int start_endpoint = 2;

    // Backtracking function:
    // i: current position in the route we're trying to fill
    // current_endpoint: the point from which we must connect the next pair
    function<bool(int,int)> build_route = [&](int i, int current_endpoint) {
        if (i == pair_count - 1) {
            // We placed all intermediate pairs. Now we must connect route[pair_count-2] to the last pair.
            int second_last_endpoint = route[i-1].second;
            // last pair index = pair_count-1
            auto oriented = try_insert_pair(second_last_endpoint, pair_count-1);
            if (oriented.first == -1) {
                return false; // can't connect the last pair
            }
            route[i] = oriented;
            return true;
        }

        // Try to find a pair for route[i].
        // We know the needed info matches points[current_endpoint].info
        int needed_info = points[current_endpoint].info;

        // We'll try all pairs that we haven't used yet and are not first or last
        bool success = false;
        // We'll try all pairs c that could possibly match orientation:
        // Actually, we must try all unused pairs and see if they can be inserted.
        for (int c = 1; c < pair_count-1; c++) {
            if (!used[c]) {
                auto oriented = try_insert_pair(current_endpoint, c);
                if (oriented.first == -1) continue; // not feasible orientation

                // If we can insert this pair, do so:
                used[c] = true;
                route[i] = oriented;
                int new_endpoint = route[i].second;

                // Recurse:
                if (build_route(i+1, new_endpoint)) {
                    success = true;
                    break;
                }

                // Backtrack if not successful:
                used[c] = false;
            }
        }

        return success;
    };

    // Run backtracking starting from i=1 (since i=0 is fixed), with start_endpoint = p2
    bool ok = build_route(1, start_endpoint);
    if (!ok) {
        cerr << "Error: No feasible route found even with backtracking.\n";
        return 1;
    }

    // At this point, route is a feasible initial solution.

    // Compute distance of the route:
    auto route_distance = [&](const vector<pair<int,int>> &rt) {
        double sum = 0.0;
        for (int i = 0; i < (int)rt.size()-1; i++) {
            sum += dist_points(points[rt[i].second], points[rt[i+1].first]);
        }
        return sum;
    };

    for (const auto& p : route) {
        std::cout << "(" << p.first << ", " << p.second << ")" << std::endl;
    }

    /*
    double best_dist = route_distance(route);
    auto best_route = route;

    // Simulated Annealing for solution space exploration
    double T = 1000.0;
    double cooling_rate = 0.9995;
    int max_iter = 1000000;
    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> urd(0.0,1.0);
    
    // Adjust approach:
    // We'll store: 
    //   route_pairs[i] = index of the original pair chosen at position i.
    //   route[i] = oriented pair (already stored)
    // With this, we know exactly which pair is at each route position.
    vector<int> route_pairs(pair_count, -1);

    // We know:
    // route[0] = (1,2) comes from pair 0
    route_pairs[0] = 0;
    route_pairs[pair_count-1] = pair_count-1;

    // For i in [1..pair_count-2], we must have recorded chosen pair index:
    {
        // We determined chosen = c at insertion time. Let's fix the code above:
        // Add a temporary vector chosen_pairs during initial build.
    }

    // Let's redo initial build slightly to store route_pairs as well:

    {
        vector<bool> used2(pair_count, false);
        used2[0] = true;
        used2[pair_count-1] = true;
        vector<pair<int,int>> initial_route(pair_count, {-1,-1});
        initial_route[0] = {1,2};
        int curr_end = 2;
        for (int i = 1; i < pair_count-1; i++) {
            int chosen = -1;
            double best_d = 1e9;
            pair<int,int> chosen_oriented = {-1,-1};
            for (int c = 1; c < pair_count-1; c++) {
                if (!used2[c]) {
                    auto oriented = try_insert_pair(curr_end, c);
                    if (oriented.first != -1) {
                        double d = dist_points(points[curr_end], points[oriented.first]);
                        if (d < best_d) {
                            best_d = d;
                            chosen = c;
                            chosen_oriented = oriented;
                        }
                    }
                }
            }
            if (chosen == -1) {
                cerr << "Error: No feasible continuation found.\n";
                return 1;
            }
            initial_route[i] = chosen_oriented;
            used2[chosen] = true;
            curr_end = initial_route[i].second;
            route_pairs[i] = chosen; 
        }

        // Insert last pair:
        {
            auto oriented = try_insert_pair(curr_end, pair_count-1);
            if (oriented.first == -1) {
                cerr << "Error: Unable to connect to the last pair.\n";
                return 1;
            }
            initial_route[pair_count-1] = oriented;
        }

        // Now we have initial_route and route_pairs
        initial_route[0] = {1,2};
        route_pairs[0] = 0;
        route_pairs[pair_count-1] = pair_count-1;

        route = initial_route;
        used = used2; // from initial build
        best_dist = route_distance(route);
        best_route = route;
    }

    // Now we have a consistent initial solution:
    // route: oriented pairs
    // route_pairs: original pair indices
    // used: which pairs are used

    vector<int> pos_in_route(pair_count, -1);
    for (int i = 0; i < pair_count; i++) {
        pos_in_route[route_pairs[i]] = i;
    }

    auto try_move = [&](vector<pair<int,int>> &curr_route, vector<int> &curr_route_pairs, double &curr_dist) {
        if (pair_count <= 2) return false;

        int i = uniform_int_distribution<int>(1, pair_count-2)(rng);
        int removed_pair = curr_route_pairs[i];

        // Remove it
        used[removed_pair] = false;
        pos_in_route[removed_pair] = -1;

        int prev_pair = curr_route_pairs[i-1];
        int next_pair = curr_route_pairs[i+1];
        // prev endpoint = curr_route[i-1].second
        int prev_endpoint = curr_route[i-1].second;
        int next_start = curr_route[i+1].first;

        double old_d = dist_points(points[prev_endpoint], points[curr_route[i].first])
                     + dist_points(points[curr_route[i].second], points[next_start]);

        int needed_info = points[prev_endpoint].info;

        vector<int> candidates;
        vector<double> candidate_deltas;
        vector<pair<int,int>> candidate_orients;
        for (int c = 1; c < pair_count-1; c++) {
            if (!used[c]) {
                auto oriented = try_insert_pair(prev_endpoint, c);
                if (oriented.first == -1) continue;
                // Check that oriented.second connects to next_start:
                // We need points[oriented.second].info == points[next_start].info to ensure feasibility
                if (points[oriented.second].info == points[next_start].info) {
                    double new_d = dist_points(points[prev_endpoint], points[oriented.first])
                                 + dist_points(points[oriented.second], points[next_start]);
                    double delta = new_d - old_d;
                    candidates.push_back(c);
                    candidate_deltas.push_back(delta);
                    candidate_orients.push_back(oriented);
                }
            }
        }

        if (candidates.empty()) {
            // revert
            used[removed_pair] = true;
            pos_in_route[removed_pair] = i;
            return false;
        }

        double sum_w = 0.0;
        for (auto d : candidate_deltas) {
            double w = (d < 0) ? 1.0 : exp(-d/T);
            sum_w += w;
        }

        double pick_val = uniform_real_distribution<double>(0.0, sum_w)(rng);
        double prefix = 0.0;
        int chosen_idx = -1;
        for (int idx = 0; idx < (int)candidates.size(); idx++) {
            double w = (candidate_deltas[idx]<0) ? 1.0 : exp(-candidate_deltas[idx]/T);
            prefix += w;
            if (prefix >= pick_val) {
                chosen_idx = idx;
                break;
            }
        }
        if (chosen_idx == -1) chosen_idx = (int)candidates.size()-1;

        int chosen_pair = candidates[chosen_idx];
        double chosen_delta = candidate_deltas[chosen_idx];
        auto chosen_oriented = candidate_orients[chosen_idx];

        // Accept move:
        curr_route[i] = chosen_oriented;
        curr_route_pairs[i] = chosen_pair;
        used[chosen_pair] = true;
        pos_in_route[chosen_pair] = i;
        curr_dist += chosen_delta;
        return true;
    };

    double curr_dist = best_dist;
    for (int iter = 0; iter < max_iter; iter++) {
        bool moved = try_move(route, route_pairs, curr_dist);
        if (moved && curr_dist < best_dist) {
            best_dist = curr_dist;
            best_route = route;
        }
        T *= cooling_rate;
        if (T < 0.01) break;
    }

    // Output the best solution
    ofstream outfile(output_file);
    if (!outfile.is_open()) {
        cerr << "Error: unable to open output file " << output_file << "\n";
        return 1;
    }

    for (auto &pr : best_route) {
        outfile << points[pr.first].x << " " << points[pr.first].y << " " << points[pr.first].info << "\n";
        outfile << points[pr.second].x << " " << points[pr.second].y << " " << points[pr.second].info << "\n";
    }

    outfile.close();
    */

    return 0;
}