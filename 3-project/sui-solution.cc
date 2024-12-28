#include "memusage.h"
#include "search-interface.h"
#include "search-strategies.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using SearchStatePtr = std::shared_ptr<SearchState>;
using SearchActionPtr = std::shared_ptr<SearchAction>;
using PathToState = std::vector<SearchActionPtr>;

std::size_t cardHash(const Card& card)
{
    std::size_t h1 = std::hash<int>()(static_cast<int>(card.color));
    std::size_t h2 = std::hash<int>()(card.value);

    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
}

std::size_t hash(const SearchState& state)
{
    const GameState& gs = state.state_;
    std::size_t hash = 0;

    for (const auto& home : gs.homes) {
        if (home.topCard().has_value()) {
            hash ^= cardHash(*home.topCard());
        }
    }

    for (const auto& free_cell : gs.free_cells) {
        if (free_cell.topCard().has_value()) {
            hash ^= cardHash(*free_cell.topCard());
        }
    }

    for (const auto& stack : gs.stacks) {
        for (const auto& card : stack.storage()) {
            hash ^= cardHash(card);
        }
    }

    return hash;
}

struct SearchStateHash
{
    std::size_t operator()(const SearchStatePtr& s) const
    {
        return hash(*s);
    }
};

bool operator==(const SearchState& lhs, const SearchState& rhs)
{
    return lhs.state_ == rhs.state_;
}

using ClosedSet = std::unordered_set<SearchStatePtr, SearchStateHash>;
using OpenQueue = std::queue<std::pair<SearchStatePtr, PathToState>>;

std::vector<SearchAction> BreadthFirstSearch::solve(
    const SearchState& init_state)
{
    ClosedSet closed;
    OpenQueue open;

    SearchStatePtr initPtr = std::make_shared<SearchState>(init_state);
    open.emplace(std::make_pair(initPtr, PathToState {}));

    std::uint64_t iterationCounter = 0;
    while (!open.empty()) {
        auto [currentState, currentPath] = open.front();
        open.pop();

        closed.insert(currentState);
        for (const SearchAction& action : currentState->actions()) {
            SearchStatePtr nextState =
                std::make_shared<SearchState>(action.execute(*currentState));

            if (closed.find(nextState) == closed.end()) {
                PathToState nextPath = currentPath;
                nextPath.push_back(std::make_shared<SearchAction>(action));

                if (nextState->isFinal()) {
                    std::vector<SearchAction> path;
                    for (const auto& actionPtr : nextPath) {
                        path.push_back(*actionPtr);
                    }
                    return path;
                }

                open.emplace(std::make_pair(nextState, nextPath));
            }
        }

        constexpr int checkInterval = 100;
        constexpr std::size_t fiftyMB = 50 * 1024 * 1024;
        if (++iterationCounter % checkInterval == 0
            && getCurrentRSS() > (this->mem_limit_ - fiftyMB)) {
            return {};
        }
    }
    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState& init_state)
{
    std::size_t depth_limit = this->depth_limit_;

    // vlastni limit pro pamet - 50MB pod hard limitem
    std::size_t lower_mem_limit = this->mem_limit_ - 50 * 1024 * 1024;

    // pocitadla na kontrolu pameti
    constexpr int checkInterval = 100;
    int iterationCounter = 0;

    // closed seznam - podle rady to delam spis jako map
    std::unordered_set<SearchStatePtr, SearchStateHash> closed;
    // zasobnik open, dfs chodi po zasobniku (pushuje, popuje)
    std::stack<std::pair<SearchStatePtr, std::vector<SearchAction>>> open;

    // inicializace jako v bfs
    SearchStatePtr init_ptr = std::make_shared<SearchState>(init_state);
    open.push({ init_ptr, {} });

    // dokud neni zasobnik prazdny, tak valim
    while (!open.empty()) {
        // kontrola pameti - pokud se blizim limitu, vracim prazdnou cestu
        if (++iterationCounter % checkInterval == 0
            && getCurrentRSS() > lower_mem_limit) {
            return {};
        }

        // vezmu si vrsek zasobniku
        auto [currentState, currentPath] = open.top();
        open.pop();

        // pokud jsem v cili, tak koncim a vracim cestu
        if (currentState->isFinal()) {
            return currentPath;
        }

        // pokud jsem presahl hloubku, tak pokracuji dal
        if (currentPath.size() > depth_limit) {
            continue;
        }

        // zkontroluji, jestli jsem uz stav videl (je v closed)
        if (closed.insert(currentState).second) {
            // projdu kazdou akci, kterou muzu udelat
            for (const SearchAction& action : currentState->actions()) {
                // vytvorim novy stav
                SearchStatePtr nextState = std::make_shared<SearchState>(
                    action.execute(*currentState));

                // zkontroluji, jestli uz stav neni v closed mape
                if (closed.find(nextState) == closed.end()) {
                    // vytvotim si novou cestu
                    auto nextPath = currentPath;
                    nextPath.push_back(action);

                    // a vlozim na zasobnik stav i cestu
                    open.push({ nextState, nextPath });
                }
            }
        }
    }

    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState& state) const
{
    return 0;
}

struct AStarState
{
    SearchStatePtr state;
    double g;
    double f;
};

using PriorityQueue = std::priority_queue<
    std::pair<AStarState, PathToState>,
    std::vector<std::pair<AStarState, PathToState>>,
    std::function<bool(const std::pair<AStarState, PathToState>,
                       const std::pair<AStarState, PathToState>)>>;

std::vector<SearchAction> AStarSearch::solve(const SearchState& init_state)
{
    auto sort = [](const std::pair<AStarState, PathToState>& lhs,
                   const std::pair<AStarState, PathToState>& rhs) {
        return lhs.first.f > rhs.first.f;
    };

    PriorityQueue open(sort);
    ClosedSet closed;

    SearchStatePtr init_ptr = std::make_shared<SearchState>(init_state);
    double h = compute_heuristic(init_state, *heuristic_);
    open.emplace(std::make_pair(AStarState { init_ptr, 0, h }, PathToState {}));

    std::uint64_t iterationCounter = 0;
    while (!open.empty()) {
        auto [currentNode, currentPath] = open.top();
        open.pop();

        if (currentNode.state->isFinal()) {
            std::vector<SearchAction> path;
            for (const auto& actionPtr : currentPath) {
                path.push_back(*actionPtr);
            }
            return path;
        }

        closed.insert(currentNode.state);
        for (const SearchAction& action : currentNode.state->actions()) {
            SearchStatePtr nextState = std::make_shared<SearchState>(
                action.execute(*currentNode.state));

            if (closed.find(nextState) == closed.end()) {
                PathToState nextPath = currentPath;
                nextPath.push_back(std::make_shared<SearchAction>(action));
                double g = currentNode.g + 1;
                double f = g + compute_heuristic(*nextState, *heuristic_);
                open.emplace(
                    std::make_pair(AStarState { nextState, g, f }, nextPath));
            }
        }

        constexpr int checkInterval = 1000;
        constexpr std::size_t fiftyMB = 50 * 1024 * 1024;
        if (++iterationCounter % checkInterval == 0
            && getCurrentRSS() > (this->mem_limit_ - fiftyMB)) {
            return {};
        }
    }

    return {};
}
