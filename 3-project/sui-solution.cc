#include "search-interface.h"
#include "search-strategies.h"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using SearchStatePtr = std::shared_ptr<SearchState>;
using SearchActionPtr = std::shared_ptr<SearchAction>;

struct CardHash
{
    std::size_t operator()(const Card& card) const
    {
        std::size_t h1 = std::hash<int>()(static_cast<int>(card.color));
        std::size_t h2 = std::hash<int>()(card.value);

        // Nalezeno na internetech jako unikatni a ryhla hashovaci funkce
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct FreeCellHash
{
    std::size_t operator()(const FreeCell& fc) const
    {
        if (fc.topCard().has_value()) {
            return CardHash()(*fc.topCard());
        }
        return 0;
    }
};

struct WorkStackHash
{
    std::size_t operator()(const WorkStack& ws) const
    {
        std::size_t h = 0;
        for (const auto& card : ws.storage()) {
            h ^= CardHash()(card);
        }
        return h;
    }
};

struct GameStateHash
{
    std::size_t operator()(const GameState& gs) const
    {
        std::size_t hash = 0;

        for (const auto& home : gs.homes) {
            if (home.topCard().has_value()) {
                hash ^= CardHash()(*home.topCard());
            }
        }

        for (const auto& free_cell : gs.free_cells) {
            hash ^= FreeCellHash()(free_cell);
        }

        for (const auto& stack : gs.stacks) {
            hash ^= WorkStackHash()(stack);
        }

        return hash;
    }
};

std::size_t hash(const SearchState& state)
{
    return GameStateHash()(state.state_);
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

std::vector<SearchAction> BreadthFirstSearch::solve(
    const SearchState& init_state)
{
    std::unordered_set<SearchStatePtr, SearchStateHash> closed;
    std::unordered_map<SearchStatePtr,
                       std::pair<SearchStatePtr, SearchActionPtr>,
                       SearchStateHash>
        statePredecessors;
    std::queue<SearchStatePtr> open;
    std::vector<SearchAction> solution;

    SearchStatePtr init_ptr = std::make_shared<SearchState>(init_state);
    open.push(init_ptr);

    while (!open.empty()) {
        SearchStatePtr currentState = open.front();
        open.pop();

        if (currentState->isFinal()) {
            while (statePredecessors.find(currentState)
                   != statePredecessors.end()) {
                auto [previousState, action] = statePredecessors[currentState];
                solution.push_back(*action);
                currentState = previousState;
            }
            std::reverse(solution.begin(), solution.end());
            return solution;
        }

        closed.insert(currentState);
        for (const SearchAction& action : currentState->actions()) {
            SearchStatePtr nextState =
                std::make_shared<SearchState>(action.execute(*currentState));

            if (closed.find(nextState) == closed.end()) {
                open.push(nextState);
                SearchActionPtr action_ptr =
                    std::make_shared<SearchAction>(action);
                statePredecessors[nextState] =
                    std::make_pair(currentState, action_ptr);
            }
        }
    }
    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState& init_state)
{
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState& state) const
{
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState& init_state)
{
    return {};
}
