#include "memusage.h"
#include "search-interface.h"
#include "search-strategies.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
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
    std::queue<std::pair<SearchStatePtr, std::vector<SearchAction>>> open;

    SearchStatePtr init_ptr = std::make_shared<SearchState>(init_state);
    open.emplace(std::make_pair(init_ptr, std::vector<SearchAction> {}));

    int counter = 0;
    while (!open.empty()) {
        auto [currentState, currentPath] = open.front();
        open.pop();

        if (currentState->isFinal()) {
            return currentPath;
        }

        closed.insert(currentState);
        for (const SearchAction& action : currentState->actions()) {
            SearchStatePtr nextState =
                std::make_shared<SearchState>(action.execute(*currentState));

            if (closed.find(nextState) == closed.end()) {
                std::vector<SearchAction> nextPath = currentPath;
                nextPath.push_back(action);
                open.emplace(std::make_pair(nextState, nextPath));
            }
        }

        constexpr int checkInterval = 100;
        constexpr std::size_t fiftyMB = 50 * 1024 * 1024;
        if (++counter % checkInterval == 0
            && getCurrentRSS() > (this->mem_limit_ - fiftyMB)) {
            return {};
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
