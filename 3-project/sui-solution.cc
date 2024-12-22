#include "search-interface.h"
#include "search-strategies.h"

#include <cstddef>
#include <queue>
#include <unordered_set>
#include <vector>

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
        std::size_t h = 0;

        for (const auto& home : gs.homes) {
            if (home.topCard().has_value()) {
                h ^= CardHash()(*home.topCard());
            }
        }

        for (const auto& free_cell : gs.free_cells) {
            h ^= FreeCellHash()(free_cell);
        }

        for (const auto& stack : gs.stacks) {
            h ^= WorkStackHash()(stack);
        }

        return h;
    }
};

std::size_t hash(const SearchState& state)
{
    return GameStateHash()(state.state_);
}

struct SearchStateHash
{
    std::size_t operator()(const SearchState& s) const
    {
        return hash(s);
    }
};

bool operator==(const SearchState& lhs, const SearchState& rhs)
{
    return lhs.state_ == rhs.state_;
}

std::vector<SearchAction> BreadthFirstSearch::solve(
    const SearchState& init_state)
{
    std::unordered_set<SearchState, SearchStateHash> closed;
    std::queue<SearchState> open;
    std::vector<SearchAction> solution;

    open.push(init_state);  // enqueue

    while (!open.empty()) {
        SearchState current = open.front();
        open.pop();  // dequeue
        if (current.isFinal()) {
            // TODO: not sure how to work with SearchAction yet
            return {};
        }

        closed.insert(current);
        for (const auto& action : current.actions()) {
            SearchState next = action.execute(current);

            // if not in closed
            if (closed.find(next) == closed.end()) {
                open.push(next);
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
