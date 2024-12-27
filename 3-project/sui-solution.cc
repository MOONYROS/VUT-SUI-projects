#include "memusage.h"
#include "search-interface.h"
#include "search-strategies.h"

#include <algorithm>
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

inline void printMemoryLimitExceeded()
{
    const std::string boldRed = "\033[1;31m";
    const std::string reset = "\033[0m";
    std::cerr << boldRed << "Memory limit reached" << reset << std::endl;
}

// Hashovaci funkce - mix ruznych shiftu, oru, nasobeni prvocisly,...
// Dost konzultovano s umelou inteligenci, mela by byt dostatecne unikatni a
// rychla.

std::size_t cardHash(const Card& card)
{
    std::size_t h1 = std::hash<int>()(static_cast<int>(card.color));
    std::size_t h2 = std::hash<int>()(card.value);

    h1 = (h1 << 13) | (h1 >> (sizeof(std::size_t) * 8 - 13));
    return h1 * 31 + h2;
}

std::size_t hash(const SearchState& state)
{
    const GameState& gs = state.state_;
    std::size_t hash = 17;

    for (const auto& home : gs.homes) {
        if (home.topCard().has_value()) {
            std::size_t card_hash = cardHash(*home.topCard());
            hash = (hash << 5) | (hash >> (sizeof(std::size_t) * 8 - 5));
            hash = hash * 31 + card_hash;
        }
    }

    for (const auto& free_cell : gs.free_cells) {
        if (free_cell.topCard().has_value()) {
            std::size_t card_hash = cardHash(*free_cell.topCard());
            hash = (hash << 5) | (hash >> (sizeof(std::size_t) * 8 - 5));
            hash = hash * 31 + card_hash;
        }
    }

    for (const auto& stack : gs.stacks) {
        for (const auto& card : stack.storage()) {
            std::size_t card_hash = cardHash(card);
            hash = (hash << 5) | (hash >> (sizeof(std::size_t) * 8 - 5));
            hash = hash * 31 + card_hash;
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

// Optimalizace BFS - pouziti uzlu, ktery obsahuje pointer na rodice a akci, ze
// ktere byl stav vytvoren. Jednoducha rekonstrukce cesty - poskakani od
// koncoveho stavu smerem k inicialnimu.

struct BFSTreeNode
{
    std::shared_ptr<BFSTreeNode> parent;
    SearchActionPtr action;
    std::shared_ptr<SearchState> state;

    BFSTreeNode(std::shared_ptr<BFSTreeNode> parent,
                SearchActionPtr action,
                std::shared_ptr<SearchState> state)
        : parent(parent),
          action(action),
          state(state)
    {
    }
};

using BFSTreeNodePtr = std::shared_ptr<BFSTreeNode>;

inline std::vector<SearchAction> retrieveSearchPath(BFSTreeNodePtr node)
{
    std::vector<SearchAction> path;
    while (node->action != nullptr) {
        path.push_back(*node->action);
        node = node->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

using ClosedSet = std::unordered_set<SearchStatePtr, SearchStateHash>;
using OpenQueue = std::queue<BFSTreeNodePtr>;

std::vector<SearchAction> BreadthFirstSearch::solve(
    const SearchState& init_state)
{
    ClosedSet closed;
    OpenQueue open;

    SearchStatePtr initPtr = std::make_shared<SearchState>(init_state);
    open.emplace(std::make_shared<BFSTreeNode>(nullptr, nullptr, initPtr));

    std::uint64_t iterationCounter = 0;
    while (!open.empty()) {
        BFSTreeNodePtr currentState = open.front();
        open.pop();

        closed.insert(currentState->state);
        for (const SearchAction& action : currentState->state->actions()) {
            SearchStatePtr nextState = std::make_shared<SearchState>(
                action.execute(*currentState->state));

            if (closed.find(nextState) == closed.end()) {
                BFSTreeNodePtr nextNode = std::make_shared<BFSTreeNode>(
                    currentState,
                    std::make_shared<SearchAction>(action),
                    nextState);

                // Optimalizace - nez vubec dam uzel do open, zkontroluji,
                // jestli uz neni cilovy.
                if (nextState->isFinal()) {
                    return retrieveSearchPath(nextNode);
                }

                open.emplace(nextNode);
            }
        }

        // Mensi optimalizace - pristup do souboru je drahy. Parkrat jsem to
        // zkousel, 100 je cislo, se kterou se mi vzdy podchytilo, kdyz jsem se
        // blizil limitu. (Snad to tak bude fungovat i pri testovani.)
        constexpr int checkInterval = 100;
        constexpr std::size_t fiftyMB = 50 * 1024 * 1024;
        if (++iterationCounter % checkInterval == 0
            && getCurrentRSS() > (this->mem_limit_ - fiftyMB)) {
            printMemoryLimitExceeded();
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

    // closed seznam - podle rady to delam spis jako map
    std::unordered_map<SearchStatePtr,
                       std::vector<SearchAction>,
                       SearchStateHash>
        closed;
    // zasobnik open, dfs chodi po zasobniku (pushuje, popuje)
    std::stack<std::pair<SearchStatePtr, std::vector<SearchAction>>> open;

    // inicializace jako v bfs
    SearchStatePtr init_ptr = std::make_shared<SearchState>(init_state);
    open.push({ init_ptr, {} });

    // dokud neni zasobnik prazdny, tak valim
    while (!open.empty()) {
        // kontrola pameti - pokud se blizim limitu, vracim prazdnou cestu
        if (getCurrentRSS() > lower_mem_limit) {
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
        if (closed.find(currentState) == closed.end()) {
            closed[currentState] = currentPath;

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

// Pro A* stejna optimalizace jako u BFS pomoci uzlu; konstrukce stromu.

struct AStarTreeNode
{
    std::shared_ptr<AStarTreeNode> parent;
    SearchActionPtr action;
    SearchStatePtr state;
    double g;
    double f;

    AStarTreeNode(std::shared_ptr<AStarTreeNode> parent,
                  SearchActionPtr action,
                  SearchStatePtr state,
                  double g,
                  double f)
        : parent(parent),
          action(action),
          state(state),
          g(g),
          f(f)
    {
    }
};

using AStarTreeNodePtr = std::shared_ptr<AStarTreeNode>;

inline std::vector<SearchAction> retrieveSearchPath(AStarTreeNodePtr node)
{
    std::vector<SearchAction> path;
    while (node->action != nullptr) {
        path.push_back(*node->action);
        node = node->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

using PriorityQueue = std::priority_queue<
    AStarTreeNodePtr,
    std::vector<AStarTreeNodePtr>,
    std::function<bool(const AStarTreeNodePtr&, const AStarTreeNodePtr&)>>;

std::vector<SearchAction> AStarSearch::solve(const SearchState& init_state)
{
    // Vsechno stejny jako u BFS, vcetne optimalizaci, jen s tim, ze pouzivame
    // prioritni frontu (max heap), sortujici funkce pod timto komentarem. Jeste
    // se navic pocitaji f-hodnoty.
    auto sort = [](const AStarTreeNodePtr& lhs, const AStarTreeNodePtr& rhs) {
        return lhs->f > rhs->f;
    };

    PriorityQueue open(sort);
    ClosedSet closed;

    SearchStatePtr initState = std::make_shared<SearchState>(init_state);
    open.emplace(std::make_shared<AStarTreeNode>(
        AStarTreeNode { nullptr,
                        nullptr,
                        initState,
                        0,
                        compute_heuristic(init_state, *heuristic_) }));

    std::uint64_t iterationCounter = 0;
    while (!open.empty()) {
        AStarTreeNodePtr currentNode = open.top();
        open.pop();

        closed.insert(currentNode->state);
        for (const SearchAction& action : currentNode->state->actions()) {
            SearchStatePtr nextState = std::make_shared<SearchState>(
                action.execute(*currentNode->state));

            if (closed.find(nextState) == closed.end()) {
                double newG = currentNode->g + 1;
                double newF = newG + compute_heuristic(*nextState, *heuristic_);
                AStarTreeNodePtr nextNode = std::make_shared<AStarTreeNode>(
                    AStarTreeNode { currentNode,
                                    std::make_shared<SearchAction>(action),
                                    nextState,
                                    newG,
                                    newF });

                // Taky stejna optimalizace jako u BFS - kontrola, jestli synove
                // jsou koncovymi uzly nez jdou do open.
                if (nextState->isFinal()) {
                    return retrieveSearchPath(nextNode);
                }
                open.emplace(nextNode);
            }
        }

        // Taky stejna optimalizace jako u BFS.
        constexpr int checkInterval = 100;
        constexpr std::size_t fiftyMB = 50 * 1024 * 1024;
        if (++iterationCounter % checkInterval == 0
            && getCurrentRSS() > (this->mem_limit_ - fiftyMB)) {
            printMemoryLimitExceeded();
            return {};
        }
    }

    return {};
}