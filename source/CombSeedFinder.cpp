// CombSeedFinder.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <array>
#include <map>
#include <functional>
#include <memory>
#include <vector>
#include <random>

#include "numcomb.h"
#include <thread>

static const std::array<std::vector<int32_t>, 3> k_points{
    std::vector<int32_t>{3, 4, 8},
    std::vector<int32_t>{1, 5, 9},
    std::vector<int32_t>{2, 6, 7}
};



std::vector<std::string> found_seeds;
decltype(found_seeds)::iterator it2_;

void cout(std::string o) {
    std::cout << o;
}
int main()
{
    for (int s = 36278711; s <= 36278711; ++s) {
        //std::cout << "turn " + std::to_string(s) + "\n";
        std::vector<comb::AreaCard> cards_;
        decltype(cards_)::iterator it_;
        //std::thread(std::system, "cls").join();
        //std::thread(cout, std::to_string(s)).join();
        
        //for (it2_ = found_seeds.begin(); it2_ != found_seeds.end(); it2_++) {
        //    std::thread(cout, *it2_).join();
        //}
        cards_.clear();
        for (const int32_t point_0 : k_points[0]) {
            for (const int32_t point_1 : k_points[1]) {
                for (const int32_t point_2 : k_points[2]) {
                    // two card for each type
                    cards_.emplace_back(point_0, point_1, point_2);
                    cards_.emplace_back(point_0, point_1, point_2);
                }
            }
        }
        for (uint32_t i = 0; i < 2; ++i) {
            cards_.emplace_back();
        }
        /*
        for(it_ = cards_.begin(); it_ != cards_.end(); it_++) {
            std::cout << (*it_).PointSum() << "\n";
        }*/
        std::string seed_str = "种子未指定";//std::to_string(s);
        //std::string seed_str = std::to_string(s);
        std::seed_seq seed(seed_str.begin(), seed_str.end());
        std::mt19937 g(seed);
        std::shuffle(cards_.begin(), cards_.end(), g);

        it_ = cards_.begin();
        if (std::all_of(cards_.begin(), cards_.begin() + 20,
            [](const comb::AreaCard& card) { return !card.IsWild(); })) {
            it_ += 20;
        }
        uint32_t l = 0;
        uint32_t ll = 0;
        uint32_t lll = 0;
        bool ox = false;
        in:for (uint16_t p = 0; p < 20; it_++, p++) {
            if ((*it_).IsWild()) {
                std::cout << "癞子 \n";
                continue;
            }
            uint32_t point1 = (*it_).Point<comb::Direct::TOP_LEFT>().value();
            uint32_t point2 = (*it_).Point<comb::Direct::VERT>().value();
            uint32_t point3 = (*it_).Point<comb::Direct::TOP_RIGHT>().value();
            std::cout << "" + std::to_string(point1) + "," + std::to_string(point2) + "," + std::to_string(point3) + "\n";
            
            if(l == 0)
            {
                l = point1;
                ll = point2;
                lll = point3;
            }
            else {
                if (point1 != l) {
                    l = -1;
                }
                if (point2 != ll) {
                    ll = -1;
                }
                if (point3 != lll) {
                    lll = -1;
                }
            }
            if (l == -1 && ll == -1 && lll == -1) {
                //break;
            }
            if (p > 10) {
                std::cout << std::to_string(p)+"\n";
                std::string num;
                if (l != -1) {
                    num = std::to_string(l);
                }
                else if (ll != -1) {
                    num = std::to_string(ll);
                }
                else {
                    num = std::to_string(lll);
                }
                //std::cout << "perfect seed:" + seed_str + ", num:" + num + "\n";
            }
        }
    }
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
