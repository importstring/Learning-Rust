// Python → Rust (iteration with running state)

// running_sum = 0; best = -10**9
// for x in nums:
//     running_sum = max(x, running_sum + x)
//     best = max(best, running_sum)

// In Rust, you'll usually keep mutable variables outside the loop,
// then update them inside:

// let mut current = 0;
// let mut best = nums[0];          // assume nums is non-empty

// for x in nums.iter() {
//     // *x is the i32 value behind &i32
//     current = current + *x;      // or some other update rule
//     if current > best {
//         best = current;
//     }
// }

// iterate by value (taking ownership) instead of by reference:
// for x in nums {
//     // x: i32 here, not &i32
// }

// iterate by index if you need indices:
// for i in 0..nums.len() {
//     let x = nums[i];
// }

// get first element safely if you want to handle empty vecs:
// let first = nums[0];   // panics if empty
// // Idiomatic safe version (if you need it later):
// if let Some(&first) = nums.first() { ... }

/*
LeetCode 53. Maximum Subarray
https://leetcode.com/problems/maximum-subarray/

Given an integer array nums, find the contiguous subarray (containing at least one number)
which has the largest sum, and return its sum.

A subarray is a contiguous part of the array.

Examples:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Input: nums =
Output: 1

Input: nums = [5,4,-1,7,8]
Output: 23
*/

struct Solution;

impl Solution {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut best = nums[0];
        let mut current = nums[0];

        for &x in nums.iter().skip(1) {
            current = (current + x).max(x);
            best.max(current);
        }

        best
    }
}

// Tests

fn print_result(case_name: &str, nums: Vec<i32>, expected: i32) {
    let actual = Solution::max_sub_array(nums.clone());
    let passed = actual == expected;

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Input:{} nums = {:?}", yellow, reset, nums);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mLeetCode Runner: Maximum Subarray\x1b[0m");

    print_result(
        "Test Case 1",
        vec![-2, 1, -3, 4, -1, 2, 1, -5, 4],
        6,
    );
    print_result("Test Case 2", vec![1], 1);
    print_result("Test Case 3", vec![5, 4, -1, 7, 8], 23);
}