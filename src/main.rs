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
LeetCode 918. Maximum Sum Circular Subarray
https://leetcode.com/problems/maximum-sum-circular-subarray/


Given a circular integer array nums of length n, return the maximum possible sum
of a non-empty subarray of nums.


A circular array means the end of the array connects to the beginning of the array.
Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element
of nums[i] is nums[(i - 1 + n) % n].


A subarray may only include each element of the fixed buffer nums at most once.
Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there does not exist i <= k1, k2 <= j
with k1 % n == k2 % n and k1 != k2.


Examples:

Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum = 3.

Input: nums = [5,-3,5]
Output: 10
Explanation: Subarray [5,5] (taking advantage of circular wrap) has maximum sum = 10.

Input: nums = [-3,-2,-3]
Output: -2
Explanation: Subarray [-2] has maximum sum.
*/


struct Solution;


impl Solution {
    pub fn max_subarray_sum_circular(nums: Vec<i32>) -> i32 {
        let mut total_sum = nums[0];

        let mut current_max = nums[0];
        let mut max_so_far  = nums[0];

        let mut current_min = nums[0];
        let mut min_so_far  = nums[0];

        for &x in nums.iter().skip(1) {
            current_max = x.max(current_max + x);
            max_so_far = max_so_far.max(current_max);

            current_min = x.min(current_min + x);
            min_so_far = min_so_far.min(current_min);

            total_sum += x;
        }

        if max_so_far < 0 {
            max_so_far     
        } else {
            max_so_far.max(total_sum - min_so_far)
        }
    }
}


// Tests


fn print_result(case_name: &str, nums: Vec<i32>, expected: i32) {
    let actual = Solution::max_subarray_sum_circular(nums.clone());
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
    println!("\x1b[1m\x1b[35mLeetCode Runner: Maximum Sum Circular Subarray\x1b[0m");

    print_result(
        "Test Case 1: non-wrap best",
        vec![1, -2, 3, -2],
        3,
    );
    print_result(
        "Test Case 2: wrap best",
        vec![5, -3, 5],
        10,
    );
    print_result(
        "Test Case 3: all negative",
        vec![-3, -2, -3],
        -2,
    );
}