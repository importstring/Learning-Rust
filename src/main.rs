// Python → Rust cheats (iteration / Vec edition)

// nums = [1, 2, 3]              ->
//   let mut nums: Vec<i32> = vec![1, 2, 3];

// len(nums)                     ->
//   nums.len()

// Access by index: nums[i]      ->
//   nums[i]

// Modify in place: nums[i] += 1 ->
//   nums[i] += 1;

// range(len(nums)) or range(n)  ->
//   for i in 0..n { ... }           // 0 up to n-1

// range(1, len(nums))           ->
//   for i in 1..nums.len() { ... }  // start at 1

// simple loop over indices of Vec ->
//   for i in 0..nums.len() { /* use nums[i] */ }

// Vec::new() (empty vector)     ->
//   let mut v: Vec<i32> = Vec::new();

// push to dynamic array         ->
//   v.push(value);

// clone a Vec (like nums.copy()) ->
//   let mut res = nums.clone();

// iterate with mutable access (like for i in range and updating in place):
//   for i in 1..nums.len() {
//       nums[i] += nums[i - 1];
//   }

/*
LeetCode 1480. Running Sum of 1d Array
https://leetcode.com/problems/running-sum-of-1d-array/

Given an array nums, we define a running sum of an array as:
  running_sum[i] = sum(nums[0] + nums[1] + ... + nums[i])

Return the running sum of nums.

Examples:

Input:  nums = [1, 2, 3, 4]
Output: [1, 3, 6, 10]
Explanation:
  running_sum[0] = 1
  running_sum[1] = 1 + 2 = 3
  running_sum[2] = 1 + 2 + 3 = 6
  running_sum[3] = 1 + 2 + 3 + 4 = 10

Input:  nums = [1, 1, 1, 1, 1]
Output: [1, 2, 3, 4, 5]

Input:  nums = [3, 1, 2, 10, 1]
Output: [3, 4, 6, 16, 17]
*/

struct Solution;

impl Solution {
    pub fn running_sum(nums: Vec<i32>) -> Vec<i32> {
        
    }
}

// Tests

fn print_result(case_name: &str, nums: Vec<i32>, expected: Vec<i32>) {
    let actual = Solution::running_sum(nums.clone());
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
    println!("\x1b[1m\x1b[35mLeetCode Runner: Running Sum\x1b[0m");

    print_result("Test Case 1", vec![1, 2, 3, 4], vec![1, 3, 6, 10]);
    print_result("Test Case 2", vec![1, 1, 1, 1, 1], vec![1, 2, 3, 4, 5]);
    print_result("Test Case 3", vec![3, 1, 2, 10, 1], vec![3, 4, 6, 16, 17]);
}